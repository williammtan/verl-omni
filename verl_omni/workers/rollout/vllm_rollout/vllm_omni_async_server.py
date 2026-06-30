# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import ray
import torch
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.utils.net_utils import get_free_port
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.replica import DiffusionOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_model_config(self, model_config):
        """AR mode uses HFModelConfig; diffusion uses DiffusionModelConfig.

        Mode is selected by ``engine_kwargs.vllm_omni.output_mode`` ("ar" vs the
        default "diffusion").
        """
        engine_kwargs = getattr(self.config, "engine_kwargs", None) or {}
        omni_kwargs = engine_kwargs.get("vllm_omni", {}) or {}
        self._ar_mode = omni_kwargs.get("output_mode", "diffusion") == "ar"

        if self._ar_mode:
            return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """AR mode: derive max_model_len. Diffusion: no max_position_embeddings."""
        if self._ar_mode:
            if self.config.max_model_len is None:
                self.config.max_model_len = self.config.prompt_length + self.config.response_length

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Diffusion needs a PIL→tensor converter; AR does not."""
        if not self._ar_mode:
            self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_override_generation_config(self) -> dict:
        """AR mode uses the parent's LLM sampling config; diffusion has none."""
        if self._ar_mode:
            return vLLMHttpServer._get_override_generation_config(self)
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _get_worker_extension_cls(self) -> str:
        return "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Strip the mode selector; in AR mode also drop diffusion-only kwargs and
        normalize underscore keys vLLM-Omni expects with dashes."""
        engine_kwargs.pop("output_mode", None)
        if self._ar_mode:
            engine_kwargs.pop("custom_pipeline", None)
            for underscore_key in ("stage_configs_path", "deploy_config", "stage_overrides", "async_chunk"):
                if underscore_key in engine_kwargs:
                    engine_kwargs[underscore_key.replace("_", "-")] = engine_kwargs.pop(underscore_key)

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        if self._ar_mode:
            # AR mode: no diffusion pipeline. Drop None entries from
            # compilation_config that OmniEngineArgs may leave behind.
            if isinstance(engine_args.get("compilation_config"), dict):
                engine_args["compilation_config"] = {
                    k: v for k, v in engine_args["compilation_config"].items() if v is not None
                }
        else:
            # inject multi-stage yaml config
            deploy_config = getattr(args, "deploy_config", None)
            if deploy_config:
                engine_args["deploy_config"] = deploy_config

            import_external_libs(self.config.external_lib)

            self.config.resolve_algorithm(self.model_config)

            pipeline_path = VllmOmniPipelineBase.get_pipeline_path(
                architecture=self.model_config.architecture,
                algorithm=self.model_config.algorithm,
            )
            # TODO (mike): read custom_pipeline from engine_args
            if pipeline_path is not None:
                engine_args["enable_dummy_pipeline"] = True
                engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

        if getattr(self.config, "step_execution", False):
            engine_args["step_execution"] = True

        diffusion_master_port, diffusion_master_sock = get_free_port("127.0.0.1", with_alive_sock=True)
        diffusion_master_sock.close()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(diffusion_master_port)
        logger.info("Using MASTER_PORT=%s for vLLM-Omni workers", os.environ["MASTER_PORT"])

        # rollout_attn_backend is a diffusion-recipe rollout field; the AR/TTS RolloutConfig does not
        # carry it, so only forward it when present (else AsyncOmni uses its own default).
        _attn_backend = getattr(self.config, "rollout_attn_backend", None)
        if _attn_backend is not None:
            engine_args["diffusion_attention_backend"] = _attn_backend
            logger.info("Setting diffusion_attention_backend=%s from rollout config", _attn_backend)

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        # TODO (mike): support multi node
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    # -----------------------------------------------------------------------
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    async def wake_up(self, tags: list[str] | None = None):
        """Override parent to use collective_rpc instead of engine.wake_up().

        The parent (verl ``1927ad33``+) calls ``self.engine.wake_up(tags=...)``
        which triggers CUDA initialisation in this HTTP server process when
        running under vLLM-Omni (AsyncOmni engine).
        Use ``collective_rpc`` instead.

        # TODO (long): drop this override once vllm-omni wake_up
        without triggering GPU initialisation.
        """
        if self.node_rank != 0:
            return
        await self.engine.collective_rpc(
            "wake_up", kwargs={"tags": tags if tags is not None else self._get_wake_up_tags()}
        )

    async def _sleep_hybrid(self):
        """Preserve non-actor pipeline weights during hybrid training sleep.

        vLLM-Omni diffusion pipelines include components such as the text
        encoder and VAE that are loaded by the rollout server, but are not part
        of the trainable actor and therefore are not included in full-model
        weight syncs. Use level-1 sleep so those weights are offloaded and can
        be restored on wake-up instead of discarded by level-2 sleep.
        """
        # TODO (andy): use `sleep_level=2` in the future when the
        #  trainer side incorporates the whole components of the model.
        await self.engine.collective_rpc("sleep", kwargs={"level": 1})
        await self.engine.reset_encoder_cache()

    # -----------------------------------------------------------------------
    # generate: shared pipeline; mode-specific steps branch on self._ar_mode
    # (_preprocess_input / _run_generation / _process_output).
    # -----------------------------------------------------------------------

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        prompt_mask: torch.BoolTensor | None = None,
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        prompt_ids = normalize_token_ids(prompt_ids)
        multi_modal_data = self._build_multi_modal_data(image_data, video_data)
        lora_request = await self._resolve_lora_request()
        prompt, params = self._preprocess_input(
            prompt_ids, sampling_params, multi_modal_data, lora_request, negative_prompt_ids, prompt_mask
        )
        final_res = await self._run_generation(prompt, params, request_id, lora_request, priority)
        return self._process_output(final_res, params, sampling_params)

    # -----------------------------------------------------------------------
    # Shared helpers for the AR and diffusion generate paths
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_multi_modal_data(image_data: Optional[list[Any]], video_data: Optional[list[Any]]) -> dict[str, Any]:
        """Assemble the vLLM multi_modal_data dict from optional image/video inputs."""
        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        return multi_modal_data

    async def _resolve_lora_request(self) -> Optional[LoRARequest]:
        """Build the actor LoRA request if a LoRA adapter is currently loaded.

        Wraps ``list_loras`` in ``try/except TypeError`` (a strict superset of the
        plain membership check): some engine backends return a non-iterable, in
        which case we assume the adapter is loaded. The diffusion path is unchanged
        in the normal (iterable) case.
        """
        if not self.lora_as_adapter:
            return None
        try:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
        except TypeError:
            lora_loaded = True
        if not lora_loaded:
            return None
        return LoRARequest(lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH)

    @staticmethod
    def _map_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
        """Map a vLLM finish_reason to verl's stop_reason vocabulary."""
        if finish_reason == "abort":
            return "aborted"
        if finish_reason in ("stop", "length"):
            return "completed"
        return finish_reason

    # -----------------------------------------------------------------------
    # Mode-specific pipeline steps
    # -----------------------------------------------------------------------

    def _tts_voice_clone_request(self, prompt_ids: list[int]):
        """Build the Qwen3-TTS talker's voice-clone request (Base task + fixed precomputed x-vector),
        so the talker conditions generation on the spoken text + clone voice. Gated on
        ``VERL_TTS_SPK_EMBED``; returns ``(None, prompt_ids)`` for non-TTS AR rollouts.

        Returns ``(additional_information, placeholder_prompt_ids)``. The talker overwrites all prompt
        embeddings, so the placeholder's *values* are ignored — only its length matters; we size it to
        the talker's prompt length (8-slot control prefix + the assistant-wrapped text)."""
        path = os.environ.get("VERL_TTS_SPK_EMBED")
        if not path:
            return None, prompt_ids
        xvec = getattr(self, "_tts_spk_xvector", None)
        if xvec is None:
            import json

            with open(path) as f:
                xvec = json.load(f)  # list[float], speaker-encoder dim (1024)
            self._tts_spk_xvector = xvec
        tok = self.model_config.tokenizer
        text = tok.decode(prompt_ids, skip_special_tokens=True).strip()
        self._tts_last_text = text  # for the VERL_TTS_DUMP diagnostic (actor-vs-rollout per-token)
        additional_information = {
            "task_type": ["Base"],
            "text": [text],
            "language": ["Auto"],
            "x_vector_only_mode": [True],
            # Non-streaming: lay the FULL text into the prefill (one set of positions before the
            # codec span), the layout the native actor reconstructs in build_talker_batch. The Base
            # default is streaming (text fed one-token-per-decode-step, summed onto codec frames),
            # whose generation log-probs can't be on-policy with the actor's full-context recompute
            # (prompt_embeds_builder.py:921-932,1277-1297). A bool here overrides the task_type default.
            "non_streaming_mode": [True],
            "voice_clone_prompt": [{"ref_spk_embedding": xvec}],
        }
        # Placeholder length MUST equal the talker's real non-streaming prompt length, else the talker
        # wedges tts_pad rows between codec_bos and decode-frame-0 (qwen3_tts_talker.py:702-706), shifting
        # every codec frame's RoPE vs the actor's reconstruction. For Base + x_vector_only + non_streaming
        # the estimator gives exactly `assistant_len + 2` (prompt_embeds_builder.py:1500-1590), where
        # assistant_len tokenizes build_assistant_text(text) with DEFAULT add_special_tokens.
        wrapped = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        assistant_len = len(tok(wrapped)["input_ids"])
        placeholder = [0] * (assistant_len + 2)
        return additional_information, placeholder

    def _preprocess_input(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        lora_request: Optional[LoRARequest],
        negative_prompt_ids: Optional[list[int]],
        prompt_mask: torch.BoolTensor | None = None,
    ):
        """Build the engine prompt + sampling params for the active mode.

        Returns ``(prompt, params)`` consumed by ``_run_generation``.
        """
        if self._ar_mode:
            max_possible_tokens = self.config.max_model_len - len(prompt_ids)
            if max_possible_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids)}) meets or exceeds the model's maximum context length "
                    f"({self.config.max_model_len}), leaving no space for generation."
                )

            if "max_tokens" in sampling_params:
                max_tokens = sampling_params.pop("max_tokens")
            elif "max_new_tokens" in sampling_params:
                max_tokens = sampling_params.pop("max_new_tokens")
            else:
                max_tokens = min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_ids),
                )
            max_tokens = max(0, min(max_tokens, max_possible_tokens))

            # Normalize ``logprobs``: bare ``True`` -> 0 (sampled-token logprob),
            # preserve explicit int counts (incl. 0), fall back to None otherwise.
            logprobs = sampling_params.pop("logprobs", None)
            if logprobs is True:
                sampling_params["logprobs"] = 0
            elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
                sampling_params["logprobs"] = logprobs
            else:
                sampling_params["logprobs"] = None
            sampling_params.setdefault("repetition_penalty", getattr(self.config, "repetition_penalty", 1.0))
            if os.environ.get("VERL_TTS_SPK_EMBED"):
                # TTS talker: stop at codec_eos so the rollout terminates at the real utterance. The
                # model emits codec_eos (talker_config.codec_eos_token_id == 2150 for this ckpt; its own
                # stop logic is `first_codebook == codec_eos_token_id`, modeling_qwen3_tts.py:2284) ~100-120
                # frames in, then generates garbage to max_tokens (2048 ≈ 170s). Making it a vLLM stop token
                # yields short, meaningful responses (clean reward audio + no garbage tail in the policy update).
                eos_id = int(os.environ.get("VERL_TTS_CODEC_EOS", "2150"))
                se = list(sampling_params.get("stop_token_ids") or [])
                if eos_id not in se:
                    sampling_params["stop_token_ids"] = se + [eos_id]
            params = SamplingParams(max_tokens=max_tokens, **sampling_params)

            tts_ai, ar_prompt_ids = self._tts_voice_clone_request(prompt_ids)
            prompt = {"prompt_token_ids": ar_prompt_ids}
            if multi_modal_data:
                prompt["multi_modal_data"] = multi_modal_data
            if tts_ai is not None:
                prompt["additional_information"] = tts_ai
            return prompt, params

        # diffusion
        default_params_list = self.engine.default_sampling_params_list

        custom_prompt: OmniCustomPrompt = {"prompt_token_ids": prompt_ids}
        if prompt_mask is not None:
            custom_prompt["prompt_mask"] = prompt_mask
        if len(default_params_list) > 1:
            # Multi-stage pipelines tag the diffusion stage so the orchestrator can route inputs correctly.
            custom_prompt["modalities"] = ["image"]
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["multi_modal_data"] = multi_modal_data
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)
        # Multi-stage models use defaults for non-diffusion stages.
        params = default_params_list[:-1] + [diffusion_sampling_params]
        return custom_prompt, params

    async def _run_generation(self, prompt, params, request_id: str, lora_request, priority: int):
        """Drive the engine and return the final OmniRequestOutput."""
        if self._ar_mode:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
        else:
            generator = self.engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=params,
            )
        final_res: Optional[OmniRequestOutput] = None
        # The talker streams its (T,16) codes one frame per decode step; the engine keeps only the
        # last yielded output, so accumulate the per-step deltas here to recover the full sequence
        # (the native actor teacher-forces them). Only for the TTS rollout (VERL_TTS_SPK_EMBED set).
        acc_codes = None
        for_tts = bool(os.environ.get("VERL_TTS_SPK_EMBED"))
        async for output in generator:
            final_res = output
            if for_tts:
                try:
                    _mm = output.multimodal_output
                    _a = _mm.get("codes", {}).get("audio") if _mm is not None else None
                    if _a is not None:
                        _a = torch.as_tensor(_a)
                        if _a.ndim == 2 and _a.shape[0] > 0:
                            if acc_codes is None:
                                acc_codes = _a
                            elif _a.shape[0] > acc_codes.shape[0]:
                                acc_codes = _a  # cumulative snapshot: keep the largest
                            else:
                                acc_codes = torch.cat([acc_codes, _a], dim=0)  # per-step delta: concat
                except Exception:  # noqa: BLE001
                    pass
        self._tts_accumulated_codes = acc_codes
        if for_tts and os.environ.get("VERL_TTS_DEBUG"):
            print(f"[tts-dbg run] accumulated_codes_shape="
                  f"{tuple(acc_codes.shape) if acc_codes is not None else None}", flush=True)
        return final_res

    def _process_output(self, final_res, params, sampling_params: dict[str, Any]):
        """Convert the engine output into the active mode's verl output dataclass."""
        if self._ar_mode:
            if final_res is None:
                raise RuntimeError("AR mode: vLLM-Omni engine yielded no output for the prompt.")

            req_output = final_res.request_output
            if req_output is None:
                raise RuntimeError("AR mode expects request_output with token IDs, but got None.")

            extra_fields = {"global_steps": self.global_steps}
            token_ids = req_output.outputs[0].token_ids
            # Surface the talker's full (T,16) sampled codes for the native Qwen3-TTS actor's
            # teacher-forced codec-0 logprob (consumed by the qwen3_tts MM patch). The accumulated
            # stream begins with `offset` zero PREFILL-PLACEHOLDER frames (the talker writes a zero
            # codes block sized to the prompt placeholder before the first decode step); the real
            # per-step frames are the SUFFIX. Strip the prefix so codes[:,0] == token_ids and the
            # actor teacher-forces the matching codec-0 + sub-codebooks (qwen3_tts_talker.py:713-718).
            # No vLLM-Omni edit. No-op for non-TTS AR.
            audio_codes = getattr(self, "_tts_accumulated_codes", None)
            if audio_codes is not None:
                # Align codes col-0 to token_ids by the best leading offset. The accumulated stream begins
                # with ~placeholder zero frames PLUS one codec_bos transition frame, so the codec-0 (col 0)
                # leads token_ids by `placeholder+1`. Matching is robust to that off-by-one — verified
                # codes[k+1,0]==token_ids[k] at 100% across samples; without it every sub-codebook the actor
                # teacher-forces is shifted one frame, which dominates the rollout↔actor logprob gap.
                L = len(token_ids)
                tid_t = torch.as_tensor(list(token_ids))
                base = audio_codes.shape[0] - L
                probe = min(L, 128)
                best_o, best_m = max(0, base), -1.0
                for o in range(max(0, base - 2), base + 3):
                    if probe > 0 and o + probe <= audio_codes.shape[0]:
                        m = (audio_codes[o : o + probe, 0] == tid_t[:probe]).float().mean().item()
                        if m > best_m:
                            best_m, best_o = m, o
                audio_codes = audio_codes[best_o : best_o + L]  # may be L-1 near the tail; actor uses min()
                extra_fields["tts_audio_codes"] = audio_codes
            if os.environ.get("VERL_TTS_DEBUG"):
                sh = tuple(audio_codes.shape) if hasattr(audio_codes, "shape") else None
                print(f"[tts-dbg proc] token_ids_len={len(token_ids)} codes_shape_after_strip={sh}", flush=True)
            log_probs = None
            if params.logprobs is not None:
                log_probs = [
                    logprobs[token_ids[i]].logprob for i, logprobs in enumerate(req_output.outputs[0].logprobs)
                ]

            if os.environ.get("VERL_TTS_DUMP"):  # capture a few rollout samples for offline diagnostics
                try:
                    n = getattr(self, "_tts_dump_n", 0)
                    if n < 4:
                        d = os.environ.get("VERL_TTS_DUMP_DIR", "/weka/whiplash-grpo/tts/dump")
                        os.makedirs(d, exist_ok=True)
                        torch.save(
                            {
                                "token_ids": torch.as_tensor(list(token_ids), dtype=torch.long),
                                "codes": audio_codes,  # (T,16) after prefix strip
                                "log_probs": (torch.as_tensor(log_probs) if log_probs is not None else None),
                                "text": getattr(self, "_tts_last_text", None),
                                "finish_reason": req_output.outputs[0].finish_reason,
                            },
                            os.path.join(d, f"sample_{n}.pt"),
                        )
                        self._tts_dump_n = n + 1
                        print(f"[tts-dump] wrote sample_{n}.pt token_ids_len={len(token_ids)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[tts-dump] failed: {e!r}", flush=True)

            finish_reason = req_output.outputs[0].finish_reason
            stop_reason = self._map_stop_reason(finish_reason)

            num_preempted = None
            if hasattr(req_output.outputs[0], "num_preempted"):
                num_preempted = req_output.outputs[0].num_preempted

            return TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                stop_reason=stop_reason,
                num_preempted=num_preempted,
                extra_fields=extra_fields,
            )

        # diffusion
        assert final_res is not None
        diffusion_output = final_res.images[0]
        if isinstance(diffusion_output, torch.Tensor):
            diffusion_output = diffusion_output.float()
        elif isinstance(diffusion_output, np.ndarray):
            diffusion_output = torch.from_numpy(diffusion_output).float()
        else:
            diffusion_output = self._to_tensor(diffusion_output).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        def _maybe_unbatch(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value[0] if value.dim() > 0 else value
            if isinstance(value, np.ndarray):
                return value[0] if value.ndim > 0 else value
            if isinstance(value, list | tuple):
                return value[0] if value else None
            return value

        extra_fields = {k: _maybe_unbatch(v) for k, v in mm_output.items() if k != "all_log_probs"}
        extra_fields["global_steps"] = self.global_steps

        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        stop_reason = self._map_stop_reason(finish_reason)

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wait_for_requests_to_drain(self):
        # TODO (mike): implement this once DP is supported.
        pass


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig | RolloutConfig,
        model_config: DiffusionModelConfig | HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
