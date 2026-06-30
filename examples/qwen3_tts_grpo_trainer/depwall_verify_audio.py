"""Rigorously verify the generated audio is CORRECT speech (not the tf5 silent-break).

The tf5 model break produces 85-99% silent / near-zero audio that whisper cannot
transcribe. Here we (a) transcribe with faster-whisper and compute CER vs the target
text, and (b) characterise the silence: raw <1e-3 ratio, edge-trimmed ratio, and the
longest contiguous silent run (a broken decoder scatters zeros THROUGHOUT; natural
speech concentrates silence at the edges and short inter-word gaps).
"""
import argparse
import numpy as np
import soundfile as sf


def cer(ref: str, hyp: str) -> float:
    ref = "".join(c.lower() for c in ref if c.isalnum() or c.isspace()).split()
    hyp = "".join(c.lower() for c in hyp if c.isalnum() or c.isspace()).split()
    r, h = " ".join(ref), " ".join(hyp)
    # char-level Levenshtein
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[len(h)] / max(1, len(r))


def analyse(w, sr, label):
    raw = float((np.abs(w) < 1e-3).mean())
    # edge-trim using a 20ms frame energy gate at -45 dBFS
    fl = int(0.02 * sr)
    frames = [w[i:i + fl] for i in range(0, len(w) - fl, fl)]
    energy = np.array([float(np.sqrt(np.mean(f.astype(np.float64) ** 2)) + 1e-12) for f in frames])
    db = 20 * np.log10(energy / (np.max(energy) + 1e-12) + 1e-12)
    voiced = np.where(db > -45)[0]
    if len(voiced):
        a, b = voiced[0] * fl, min(len(w), (voiced[-1] + 1) * fl)
        trimmed = w[a:b]
    else:
        trimmed = w
    trim_ratio = float((np.abs(trimmed) < 1e-3).mean())
    # frame-energy silence: fraction of 20ms frames below -45 dBFS (VAD-style). This is the
    # metric that actually discriminates correct speech (~few %) from the tf5 break (~85-99%);
    # the raw <1e-3 sample ratio over-counts natural sub-(-60dBFS) samples in clean TTS pauses.
    frame_silence = float(1.0 - len(voiced) / max(1, len(frames)))
    # longest contiguous <1e-3 run, as a fraction of total
    sil = np.abs(w) < 1e-3
    longest = cur = 0
    for s in sil:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    print(f"[{label}] dur={len(w)/sr:.2f}s peak={np.max(np.abs(w)):.4f} "
          f"rms={np.sqrt(np.mean(w**2)):.4f} raw_silence={raw:.3f} "
          f"frame_silence={frame_silence:.3f} longest_silent_run={longest/len(w):.3f}",
          flush=True)
    return frame_silence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="/weka/whiplash-grpo/depwall_clone.wav")
    ap.add_argument("--text", default="Hello, how can I help you today?")
    args = ap.parse_args()

    w, sr = sf.read(args.wav)
    w = np.asarray(w, np.float32)
    analyse(w, sr, "clip")

    # faster-whisper transcription -> CER (the project's real correctness signal)
    from faster_whisper import WhisperModel
    # faster-whisper (ctranslate2) wants CUDA-12 libs; this is a CUDA-13 image, so run on CPU
    # (one short clip — CPU is fine, and the GPU rollout/reward use their own kernels).
    model = WhisperModel("Systran/faster-whisper-large-v3", device="cpu", compute_type="int8")
    segments, info = model.transcribe(args.wav, language="en", beam_size=5)
    hyp = " ".join(s.text for s in segments).strip()
    c = cer(args.text, hyp)
    print(f"[whisper] target = {args.text!r}", flush=True)
    print(f"[whisper] hypo   = {hyp!r}", flush=True)
    print(f"[whisper] CER    = {c:.3f}", flush=True)
    # Correct speech: whisper recovers the text (CER low). The tf5 break -> empty/garbage hypo.
    assert c < 0.20, f"high CER ({c:.2f}) — audio is not intelligible speech"
    print("[whisper] PASS: audio is intelligible, correct speech (NOT the tf5 silent break)", flush=True)


if __name__ == "__main__":
    main()
