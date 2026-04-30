# Audio-Visual Speech Recognition (AVSR) Evaluation

This folder contains the AVSR experiments reported in the paper:
**Shallow fusion between our VSRo model (video) and a fine-tuned Whisper
model (audio) under noisy conditions.**

The motivation: visual speech recognition is unaffected by acoustic noise.
At low SNRs, fusing video predictions with audio predictions improves over
audio-only (Whisper) decoding.

## Quick start

From the repository root, after running the main `bash scripts/setup.sh`:

```bash
cd evaluation/avsr
pip install transformers torchaudio          # if not already installed
python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR-5.mp4
```

Expected output:

```
[device] cuda
[load] Downloading VSR model from iulik-pisik/ro_vsr_175h_auto ...
[load] Downloading Whisper from alexandradiaconu/whisper-small-ro-noisy ...
[load] ✅ All models loaded successfully
[video] Frames extracted: (1, 3, 100, 96, 96)
[audio] Samples loaded:   64000 (4.00s @ 16 kHz)
[infer] Running inference (mode=hibrid_logp) ...
──────────────────────────────────────────────────────────────────────
File:           samples_avsr/sample_1_babble_SNR-5.mp4
Mode:           hibrid_logp
Transcription:  ...
──────────────────────────────────────────────────────────────────────
```

## Decoding modes

The `--mode` flag selects how predictions are produced:

- **`hibrid_logp`** (default): shallow fusion at the log-probability level —
  averages `log_softmax(audio_logits)` and `log_softmax(video_logits)` at
  each decoding step. **This is the main reported method.**
- **`whisper`**: audio-only baseline (Whisper alone).
- **`multivsr`**: video-only baseline (VSRo alone).

To compare all three on the same clip:

```bash
for mode in whisper multivsr hibrid_logp; do
    python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR-5.mp4 --mode $mode
done
```

At low SNR (e.g., −5 dB), `hibrid_logp` should outperform `whisper`,
demonstrating the benefit of incorporating visual information.

## Demo samples

We provide [N] demo MP4 clips with noise pre-mixed into the audio track
at varying SNRs. The video remains a clean 160×160 face crop; only the
audio is degraded. This means the same clip can be decoded with all three
modes to directly observe the effect of fusion.

| File | Noise type | SNR (dB) | Reference |
| --- | --- | --- | --- |
| `sample_1_babble_SNR-5.mp4` | babble (4-speaker) | −5 | "..." |
| `sample_2_babble_SNR0.mp4` | babble (4-speaker) | 0 | "..." |
| `sample_3_babble_SNR10.mp4` | babble (4-speaker) | 10 | "..." |
| `sample_4_gaussian_SNR-5.mp4` | gaussian (white) | −5 | "..." |
| `sample_5_gaussian_SNR5.mp4` | gaussian (white) | 5 | "..." |

Babble noise is sampled from [MUSAN](https://www.openslr.org/17/), mixed
from 4 randomly selected speakers. See `samples_avsr_metadata.csv` for
full metadata.

## Reported results

WER (%) on `test_valid` (100 clips), averaged across the noise type at
each SNR:

| SNR (dB) | Whisper (zero-shot) | Whisper (fine-tuned) | VSRo (video) | Shallow Fusion (logp) |
| --- | --- | --- | --- | --- |
| −5 | XX.X | XX.X | XX.X | **XX.X** |
| 0 | XX.X | XX.X | XX.X | **XX.X** |
| 5 | XX.X | XX.X | XX.X | **XX.X** |
| 10 | XX.X | XX.X | XX.X | **XX.X** |
| 15 | XX.X | XX.X | XX.X | **XX.X** |

(Replace placeholders with your actual numbers from
`results_consolidated.csv`.)

## CLI options

```
--fpath          Path to input video (.mp4 or .avi, 160x160 with audio track)
--vsr_model      HF repo for the VSR model (default: iulik-pisik/ro_vsr_175h_auto)
--whisper_model  HF repo for Whisper (default: alexandradiaconu/whisper-small-ro-noisy)
--vtp_path       Path to VTP checkpoint (default: ../../checkpoints/feature_extractor.pth)
--mode           hibrid_logp | whisper | multivsr (default: hibrid_logp)
--beam_size      Beam size (default: 5)
--max_len        Max output tokens (default: 256)
--device         cuda | cpu (default: auto-detect)
```

## Running on your own clips

If you want to test on your own noisy clips:

1. Make sure your video is a 160×160 face crop (use the MultiVSR
   preprocessing pipeline — see `../../docs/PREPROCESSING.md`).
2. Mix audio with noise at the desired SNR (e.g., with `ffmpeg` and
   MUSAN babble samples).
3. Mux the noisy audio back into the video as a single MP4:
   ```bash
   ffmpeg -i your_video.mp4 -i your_noisy_audio.wav \
          -c:v copy -map 0:v:0 -map 1:a:0 -shortest \
          your_clip_with_noise.mp4
   ```
4. Run inference:
   ```bash
   python inference_avsr.py --fpath your_clip_with_noise.mp4
   ```

## Citation

If you use the AVSR setup, please cite our paper as well as the underlying
Whisper model:

```bibtex
@misc{radford2022whisper,
  title  = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg
            and McLeavey, Christine and Sutskever, Ilya},
  year   = {2022},
  url    = {https://arxiv.org/abs/2212.04356},
}
```
