# VSRo-200: Sentence-level Visual Speech Recognition for Romanian

Inference code and demo for the VSRo-200 dataset and baseline models.

We introduce **VSRo-200**, the first sentence-level dataset and visual
speech recognition system for Romanian. The dataset contains approximately
200 hours of Romanian podcast recordings with Whisper-generated
transcriptions and word-level alignments. Prior work (2020–2025) addressed
only isolated word classification on the LRRo dataset; we provide the
first sentence-level baseline for Romanian.


<p align="center">
    <img src="samples/VSRo-200.gif"/>
</p>

This repository contains:

- The inference code and pre-trained encoder-decoder models
- Demo clips covering different categories of our test set (clean
  podcasts, vlogs, archival footage, noisy scenes, academic talks)
- Documentation for evaluating the model on your own clips

The training pipeline and the full dataset are described in the paper
and released separately on the HuggingFace Hub.

## Installation

```bash
git clone https://github.com/iuliaudrea/ro-vsr.git
cd ro-vsr
pip install -r requirements.txt
bash scripts/setup.sh
```

`setup.sh` handles everything: it clones MultiVSR (model architecture and
tokenizer) and downloads the VTP feature extractor (~1 GB, from VGG Oxford).

> **Note**: Our code builds on the architecture from
> [MultiVSR](https://github.com/Sindhu-Hegde/multivsr) (Prajwal et al., 2025).
> See [`docs/CREDITS.md`](docs/CREDITS.md) for full attribution.

Tested on Python 3.10–3.12, on both GPU and CPU.

## Quick inference on a sample

```bash
python inference.py --fpath samples/sample_1.avi
```

Expected output:

```
[device] cuda
[load] Downloading encoder-decoder from iulik-pisik/ro_vsr_175h_auto ...
[load] ✅ Models loaded successfully
[video] Frames extracted: (1, 3, 146, 96, 96)
[infer] Running inference ...
──────────────────────────────────────────────────────────────────────
File:           samples/sample_1.avi
Transcription:  nu mă interesează să demonstrez ceva ce am avut de demonstrat sigur că am cam demonstrat așa
──────────────────────────────────────────────────────────────────────
```

Inference runs on both GPU (~2s/clip on T4) and CPU (~45 min/clip).
A GPU is recommended but not required.

## Demo samples

We provide 8 demo clips covering different categories of our test set.
The selection includes both well-performing cases and deliberate failure
modes that illustrate the model's limitations:

| File | Category | Subset | Reference |
| --- | --- | --- | --- |
| `sample_1.avi` | Podcast (clean) | test_seen | "nu mă interesează să demonstrez ceva..." |
| `sample_2.avi` | Podcast (clean) | test_seen | "băi și îți dai seama nu aveți voie..." |
| `sample_3.avi` | Vlog | test_ood | "..." |
| `sample_4.avi` | Black-and-white | test_ood | "..." (archival footage; failure mode: low resolution) |
| `sample_5.avi` | Noisy scene | test_ood | "..." (relevant for AVSR comparison) |
| `sample_6.avi` | Academic talk | test_ood | "..." (failure mode: technical vocabulary) |
| `sample_7.avi` | Podcast | test_unseen | "..." (unseen speaker) |
| `sample_8.avi` | Podcast | test_unseen | "..." (unseen speaker) |

See [`samples/samples_metadata.csv`](samples/samples_metadata.csv) for
full metadata.

To run inference on all samples at once:

```bash
for i in 1 2 3 4 5 6 7 8; do
    python inference.py --fpath samples/sample_${i}.avi
done
```

## CLI options

```bash
python inference.py \
    --fpath samples/sample_1.avi \
    --model iulik-pisik/ro_vsr_175h_auto \
    --beam_size 5 \
    --max_len 256 \
    --no_repeat_ngram_size 5
```

Available options:

- `--fpath`: input .avi clip (160x160, mouth crop) — **required**
- `--model`: HuggingFace repo of the encoder-decoder model (default: `ro_vsr_175h_auto`)
- `--vtp_path`: path to the feature extractor (default: `checkpoints/feature_extractor.pth`)
- `--beam_size`: beam size (default: `5`)
- `--max_len`: max output tokens (default: `256`)
- `--no_repeat_ngram_size`: block n-grams of this size from repeating (default: `5`, set to `0` to disable)
- `--device`: `cuda` or `cpu` (default: auto-detect)

## Available models

| Model | Training data | WER test_seen | WER test_unseen |
| --- | --- | --- | --- |
| `iulik-pisik/ro_vsr_175h_auto` | ~175h Romanian podcasts | XX% | YY% |
| `iulik-pisik/ro_vsr_125h_auto` | ~125h Romanian podcasts | XX% | YY% |

## Inference on raw video

Direct inference works only on **already-preprocessed** clips (160x160 with
mouth crop). For raw video (an arbitrary .mp4 of someone speaking),
face detection and mouth region extraction must be applied first. We
recommend the MultiVSR preprocessing pipeline (based on SyncNet):

```bash
git clone https://github.com/Sindhu-Hegde/multivsr.git
cd multivsr/preprocess
python run_pipeline.py \
    --videofile <path/to/video.mp4> \
    --reference my_clip \
    --data_dir /tmp/processed
```

Then:

```bash
python inference.py --fpath /tmp/processed/my_clip/pycrop/00000.avi
```

See [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) for details.

## Dataset

The full VSRo-200 dataset (~200h of Romanian podcasts with transcriptions
and word-level alignments) is available on HuggingFace:
[iulik-pisik/ro_vsr](https://huggingface.co/datasets/iulik-pisik/ro_vsr).

See [`docs/DATASET.md`](docs/DATASET.md) for splits, preprocessing, and
statistics.

## Other experiments

The paper additionally reports results on:

- **Audio-Visual Speech Recognition (AVSR)**: noise-robust evaluation
  combining VSRo-200 with audio fused via cross-modal attention.
- **Word-level classification on LRRo**: evaluation of our encoder
  representations on the LRRo benchmark (Țucă et al., 2020).

The code for these experiments will be released in `evaluation/` in a
follow-up update. The LRRo dataset must be obtained directly from its
authors and is not redistributed here.

## Citation

If you use this code, the models, or the VSRo-200 dataset, please cite:

```bibtex
@inproceedings{[id]2026vsro200,
    author    = "[Author 1] and [Author 2]",
    title     = "[Paper title]",
    booktitle = "Advances in Neural Information Processing Systems Datasets and Benchmarks Track",
    year      = "2026",
}
```

Please also cite the foundational work we build upon:

```bibtex
@inproceedings{prajwal2025multivsr,
    author    = "Prajwal, K R and Hegde, Sindhu and Zisserman, Andrew",
    title     = "Scaling Multilingual Visual Speech Recognition",
    booktitle = "ICASSP",
    pages     = "1-5",
    year      = "2025",
}
```

## License

Our code: **MIT**.
Code from MultiVSR (`ro_vsr/models.py`, `ro_vsr/tokenizer.py`,
downloaded by `scripts/setup.sh`):
property of [Prajwal & Hegde](https://github.com/Sindhu-Hegde/multivsr).