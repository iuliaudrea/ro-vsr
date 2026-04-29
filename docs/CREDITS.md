# Credits

This repository builds on prior work. We gratefully acknowledge:

## MultiVSR (model architecture)

[Sindhu-Hegde/multivsr](https://github.com/Sindhu-Hegde/multivsr) — Prajwal,
Hegde, and Zisserman (ICASSP 2025).

The following files are downloaded by `scripts/setup.sh` from their
repository and used **without modification**:

- `ro_vsr/models.py` — encoder-decoder + visual encoder architecture
- `ro_vsr/tokenizer.py` — Whisper-style tokenizer
- `ro_vsr/checkpoints/multilingual/` — local tokenizer files

Our original code is in:

- `inference.py` — single-clip inference script
- `evaluate.py` — batch evaluation on a test set
- `ro_vsr/beam_search_ngram.py` — beam search with n-gram blocking
- `ro_vsr/dataloader_utils.py` — extracted `subsequent_mask` utility

```bibtex
@inproceedings{prajwal2025multivsr,
    author    = "Prajwal, K R and Hegde, Sindhu and Zisserman, Andrew",
    title     = "Scaling Multilingual Visual Speech Recognition",
    booktitle = "ICASSP",
    pages     = "1-5",
    year      = "2025",
}
```

## VTP feature extractor

[VGG Oxford VTP for lip-reading](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/) —
Prajwal et al.

We use the `feature_extractor.pth` checkpoint trained on the extended
training data, downloaded by `scripts/setup.sh`.

## Whisper tokenizer

[OpenAI Whisper](https://github.com/openai/whisper) — used indirectly
through the `tokenizer.py` from MultiVSR.
