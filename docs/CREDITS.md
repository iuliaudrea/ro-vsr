# Credits

Acest repository se sprijină pe lucrări anterioare. Atribuim cu recunoștință:

## MultiVSR (arhitectura modelului)

[Sindhu-Hegde/multivsr](https://github.com/Sindhu-Hegde/multivsr) — Prajwal,
Hegde și Zisserman (ICASSP 2025).

Următoarele fișiere sunt descărcate de `scripts/setup.sh` din repo-ul lor și
folosite **fără modificări**:

- `ro_vsr/models.py` — definiția arhitecturii encoder-decoder + visual encoder
- `ro_vsr/tokenizer.py` — tokenizer Whisper

Codul nostru original este în:

- `inference.py` — scriptul de inferență
- `evaluate.py` — evaluare batch pe test set
- `ro_vsr/beam_search_ngram.py` — beam search cu n-gram blocking
- `ro_vsr/dataloader_utils.py` — `subsequent_mask` extras

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

Folosim checkpoint-ul `feature_extractor.pth` antrenat pe extended train data,
descărcat de `scripts/download_checkpoints.sh`.

## Whisper tokenizer

[OpenAI Whisper](https://github.com/openai/whisper) — folosit indirect prin
`tokenizer.py` din MultiVSR.
