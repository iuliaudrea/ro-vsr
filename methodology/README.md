# Methodology

This folder contains the full methodology used to build the VSRo-200
dataset and train the models reported in the paper.

> ⚠️ **End users do not need to run any of this code.** Everything
> here is for transparency and reproducibility. To **use** the
> released dataset and models, see [`../dataset/`](../dataset/) and
> the inference scripts in the repository root.


## Subfolders

### [`data_preparation/`](data_preparation/)

Scripts that produce the timestamps and transcripts published as the
VSRo-200 splits on HuggingFace:

- `extract_host_embeddings.py` — face embeddings for podcast hosts
- `filter_with_pyannote.py` — scene + face + audio filtering pipeline
- `pseudo_label_whisper.py` — Whisper-large transcription of clips

### [`model_training/`](model_training/)

Scripts that produce the models published on HuggingFace:

- `finetune_vsr.py` — fine-tunes MultiVSR encoder-decoder on VSRo-200
- `finetune_whisper_avsr.py` — Whisper-small fine-tuning with noise
  augmentation (used for the AVSR shallow fusion experiments)
- `train_lrro_mlp.py` — attention-pooling + MLP head trained on LRRo
  encoder embeddings




