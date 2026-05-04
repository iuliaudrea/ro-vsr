# VSRo-200: Visual Speech Recognition for Romanian

We introduce **VSRo-200**, the first sentence-level dataset and visual speech recognition system for Romanian. The dataset contains approximately 200 hours of Romanian podcast recordings with both Whisper-generated transcriptions and human annotations. 

Building upon the MultiVSR architecture, which we fine-tune with our data, we establish the first sentence-level VSR baseline for the Romanian language. In addition, we provide an audio-visual speech recognition (AVSR) extension for noisy conditions and a word-level evaluation on the LRRo benchmark.

🤗 **[Dataset & Models on Hugging Face: vsro200](https://huggingface.co/datasets/vsro200/vsro200)**


<br>
<p align="center">
    <img src="dataset/vsro_dataset.gif"/>
</p>
<br>

## What's in this repo

* `dataset/` — how to reconstruct the VSRo-200 dataset using the timestamps and transcripts published on HuggingFace
* `inference.py` — single-clip VSR inference, with sample inputs in `samples/`
* `evaluation/` — audio-visual fusion (`avsr/`) and LRRo word-level
  classification (`lrro_classification/`)
* `methodology/` — training scripts and data preparation pipeline,
  for full reproducibility

## Installation

```bash
git clone https://github.com/iuliaudrea/ro-vsr.git
cd ro-vsr
bash scripts/setup.sh
pip install -r requirements.txt
```

`setup.sh` clones MultiVSR (model architecture and
tokenizer) and downloads the VTP feature extractor (~1 GB, from VGG Oxford).


## Quick inference on a sample

```bash
python inference.py --fpath samples/sample_1.avi
```

Expected output:

```
[device] cuda
[load] Downloading encoder-decoder from vsro200/models-vsro200/checkpoints/model_200h_auto.pt ...
[load] ✅ Models loaded successfully
[video] Frames extracted: (1, 3, 146, 96, 96)
[infer] Running inference ...
──────────────────────────────────────────────────────────────────────
File:           samples/sample_1.avi
Transcription:  poate subconștientul meu pentru că totuși am învățat și spun sincer nu neapărat că mi-aș dori îmi doresc o familie dar parcă deja
Reference:      poate în subconștientul meu pentru că totuși am o vârstă și spun sincer nu neapărat că mi-aș dori îmi doresc o familie dar parcă deja
WER:            12.00%
CER:            6.77%
──────────────────────────────────────────────────────────────────────
```

### Sample clips

The `samples/` folder ships with several short VSR clips. Each clip is
a 224×224 face crop produced by MultiVSR's preprocessing pipeline.

| Sample | Reference |
| :--- | :--- |
| `samples/sample_1.avi` | poate în subconștientul meu pentru că totuși am o vârstă și spun sincer nu neapărat că mi-aș dori îmi doresc o familie dar parcă deja |
| `samples/sample_2.avi` | băi și îți dai seama nu aveți voie să fumați o să vă spun |
| `samples/sample_3.avi` | mi-au suportat multe și eu le-am suportat eu lor dar așa e |
| `samples/sample_4.avi` | cumva vreau să și înțeleg de ce în anumite momente eu des- detest omenirea și umanitatea și înțelegi |
| `samples/sample_5.avi` | cred că ăsta e cel mai frumos mod să descoperi un oraș o iei așa la pas nu știi nimic nu știu la ce să te aștepți și acum am dat și de china town |
| `samples/sample_6.avi` | opoziții tot astfel cum o judecată are două valori adevărată sau fals |
| `samples/sample_7.avi` | dar nu-l mai poate suporta nimeni atâta timp cât te bagi tu în în poliție |
| `samples/sample_8.avi` | ne referim la creșterea particulelor de ldl colesterol care sunt agresive la nivelul peretelui arterial la trigliceride sau din contră scăderea acelor particule de colesterol |

For full details, including sample splits and video categories, see [`samples/samples_metadata.csv`](samples/samples_metadata.csv).

## Other experiments

### Audio-visual fusion (AVSR)

We combine the VSR encoder-decoder with a Romanian-tuned Whisper-small
(`vsro200/whisper-small-vsro200`) using shallow log-probability fusion
during decoding. This significantly improves robustness in noisy
conditions, particularly at low SNRs.

See [`evaluation/avsr/`](evaluation/avsr/) for inference code and sample inputs.

### Word-level classification on LRRo

The frozen VSR encoder transfers well to word-level lipreading. We add
a lightweight attention-pooling + MLP head and evaluate on LRRo
(Jitaru et al., 2020). See [`evaluation/lrro_classification/`](evaluation/lrro_classification/)
for inference code and sample inputs.

## Methodology

The notebooks and scripts used to **build** the dataset and **train**
the released models are documented in
[`methodology/`](methodology/) for full reproducibility. End users
do not need to run any of this code to use the released artifacts.

## Citation

If you use this code, the models, or the VSRo-200 dataset, please cite:

```bibtex
@inproceedings{[id]2026vsro200,
    author    = "",
    title     = "VSRo-200: A Romanian Visual Speech Recognition Dataset for Studying Supervision and Multimodal Robustness",
    year      = "2026"
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
