# Romanian Visual Speech Recognition

Cod oficial pentru lucrarea **"[Numele lucrării tale]"**.

| 📝 Paper | 🤗 Dataset | 🤗 Models |
| --- | --- | --- |
| [Paper](https://...) | [iulik-pisik/ro_vsr](https://huggingface.co/datasets/iulik-pisik/ro_vsr) | [iulik-pisik/ro_vsr_125h_auto](https://huggingface.co/iulik-pisik/ro_vsr_125h_auto) |

Introducem **primul sistem sentence-level de Visual Speech Recognition pentru
limba română**, antrenat pe ~125 ore de podcast-uri în limba română cu
transcripții generate cu Whisper și aliniere la nivel de cuvânt.

Lucrările anterioare (2020–2025) abordau doar clasificare de cuvinte izolate
pe dataset-ul LRRo. Contribuția principală a acestei lucrări:

- 🎬 Primul **dataset sentence-level** open-source pentru română VSR
  (~175h+ de date brute, ~125h cu transcripții curate)
- 🧠 Prima **baseline pe propoziții** pentru română (WER ~XX% pe test_seen,
  ~YY% pe test_unseen)
- 🔧 Pipeline complet open-source: preprocesare, antrenare, inferență

## News

- **[2026.04]** 🚀 Lansare cod și modele

## Instalare

```bash
git clone https://github.com/iulik-pisik/ro-vsr.git
cd ro-vsr
python -m venv env_ro_vsr
source env_ro_vsr/bin/activate
pip install -r requirements.txt
```


> **Notă**: Codul nostru se bazează pe arhitectura din
> [MultiVSR](https://github.com/Sindhu-Hegde/multivsr) (Prajwal et al., 2025).
> Vezi [`docs/CREDITS.md`](docs/CREDITS.md) pentru atribuiri complete.

Testat pe Python 3.10 și Python 3.11 cu CUDA 11.8 / 12.1.

## Inferență rapidă pe un sample

```bash
python inference.py --fpath samples/sample_1.avi
```

Output așteptat:

```
[device] cuda
[load] Descarc enc-dec din iulik-pisik/ro_vsr_125h_auto ...
[load] ✅ Modele încărcate cu succes
[video] Frames extrase: (1, 3, 87, 96, 96)
[infer] Rulez inferența ...
──────────────────────────────────────────────────────────────────────
Fișier:       samples/sample_1.avi
Transcriere:  și atunci am început să mă gândesc la ce înseamnă
──────────────────────────────────────────────────────────────────────
```

## Sample-uri demo

| Fișier | Durată | Transcriere reală |
| --- | --- | --- |
| `samples/sample_1.avi` | ~3.5s | *(text de referință)* |
| `samples/sample_2.avi` | ~4.2s | *(text de referință)* |
| `samples/sample_3.avi` | ~3.8s | *(text de referință)* |
| `samples/sample_4.avi` | ~5.0s | *(text de referință)* |
| `samples/sample_5.avi` | ~4.5s | *(text de referință)* |

Vezi [`samples/samples_metadata.csv`](samples/samples_metadata.csv) pentru
metadate complete.

## Opțiuni CLI

```bash
python inference.py \
    --fpath samples/sample_1.avi \
    --model iulik-pisik/ro_vsr_125h_auto \
    --beam_size 5 \
    --max_len 256
```

Opțiuni disponibile:

- `--fpath`: clipul .avi de input (160x160, crop pe gură) — **obligatoriu**
- `--model`: repo HuggingFace al modelului enc-dec (default: `ro_vsr_125h_auto`)
- `--vtp_path`: path către feature extractor (default: `checkpoints/feature_extractor.pth`)
- `--beam_size`: beam size (default: `5`)
- `--max_len`: max output tokens (default: `256`)
- `--device`: `cuda` sau `cpu` (default: auto)

## Modele disponibile

| Model | Date antrenare | WER test_seen | WER test_unseen |
| --- | --- | --- | --- |
| `iulik-pisik/ro_vsr_125h_auto` | ~125h podcast-uri RO | XX% | YY% |
| `iulik-pisik/ro_vsr_150h_auto` | ~150h podcast-uri RO | XX% | YY% |
| `iulik-pisik/ro_vsr_100h` | ~100h subset | XX% | YY% |

## Pe video brut (din afara dataset-ului)

Inferența directă funcționează doar pe clipuri **deja preprocesate**
(160x160, crop pe gură). Pentru video brut (un .mp4 oarecare cu o persoană
care vorbește), trebuie întâi rulată detecția feței și extragerea regiunii
gurii. Recomandăm pipeline-ul de preprocesare al MultiVSR (bazat pe SyncNet):

```bash
git clone https://github.com/Sindhu-Hegde/multivsr.git
cd multivsr/preprocess
python run_pipeline.py \
    --videofile <calea_la_video.mp4> \
    --reference my_clip \
    --data_dir /tmp/processed
```

Apoi:

```bash
python inference.py --fpath /tmp/processed/my_clip/pycrop/00000.avi
```

Detalii în [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md).

## Dataset

Dataset-ul complet (~175h podcast-uri în limba română cu transcripții și
aliniere word-level) este disponibil pe HuggingFace:
[iulik-pisik/ro_vsr](https://huggingface.co/datasets/iulik-pisik/ro_vsr).

Vezi [`docs/DATASET.md`](docs/DATASET.md) pentru detalii despre splits,
preprocesare și statistici.

## Citation

Dacă folosești codul, modelele sau dataset-ul, te rugăm să citezi:

```bibtex
@inproceedings{[id]2026rovsr,
    author    = "[Iulia Nume] and [Advisor]",
    title     = "[Titlul lucrării]",
    booktitle = "Advances in Neural Information Processing Systems Datasets and Benchmarks Track",
    year      = "2026",
}
```

Te rugăm să citezi și lucrarea de bază pe care ne sprijinim:

```bibtex
@inproceedings{prajwal2025multivsr,
    author    = "Prajwal, K R and Hegde, Sindhu and Zisserman, Andrew",
    title     = "Scaling Multilingual Visual Speech Recognition",
    booktitle = "ICASSP",
    pages     = "1-5",
    year      = "2025",
}
```

## Licență

Codul nostru: **MIT**.
Codul preluat din MultiVSR (`ro_vsr/models.py`, `ro_vsr/tokenizer.py`,
descărcat de `scripts/setup.sh`):
proprietate [Prajwal & Hegde](https://github.com/Sindhu-Hegde/multivsr).
