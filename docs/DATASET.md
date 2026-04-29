# Dataset

Dataset-ul complet este disponibil pe HuggingFace:
[iulik-pisik/ro_vsr](https://huggingface.co/datasets/iulik-pisik/ro_vsr).

## Statistici

- **Date brute**: ~175h+ podcast-uri în limba română
- **Date cu transcripții**: ~125h (Whisper transcript + MFA word alignment)
- **Vorbitori**: [N] vorbitori unici
- **Sursă**: extras din [N] podcast-uri publice

## Splits

| Split | Durată | # Clipuri | Descriere |
| --- | --- | --- | --- |
| `train` | ~125h | ~27.6k | Date de antrenare |
| `val` | ~3.5h | ~XXX | Validation |
| `test_seen` | ~2.6h | ~XXX | Vorbitori văzuți la train |
| `test_unseen` | ~6.6h | ~XXX | Vorbitori noi |
| `test_ood` | — | 375 | Domain shift (TODO: descriere) |

## Format

Date video stocate ca **arhive .tar per podcast**, cu clipuri `.avi` (160x160).
CSV-uri cu `file_path`, `transcript`, plus alinierile word-level.

## Descărcare

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="iulik-pisik/ro_vsr",
    repo_type="dataset",
    local_dir="./ro_vsr_data",
)
```

⚠️ Atenție: dataset-ul integral are sute de GB. Pentru testare rapidă,
descarcă doar `samples/` (vezi [`scripts/download_samples.sh`](../scripts/download_samples.sh)).

## Preprocesare

Toate clipurile sunt deja preprocesate:
- Crop pe față detectată cu [SyncNet](https://github.com/joonson/syncnet_python)
- Resize la 160x160
- 25 fps

Pentru detalii reproductive complete, vezi pipeline-ul de preprocesare în
[lucrare](https://...) (Section X.Y).
