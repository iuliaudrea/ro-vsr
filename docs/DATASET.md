# VSRo-200 Dataset

The full dataset is available on HuggingFace:
[vsro200/vsro200](https://huggingface.co/datasets/vsro200/vsro200).

## Statistics

- **Raw video**: ~200h of Romanian podcasts
- **Annotated subset**: ~125h (Whisper transcripts + MFA word alignments)
- **Speakers**: [N] unique speakers
- **Source**: [N] publicly available podcasts

## Splits

| Split | Duration | # Clips | Description |
| --- | --- | --- | --- |
| `train` | ~125h | ~27.6k | Training data |
| `val` | ~3.5h | ~XXX | Validation |
| `test_seen` | ~2.6h | ~XXX | Speakers seen during training |
| `test_unseen` | ~6.6h | ~XXX | New speakers |
| `test_ood` | — | 375 | Domain shift evaluation |

## Format

Video data is stored as **per-podcast .tar archives**, with `.avi` clips
at 160x160 resolution. CSV files contain `file_path`, `transcript`, plus
word-level alignments.

## Download

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="vsro200/vsro200",
    repo_type="dataset",
    local_dir="./vsro200_data",
)
```

⚠️ Note: the full dataset is hundreds of GBs. For quick testing, see the
demo samples included in `samples/`.

## Preprocessing

All clips are pre-processed:
- Face crop using [SyncNet](https://github.com/joonson/syncnet_python)
- Resize to 160x160
- 25 fps

For full reproducibility details, see Section X.Y of the [paper](https://...).
