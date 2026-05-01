# Word-level classification on LRRo

This folder contains the LRRo evaluation reported in our paper:
**Word-level lipreading on the LRRo benchmark using our pre-trained
encoder representations.**

We freeze the VTP visual encoder and our VSR transformer encoder
(trained on the VSRo-200 dataset) and train a lightweight
attention-pooling + MLP head on LRRo. This evaluates the cross-domain
transferability of our encoder representations to a standard Romanian
lip-reading benchmark.

## ⚠️ LRRo dataset is not redistributed

LRRo (Jitaru et al., 2020) must be obtained directly from the official
source: **https://bionescu.aimultimedialab.ro/LRRo.html**

We provide only the **inference code** and the **trained MLP heads**.
The dataset itself remains the property of its authors (Jitaru,
Abdulamit, and Ionescu — University Politehnica of Bucharest).

If you cite our LRRo evaluation, please also cite the original LRRo
paper:

```bibtex
@inproceedings{jitaru2020lrro,
    author    = "Jitaru, Andrei Cosmin and Abdulamit, Șeila and Ionescu, Bogdan",
    title     = "{LRRo}: A Lip Reading Data Set for the Under-resourced Romanian Language",
    booktitle = "Proceedings of the 11th ACM Multimedia Systems Conference",
    pages     = "267-272",
    year      = "2020",
}
```

## Quick start

After setting up the main repository (`bash scripts/setup.sh` from the
repo root), download LRRo from the official source. The expected
folder structure is:

```
/path/to/lrro/LRRo data set/
├── Lab_LRRo_data_set/
│   ├── train/<word>/<clip_id>/0.jpg, 1.jpg, ...
│   ├── val/<word>/<clip_id>/...
│   └── test/<word>/<clip_id>/...
└── Wild_LRRo_data_set/
    ├── train/<word>/<clip_id>/...
    ├── val/...
    └── test/...
```

Then run inference on any clip:

```bash
cd evaluation/lrro_classification
python inference_lrro.py --clip_dir /path/to/lrro/Lab_LRRo_data_set/test/buna/12345/
```

Expected output:

```
[device] cuda
[load] Loading VTP visual encoder ...
[load] Downloading VSR encoder from iulik-pisik/ro_vsr_150h_auto
[load] Encoder embedding dimension: 768
[load] Downloading MLP from iulik-pisik/ro_vsr_classification_mlps/48_bottom/best_lab_clf.pt
[load] MLP has 48 output classes
[video] Frames extracted: (1, 3, 29, 96, 96)
[infer] Running inference ...
──────────────────────────────────────────────────────────────────────
Clip:            /path/to/.../buna/12345/
Strategy:        48_bottom
MLP split:       lab  (48 classes)
Top-5 predictions:
  1. class_3              ████████████████████████░░░░░░  82.43%
  2. class_17             ███░░░░░░░░░░░░░░░░░░░░░░░░░░░  10.12%
  ...
──────────────────────────────────────────────────────────────────────
```

## Showing actual word labels

By default, predictions are displayed as `class_0`, `class_1`, etc.,
since we cannot redistribute the LRRo word list. To see actual word
predictions, build a class-to-index mapping from your LRRo download
and pass it via `--class_map`:

```python
# Build the mapping (run this once after downloading LRRo)
import os, json

lrro_dir = "/path/to/lrro/LRRo data set"

for split_dataset, json_name in [
    ("Lab_LRRo_data_set", "lab_classes.json"),
    ("Wild_LRRo_data_set", "wild_classes.json"),
]:
    train_dir = os.path.join(lrro_dir, split_dataset, "train")
    words = sorted(os.listdir(train_dir))
    mapping = {str(i): w for i, w in enumerate(words)}
    with open(json_name, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Wrote {json_name} with {len(words)} classes")
```

Then run:

```bash
python inference_lrro.py \
    --clip_dir /path/to/lrro/Lab_LRRo_data_set/test/<word>/<clip_id>/ \
    --class_map lab_classes.json
```

The mapping reproduces the alphabetical class ordering used during
training, so it matches the trained MLP exactly.

## Configuration

The VTP visual encoder and the VSR encoder are **fixed**:

- **VTP**: original feature extractor from VGG Oxford (downloaded by
  the main `setup.sh`)
- **VSR encoder**: `iulik-pisik/ro_vsr_150h_auto` (downloaded
  automatically at first run)

The **preprocessing strategy** and the **MLP head** are configurable.
We trained MLPs on four different preprocessing strategies, each with
a separate head for the LAB and WILD splits of LRRo.

### Preprocessing strategies

LRRo provides 64x64 grayscale mouth crops. Our VTP encoder expects
96x96 RGB face-like inputs. We tested four strategies for adapting
LRRo frames:

| Strategy | Description | Notes |
| --- | --- | --- |
| `48_bottom` (default) | Resize to 48x48, place center-bottom on 96x96 gray canvas | Best result; mimics where a real mouth would appear in a face crop |
| `64_bottom` | Keep 64x64, place center-bottom on 96x96 gray canvas | No information loss from resizing |
| `64_middle` | Keep 64x64, place exactly center on 96x96 gray canvas | Centered placement |
| `96_resize` | Resize directly to 96x96 (no padding) | Frame fills the entire canvas |

### CLI options

```
--clip_dir   Folder containing the clip's .jpg frames (required)
--strategy   48_bottom | 64_bottom | 64_middle | 96_resize  (default: 48_bottom)
--split      lab | wild  (default: lab)
              - lab: MLP for LRRo Lab subset (48 classes)
              - wild: MLP for LRRo Wild subset (21 classes)
--top_k      Number of top predictions to display (default: 5)
--class_map  Optional JSON mapping {class_idx: word_label}
--device     cuda | cpu (default: auto-detect)
```

### Examples

```bash
# Default: 48_bottom strategy + LAB MLP
python inference_lrro.py --clip_dir /path/to/clip

# Compare all four strategies on the same clip:
for s in 48_bottom 64_bottom 64_middle 96_resize; do
    python inference_lrro.py --clip_dir /path/to/clip --strategy $s
done

# Wild MLP (21 classes)
python inference_lrro.py \
    --clip_dir /path/to/lrro/Wild_LRRo_data_set/test/<word>/<clip_id>/ \
    --split wild \
    --class_map wild_classes.json
```

## Reported results

Results from the paper, on the LRRo official `test` split:

| Method | Lab Top-1 | Lab Top-5 | Wild Top-1 | Wild Top-5 |
| --- | --- | --- | --- | --- |
| LRRo baseline (VGG-M, Inception-V4) | 71.0% | 92.0% | 33.0% | 62.0% |
| **Ours** (VSR encoder + Attn+MLP, 48_bottom) | **XX.X%** | **XX.X%** | **XX.X%** | **XX.X%** |

(Replace placeholders with your actual numbers.)

The baseline model on LRRo is trained from scratch on LRRo data only.
Our approach reuses an encoder trained on a much larger sentence-level
podcast dataset (VSRo-200) and only learns a lightweight head on LRRo,
demonstrating effective cross-domain transfer.

## Architecture

**Encoder pipeline (frozen):**
- VTP visual encoder → token embeddings per frame
- VSR transformer encoder → contextualized sequence embeddings

**Classification head (trainable, ~700k parameters):**
- Attention pooling over the temporal dimension
- Linear → BatchNorm → ReLU → Dropout(0.6) → Linear (num_classes)

The attention pooling computes per-frame importance scores and produces
a fixed-size representation per clip, regardless of clip length.

## Pre-trained MLPs

All trained MLP heads are available on HuggingFace:
[iulik-pisik/ro_vsr_classification_mlps](https://huggingface.co/iulik-pisik/ro_vsr_classification_mlps)

```
ro_vsr_classification_mlps/
├── 48_bottom/
│   ├── best_lab_clf.pt
│   └── best_wild_clf.pt
├── 64_bottom/
│   ├── best_lab_clf.pt
│   └── best_wild_clf.pt
├── 64_middle/
│   ├── best_lab_clf.pt
│   └── best_wild_clf.pt
└── 96_resize/
    ├── best_lab_clf.pt
    └── best_wild_clf.pt
```

The MLPs are downloaded automatically when running `inference_lrro.py`.
