# Word-level classification on LRRo

We freeze the VTP visual encoder and our VSR transformer encoder
([MultiVSR](https://github.com/Sindhu-Hegde/multivsr) architecture
trained on the VSRo-200 dataset) and train a lightweight
attention-pooling + MLP head on LRRo. This evaluates the cross-domain
transferability of our encoder representations to a standard Romanian
lip-reading benchmark.

### ⚠️ LRRo dataset is not redistributed

LRRo (Jitaru et al., 2020) must be obtained directly from the official
source. We provide only the **inference code** and the **trained MLP
heads**; the dataset itself remains the property of its authors
(Jitaru, Abdulamit, and Ionescu — University Politehnica of Bucharest).

<!-- 
## Download LRRo

The dataset is hosted on Zenodo. The download is a RAR archive even
though it has a `.tar.gz` extension, so we use `unrar` to extract.

```bash
# 1. Install unrar (Ubuntu/Debian/Colab)
sudo apt-get install -y unrar

# 2. Download the main archive (~XGB)
mkdir -p /path/to/lrro
curl -L -o /tmp/lrro_main.tar.gz \
    "https://zenodo.org/records/3753559/files/LRRo_data_set.tar.gz"

# 3. Extract the main archive
unrar x -o+ /tmp/lrro_main.tar.gz /path/to/lrro/

# 4. Extract the two sub-archives (Lab and Wild)
for sub in /path/to/lrro/"LRRo data set"/*.tar.gz; do
    unrar x -o+ "$sub" "$(dirname "$sub")/"
done

# 5. Verify the structure
ls "/path/to/lrro/LRRo data set/Lab_LRRo_data_set/test/"
# Should list folders: ace, agresivitate, anumiti, anume, ...
```

After extraction, the structure is:

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

### Quick download check

Pick any clip from the test set to verify everything is in place:

```bash
ls "/path/to/lrro/LRRo data set/Wild_LRRo_data_set/test/" | head
ls "/path/to/lrro/LRRo data set/Wild_LRRo_data_set/test/problema/" | head -3
``` -->

## Running inference

After setting up the main repository (`bash scripts/setup.sh` from the
repo root) and downloading [LRRo](https://zenodo.org/records/3753559/files/LRRo_data_set.tar.gz):

```bash
cd evaluation/lrro_classification
python inference_lrro.py \
    --clip_dir "/path/to/lrro/LRRo data set/Wild_LRRo_data_set/test/problema/n171/" \
    --split wild
```

Expected output:

```
[device] cuda
[load] Loading VTP visual encoder ...
[load] Downloading VSR encoder from iulik-pisik/ro_vsr_150h_auto
[load] Downloading MLP from iulik-pisik/ro_vsr_classification_mlps/64_bottom/best_wild_clf.pt
[load] MLP has 21 output classes
[load] Class names auto-detected from LRRo folder structure
[infer] Running inference ...
──────────────────────────────────────────────────────────────────────
Clip:            /path/to/lrro/.../problema/n171/
Strategy:        64_bottom
MLP split:       wild  (21 classes)
True label:      problema
Top-5 predictions:
  1. problema             ████████████████████████████░░  98.43%  ←
  2. pentru               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.12%
  3. ...
──────────────────────────────────────────────────────────────────────
```

Class names are **auto-detected** from the LRRo folder structure when
the clip path is inside an LRRo dataset tree.

## Configuration

The VTP visual encoder and the VSR encoder are **fixed**. The **preprocessing strategy** and the **MLP head split** are
configurable. We trained MLPs on four different preprocessing
strategies, each with separate heads for the LAB and WILD splits.

### Preprocessing strategies

LRRo provides 64x64 grayscale mouth crops. Our VTP encoder expects
96x96 RGB face-like inputs. We tested four strategies for adapting
LRRo frames to this input format:

| Strategy | Description |
| --- | --- |
| `96_resize` | Resize directly to 96x96 |
| `64_middle` | Keep 64x64, place exactly center on 96x96 gray canvas |
| `64_bottom` | Keep 64x64, place center-bottom on 96x96 gray canvas |

### CLI options

```
--clip_dir   Folder containing the clip's .jpg frames (required)
--strategy   64_bottom | 64_middle | 96_resize  (default: 64_bottom)
--split      lab | wild  (default: lab)
              - lab: MLP for LRRo Lab subset (48 classes)
              - wild: MLP for LRRo Wild subset (21 classes)
--top_k      Number of top predictions to display (default: 5)
--class_map  Optional JSON {class_idx: word_label} (auto-detected by default)
--device     cuda | cpu (default: auto-detect)
```

Make sure to use `--split lab` for clips from `Lab_LRRo_data_set/`
and `--split wild` for clips from `Wild_LRRo_data_set/`. The script
will warn you if there's a mismatch.


## Reported results

Results from the paper, on the LRRo official `test` split:

| Method | Lab Top-1 | Lab Top-5 | Wild Top-1 | Wild Top-5 |
| --- | --- | --- | --- | --- |
| LRRo baseline (VGG-M, Inception-V4) | 71.0% | 92.0% | 33.0% | 62.0% |
| **Ours** (VSR encoder + Attn+MLP, 64_bottom) | **95.0%** | **99.4%** | **72.7%** | **92.6%** |


The LRRo baseline is trained from scratch on LRRo data only. Our
approach reuses an encoder trained on a much larger sentence-level
podcast dataset (VSRo-200) and only learns a lightweight head on
LRRo, demonstrating effective cross-domain transfer.


## Pre-trained MLPs

All trained MLP heads are available on HuggingFace:
[iulik-pisik/ro_vsr_classification_mlps](https://huggingface.co/iulik-pisik/ro_vsr_classification_mlps)

```
ro_vsr_classification_mlps/
├── 64_bottom/{best_lab_clf.pt, best_wild_clf.pt}
├── 64_middle/{best_lab_clf.pt, best_wild_clf.pt}
└── 96_resize/{best_lab_clf.pt, best_wild_clf.pt}
```

The MLPs are downloaded automatically when running `inference_lrro.py`.

## Citation

If you use this setup, please also cite the original LRRo paper:

```bibtex
@inproceedings{jitaru2020lrro,
    author    = "Jitaru, Andrei Cosmin and Abdulamit, Șeila and Ionescu, Bogdan",
    title     = "{LRRo}: A Lip Reading Data Set for the Under-resourced Romanian Language",
    booktitle = "Proceedings of the 11th ACM Multimedia Systems Conference",
    pages     = "267-272",
    year      = "2020",
}
```