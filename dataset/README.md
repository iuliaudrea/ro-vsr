# VSRo-200 Dataset

This folder contains the scripts needed to **reconstruct the VSRo-200
dataset** from its source YouTube videos.

We do not redistribute the source videos themselves. Instead, we
publish on HuggingFace the timestamps and transcripts that define the
dataset, and provide here the scripts to download and cut the clips
from YouTube.

## Dataset on HuggingFace

The VSRo-200 metadata is hosted at
[vsro200/vsro200](https://huggingface.co/datasets/vsro200/vsro200). Each split is a CSV with
the format:

```csv
file_path,start,end,transcript,gender
y7j5T-AHxTc/00001,12.450,28.180,"...",F
y7j5T-AHxTc/00002,45.020,52.860,"...",F
```

Where:

- `file_path` is `<youtube_id>/<5-digit-index>` and identifies each clip
- `start`, `end` are timestamps in seconds within the source video
- `transcript` is the Romanian transcription
- `gender` is the speaker's gender (M/F); replaced by `domain` for the
  `test_ood` split

Available splits:
| Split | Description | Extra columns |
| --- | --- | --- |
| `trainval_annot.csv` | Manually annotated train/val | gender |
| `trainval_auto.csv` | Auto-transcribed train/val (Whisper-large) | gender |
| `test_seen.csv` | Test split, speakers seen during training | gender |
| `test_unseen.csv` | Test split, unseen speakers | gender |
| `test_ood.csv` | Out-of-domain test (different content type) | domain |

## Reconstruction workflow

### Dependencies

```bash
pip install yt-dlp pandas
```

You'll also need `ffmpeg` available on your system PATH.

### Step 1 — Download the timestamps CSV(s) from HuggingFace

Pick whichever splits you need

```bash
pip install -U "huggingface_hub[cli]"

# Download a single split
huggingface-cli download vsro200/vsro200 test_seen.csv \
    --repo-type dataset --local-dir ./

# Or download all CSVs at once
huggingface-cli download vsro200/vsro200 \
    --repo-type dataset --local-dir ./ \
    --include "*.csv"
```

### Step 2 — Download the source YouTube videos

```bash
python download_videos.py --ids youtube_ids.txt --output_dir videos/
```

This downloads each video listed in `youtube_ids.txt` as
`videos/<youtube_id>.mp4`. Videos that are already downloaded are
skipped.

### Step 3 — Cut the clips

For each split you downloaded:

```bash
python cut_clips.py \
    --csv test_seen.csv \
    --videos_dir videos/ \
    --output_dir clips/test_seen/
```

The output is laid out as:

```
clips/test_seen/
├── ZWNM2sZxtRg/
│   ├── 00001.mp4
│   ├── 00002.mp4
│   └── ...
├── abc123def/
│   ├── 00001.mp4
│   └── ...
```

If a source video is missing (e.g., removed from YouTube), the script
errors out by default. Pass `--skip_missing` to skip those clips
silently and continue with what's available.

### Step 4 — Extract face tracks for VSR

Once you have guest-only clips, feed them to MultiVSR's pipeline to
produce the 224×224 .avi face crops used by our inference scripts:

```bash
git clone https://github.com/Sindhu-Hegde/multivsr.git /tmp/multivsr
cd /tmp/multivsr/preprocess

for clip in /path/to/clips/test_seen/*/*.mp4; do
    base=$(dirname "$clip" | xargs basename)
    name=$(basename "$clip" .mp4)
    python run_pipeline.py \
        --videofile "$clip" \
        --reference "${base}_${name}" \
        --data_dir /path/to/face_tracks/
done
```

The output face tracks land in
`/path/to/face_tracks/<reference>/pycrop/*.avi`, ready to be passed
to our `inference.py`.

