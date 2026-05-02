# VSRo-200 Dataset Construction

This folder contains the scripts used to build the **VSRo-200** dataset
from raw Romanian podcast videos on YouTube.

The pipeline takes a list of podcast YouTube IDs and produces clean
guest-only video clips, ready to be passed through the
[MultiVSR face-tracking pipeline](https://github.com/Sindhu-Hegde/multivsr/tree/master/preprocess)
for the final 160×160 .avi face crops.

<p align="center">
    <img src="vsro_dataset.gif"/>
</p>

## Pipeline overview

```
        Raw YouTube podcasts
                │
                ▼  Step 1: download_video.py        (yt-dlp)
            Local .mp4 files
                │
                ▼  Step 2: extract_host_embeddings.py  (per host, one-time)
            host_embeddings/*.npy
                │
                ▼  Step 3: preprocess_pipeline.py
                ├─ PySceneDetect     (scenes)
                ├─ InsightFace       (filter: guest only, single face)
                └─ pyannote          (filter: guest speaks, no overlap)
                │
                ▼
          timestamps CSV (start, end per guest interval)
                │
                ▼  Step 4: cut_clips.py             (ffmpeg)
            Guest-only .mp4 clips
                │
                ▼  Step 5: MultiVSR run_pipeline.py (external)
            Face tracks (224×224 .avi)
```

## Files in this folder

| File | Purpose |
| --- | --- |
| `podcast_ids.txt` | List of YouTube IDs used to build VSRo-200 (one per line) |
| `download_video.py` | Download a single YouTube video by ID |
| `extract_host_embeddings.py` | Compute a host's face embedding from a folder of images |
| `preprocess_pipeline.py` | 3-step pipeline → CSV of guest-only timestamps |
| `cut_clips.py` | Cut MP4 clips from a video using the timestamps CSV |

## Dependencies

```bash
pip install scenedetect[opencv]
pip install insightface onnxruntime-gpu
pip install pyannote.audio
pip install yt-dlp
pip install pandas
```

You will also need `ffmpeg` available on your system PATH.

## Step 1 — Download YouTube videos

Use `download_video.py` for a single video:

```bash
python download_video.py --video_id <VIDEO_ID> --output_dir videos/
```

To download every podcast in `podcast_ids.txt`:

```bash
mkdir -p videos
while IFS= read -r id; do
    [[ "$id" =~ ^#.*$ || -z "$id" ]] && continue
    python download_video.py --video_id "$id" --output_dir videos/
done < podcast_ids.txt
```


## Step 2 — Extract host embeddings

For each host that may appear in your podcasts (the show's main
presenter), prepare a folder with **5 clear images** of
their face. Then run:

```bash
python extract_host_embeddings.py \
    --image_dir host_images/florin_calinescu/ \
    --output host_embeddings/florin_calinescu.npy
```

Repeat for each host. The pipeline filter in Step 3 will match any
face against these embeddings; matching faces are treated as the host
and rejected, keeping only guest-speaking moments.

> **Why we don't ship the embeddings.** Face embeddings derived from
> real people are biometric data subject to GDPR. We provide the
> extraction code so users generate embeddings for their own podcasts
> on their own machines.

## Step 3 — Run the preprocessing pipeline

This produces a CSV of (start, end) timestamps marking guest-only
speaking intervals.

**First, set your HuggingFace token** (required by `pyannote`):

```bash
export HF_TOKEN=hf_xxxxxxxxxx
```

Get a token at https://huggingface.co/settings/tokens and accept the
license at
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

Then on a single video:

```bash
python preprocess_pipeline.py \
    --video_path videos/abc123.mp4 \
    --host_embeddings host_embeddings/ \
    --output_csv timestamps/abc123.csv
```

To process all downloaded podcasts:

```bash
mkdir -p timestamps
for video in videos/*.mp4; do
    base=$(basename "$video" .mp4)
    python preprocess_pipeline.py \
        --video_path "$video" \
        --host_embeddings host_embeddings/ \
        --output_csv "timestamps/${base}.csv"
done
```

### What the pipeline does

1. **PySceneDetect** — splits the video into scenes whenever the camera
   cuts (typically host ↔ guest in a podcast).
2. **InsightFace** — for each scene, samples the midpoint frame and
   keeps it only if (a) exactly one face is visible AND (b) that face
   does not match any known host embedding.
3. **pyannote** — runs speaker diarization within each kept scene and
   keeps only the intervals where exactly one speaker is active (drops
   overlapped speech where the host interjects). Adjacent intervals
   from the same speaker are merged when separated by short pauses
   (≤ 1 s); intervals are split at safe cut-points if longer than
   30 seconds; intervals shorter than 3 seconds are dropped.

The output CSV looks like:

```csv
clip_id,start,end,duration,speaker
abc123_clip_1,12.450,28.180,15.730,SPEAKER_01
abc123_clip_2,45.020,52.860,7.840,SPEAKER_01
...
```

## Step 4 — Cut the clips

Once you have a timestamps CSV, extract the actual .mp4 clips:

```bash
python cut_clips.py \
    --video_path videos/abc123.mp4 \
    --timestamps_csv timestamps/abc123.csv \
    --output_dir clips/abc123/
```

The clips are cut with `ffmpeg` using a hybrid seek strategy
(5-second pre-roll buffer + ultrafast x264 re-encoding) for
frame-accurate cuts without black frames at clip boundaries.

To process the full dataset:

```bash
mkdir -p clips
for csv in timestamps/*.csv; do
    base=$(basename "$csv" .csv)
    python cut_clips.py \
        --video_path "videos/${base}.mp4" \
        --timestamps_csv "$csv" \
        --output_dir "clips/${base}/"
done
```

## Step 5 — Extract face tracks (MultiVSR pipeline)

Once you have guest-only clips, feed them to MultiVSR's pipeline to
produce the 160×160 .avi face crops used for inference:

```bash
git clone https://github.com/Sindhu-Hegde/multivsr.git /tmp/multivsr
cd /tmp/multivsr/preprocess

for clip in /path/to/clips/*/*.mp4; do
    base=$(basename "$clip" .mp4)
    python run_pipeline.py \
        --videofile "$clip" \
        --reference "$base" \
        --data_dir /path/to/face_tracks/
done
```

The output 160×160 face tracks land in
`/path/to/face_tracks/<reference>/pycrop/*.avi`.

## License

Source videos remain the property of their original YouTube creators.
