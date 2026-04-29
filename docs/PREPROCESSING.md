# Preprocessing raw video

`inference.py` expects already-preprocessed input: `.avi` clips at
**160x160** resolution with the face cropped and centered on the mouth.
This format is identical to the one used by
[MultiVSR](https://github.com/Sindhu-Hegde/multivsr).

For raw video (a YouTube clip, a personal recording, etc.), the mouth
region must first be extracted. We recommend the MultiVSR preprocessing
pipeline (adapted from [SyncNet](https://github.com/joonson/syncnet_python)).

## Steps

```bash
# 1. Clone MultiVSR
git clone https://github.com/Sindhu-Hegde/multivsr.git
cd multivsr

# 2. Install dependencies (in a separate environment from ours)
pip install -r requirements.txt

# 3. Run preprocessing
cd preprocess
python run_pipeline.py \
    --videofile /path/to/video.mp4 \
    --reference my_clip \
    --data_dir /tmp/processed
```

The output (160x160 clips) lands in `/tmp/processed/my_clip/pycrop/*.avi`.

## Then run inference

```bash
cd /path/to/ro-vsr
python inference.py --fpath /tmp/processed/my_clip/pycrop/00000.avi
```

## Limitations

- The MultiVSR pipeline auto-detects faces; if the video has multiple
  speakers, it produces one clip per person (face track).
- Very short clips (<1s) or clips with mouth occlusion will give poor
  results.
- Our model was trained on **Romanian podcasts** with a single speaker
  in frame — performance on other content types (movies, conferences,
  multilingual content) has not been evaluated.
