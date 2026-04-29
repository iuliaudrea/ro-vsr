#!/usr/bin/env bash
# One-shot setup script. Run once after `git clone`, before the first
# inference call.
#
# This script downloads two external resources we depend on:
#   1. MultiVSR (https://github.com/Sindhu-Hegde/multivsr) — model
#      architecture and tokenizer, by Prajwal, Hegde & Zisserman.
#   2. VTP feature extractor (https://www.robots.ox.ac.uk/~vgg/research/
#      vtp-for-lip-reading/) — visual front-end, by Prajwal et al.
#
# These external files are placed inside our `ro_vsr/` package and
# `checkpoints/` folder for easy importing. They are NOT redistributed
# in our repository — `.gitignore` excludes them.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

# ─────────────────────────────────────────────────────────────────────
# 1. From MultiVSR (Prajwal, Hegde & Zisserman, ICASSP 2025)
# ─────────────────────────────────────────────────────────────────────
echo "[1/2] Fetching MultiVSR (Prajwal, Hegde & Zisserman, ICASSP 2025)"
echo "       Source: https://github.com/Sindhu-Hegde/multivsr"
git clone --depth 1 https://github.com/Sindhu-Hegde/multivsr.git "$TMP_DIR/multivsr" 2>&1 | tail -1

echo "[1/2] Copying MultiVSR's models.py and tokenizer.py into ro_vsr/ ..."
for f in models.py tokenizer.py; do
    cp "$TMP_DIR/multivsr/$f" "$REPO_DIR/ro_vsr/$f"
done
echo "      ✅ ro_vsr/models.py, ro_vsr/tokenizer.py (from MultiVSR)"

echo "[1/2] Copying MultiVSR's tokenizer files into ro_vsr/checkpoints/multilingual/ ..."
mkdir -p "$REPO_DIR/ro_vsr/checkpoints/multilingual"
cp -r "$TMP_DIR/multivsr/checkpoints/multilingual/." "$REPO_DIR/ro_vsr/checkpoints/multilingual/"
echo "      ✅ ro_vsr/checkpoints/multilingual/ (from MultiVSR)"

# Add get_tokenizer() if it doesn't already exist
if ! grep -q "^def get_tokenizer" "$REPO_DIR/ro_vsr/tokenizer.py"; then
    printf "\n\ndef get_tokenizer():\n    return tokenizer\n" >> "$REPO_DIR/ro_vsr/tokenizer.py"
fi

rm -rf "$TMP_DIR"

# ─────────────────────────────────────────────────────────────────────
# 2. VTP feature extractor (Prajwal et al., from VGG Oxford)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Fetching VTP feature extractor (Prajwal et al., VGG Oxford)"
echo "       Source: https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/"

mkdir -p "$REPO_DIR/checkpoints"
VTP_PATH="$REPO_DIR/checkpoints/feature_extractor.pth"

if [ -f "$VTP_PATH" ]; then
    echo "[2/2] ✅ feature_extractor.pth already exists, skipping download"
else
    echo "[2/2] Downloading ~1 GB ..."
    wget -q --show-progress -O "$VTP_PATH" \
        "https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/feature_extractor.pth"
    echo "      ✅ Saved to checkpoints/feature_extractor.pth (from VGG Oxford)"
fi

echo ""
echo "🎉 Done. You can now run:"
echo "        python inference.py --fpath samples/sample_1.avi"