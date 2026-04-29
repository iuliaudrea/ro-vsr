#!/usr/bin/env bash
# One-shot setup script: clones MultiVSR (architecture code + tokenizer)
# and downloads the VTP feature extractor from VGG Oxford.
#
# Run once after `git clone`, before the first inference call.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

# ─────────────────────────────────────────────────────────────────────
# 1. Clone MultiVSR and copy the files we depend on
# ─────────────────────────────────────────────────────────────────────
echo "[setup] Cloning MultiVSR (Prajwal et al., 2025) ..."
git clone --depth 1 https://github.com/Sindhu-Hegde/multivsr.git "$TMP_DIR/multivsr" 2>&1 | tail -1

echo "[setup] Copying Python files into ro_vsr/ ..."
for f in models.py tokenizer.py; do
    cp "$TMP_DIR/multivsr/$f" "$REPO_DIR/ro_vsr/$f"
done
echo "        ✅ ro_vsr/models.py, ro_vsr/tokenizer.py"

echo "[setup] Copying tokenizer files ..."
mkdir -p "$REPO_DIR/ro_vsr/checkpoints/multilingual"
cp -r "$TMP_DIR/multivsr/checkpoints/multilingual/." "$REPO_DIR/ro_vsr/checkpoints/multilingual/"
echo "        ✅ ro_vsr/checkpoints/multilingual/"

# Add get_tokenizer() if it doesn't already exist
if ! grep -q "^def get_tokenizer" "$REPO_DIR/ro_vsr/tokenizer.py"; then
    printf "\n\ndef get_tokenizer():\n    return tokenizer\n" >> "$REPO_DIR/ro_vsr/tokenizer.py"
fi

rm -rf "$TMP_DIR"

# ─────────────────────────────────────────────────────────────────────
# 2. Download the VTP feature extractor from VGG Oxford
# ─────────────────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/checkpoints"
VTP_PATH="$REPO_DIR/checkpoints/feature_extractor.pth"

if [ -f "$VTP_PATH" ]; then
    echo "[setup] ✅ feature_extractor.pth already exists, skipping download"
else
    echo "[setup] Downloading VTP feature extractor from VGG Oxford (~1 GB) ..."
    wget -q --show-progress -O "$VTP_PATH" \
        "https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/feature_extractor.pth"
    echo "        ✅ Saved to checkpoints/feature_extractor.pth"
fi

echo ""
echo "[setup] 🎉 Done. You can now run:"
echo "        python inference.py --fpath samples/sample_1.avi"
