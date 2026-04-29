#!/usr/bin/env bash
# Descarcă checkpoint-ul VTP feature extractor (preantrenat de cei de la VGG Oxford).
# Modelul enc-dec se descarcă automat de pe HuggingFace la prima inferență.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_DIR/checkpoints"

VTP_URL="https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/feature_extractor.pth"
VTP_PATH="$REPO_DIR/checkpoints/feature_extractor.pth"

if [ -f "$VTP_PATH" ]; then
    echo "[ckpt] feature_extractor.pth există deja, skip"
else
    echo "[ckpt] Descarc VTP feature extractor ..."
    wget -O "$VTP_PATH" "$VTP_URL"
    echo "[ckpt] ✅ Salvat în $VTP_PATH"
fi
