#!/usr/bin/env bash
# Script unic de setup: clonează MultiVSR (cod arhitectură + tokenizer)
# și descarcă VTP feature extractor de la VGG Oxford.
#
# De rulat o singură dată după git clone, înainte de prima inferență.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

# ─────────────────────────────────────────────────────────────────────
# 1. Clonează MultiVSR și copiază fișierele de care depindem
# ─────────────────────────────────────────────────────────────────────
echo "[setup] Clonez MultiVSR ..."
git clone --depth 1 https://github.com/Sindhu-Hegde/multivsr.git "$TMP_DIR/multivsr" 2>&1 | tail -1

echo "[setup] Copiez fișierele Python în ro_vsr/ ..."
for f in models.py tokenizer.py; do
    cp "$TMP_DIR/multivsr/$f" "$REPO_DIR/ro_vsr/$f"
done
echo "        ✅ ro_vsr/models.py, ro_vsr/tokenizer.py"

echo "[setup] Copiez fișierele de tokenizer ..."
mkdir -p "$REPO_DIR/ro_vsr/checkpoints/multilingual"
cp -r "$TMP_DIR/multivsr/checkpoints/multilingual/." "$REPO_DIR/ro_vsr/checkpoints/multilingual/"
echo "        ✅ ro_vsr/checkpoints/multilingual/"

# Adaugă funcția get_tokenizer() dacă nu există
if ! grep -q "^def get_tokenizer" "$REPO_DIR/ro_vsr/tokenizer.py"; then
    printf "\n\ndef get_tokenizer():\n    return tokenizer\n" >> "$REPO_DIR/ro_vsr/tokenizer.py"
fi

rm -rf "$TMP_DIR"

# ─────────────────────────────────────────────────────────────────────
# 2. Descarcă VTP feature extractor (de la VGG Oxford)
# ─────────────────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/checkpoints"
VTP_PATH="$REPO_DIR/checkpoints/feature_extractor.pth"

if [ -f "$VTP_PATH" ]; then
    echo "[setup] ✅ feature_extractor.pth există deja"
else
    echo "[setup] Descarc VTP feature extractor (~1 GB) ..."
    wget -q --show-progress -O "$VTP_PATH" \
        "https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/feature_extractor.pth"
    echo "        ✅ checkpoints/feature_extractor.pth"
fi

echo ""
echo "[setup] 🎉 Gata. Acum poți rula:"
echo "        python inference.py --fpath samples/sample_1.avi"