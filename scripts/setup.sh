#!/usr/bin/env bash
# Setup script: clonează MultiVSR și copiază fișierele de care depindem.
#
# De rulat o singură dată după git clone, înainte de prima inferență.
#
# Avem nevoie de:
#   - ro_vsr/models.py
#   - ro_vsr/tokenizer.py
#   - ro_vsr/checkpoints/multilingual/  (fișierele locale de tokenizer:
#                                        vocab.json, merges.txt, etc.
#                                        Path-ul e relativ la tokenizer.py)

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

echo "[setup] Clonez MultiVSR în $TMP_DIR ..."
git clone --depth 1 https://github.com/Sindhu-Hegde/multivsr.git "$TMP_DIR/multivsr"

echo "[setup] Copiez fișierele Python în ro_vsr/ ..."
for f in models.py tokenizer.py; do
    cp "$TMP_DIR/multivsr/$f" "$REPO_DIR/ro_vsr/$f"
    echo "        ✅ ro_vsr/$f"
done

echo "[setup] Copiez fișierele de tokenizer în ro_vsr/checkpoints/multilingual/ ..."
mkdir -p "$REPO_DIR/ro_vsr/checkpoints/multilingual"
cp -r "$TMP_DIR/multivsr/checkpoints/multilingual/." "$REPO_DIR/ro_vsr/checkpoints/multilingual/"
N_FILES=$(ls "$REPO_DIR/ro_vsr/checkpoints/multilingual" | wc -l | tr -d ' ')
echo "        ✅ ro_vsr/checkpoints/multilingual/ ($N_FILES fișiere)"

echo "[setup] Curăț ..."
rm -rf "$TMP_DIR"

echo ""
echo "[setup] ⚠️  Verifică ro_vsr/tokenizer.py — trebuie să aibă o funcție"
echo "        get_tokenizer() care returnează tokenizer-ul. Dacă nu, adaugă"
echo "        la sfârșitul fișierului:"
echo ""
echo "        def get_tokenizer():"
echo "            return tokenizer"
echo ""
echo "[setup] ✅ Gata. Acum rulează: bash scripts/download_checkpoints.sh"