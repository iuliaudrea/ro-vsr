#!/usr/bin/env bash
# Descarcă câteva clipuri demo de pe HuggingFace pentru testare rapidă.
#
# Sample-urile sunt clipuri AVI scurte (160x160, deja preprocesate)
# extrase din test_seen / test_unseen ale dataset-ului ro_vsr.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLES_DIR="$REPO_DIR/samples"
mkdir -p "$SAMPLES_DIR"

# TODO: înlocuiește cu URL-urile reale către sample-urile pe care le încarci
# pe HuggingFace, în repo-ul iulik-pisik/ro_vsr (folder samples/) sau într-un
# repo dedicat iulik-pisik/ro_vsr_demo.

BASE_URL="https://huggingface.co/datasets/iulik-pisik/ro_vsr/resolve/main/samples"

SAMPLES=(
    "sample_1.avi"
    "sample_2.avi"
    "sample_3.avi"
    "sample_4.avi"
    "sample_5.avi"
)

for s in "${SAMPLES[@]}"; do
    if [ -f "$SAMPLES_DIR/$s" ]; then
        echo "[samples] $s există deja, skip"
    else
        echo "[samples] Descarc $s ..."
        wget -q -O "$SAMPLES_DIR/$s" "$BASE_URL/$s"
        echo "          ✅ samples/$s"
    fi
done

# Și CSV-ul cu metadate
META_URL="$BASE_URL/samples_metadata.csv"
if [ ! -f "$SAMPLES_DIR/samples_metadata.csv" ]; then
    wget -q -O "$SAMPLES_DIR/samples_metadata.csv" "$META_URL"
    echo "[samples] ✅ samples/samples_metadata.csv"
fi

echo "[samples] ✅ Toate sample-urile descărcate"
