"""
Extract a face embedding for a podcast host from a folder of host images.

The embedding is the L2-normalized mean of per-image face embeddings
produced by InsightFace's `buffalo_l` model. It is used in
`preprocess_pipeline.py` (Step 2) to filter out scenes where the host
appears alone (we keep only scenes featuring the guest speaker).

Example usage:
    python extract_host_embeddings.py \
        --image_dir host_images/florin_calinescu/ \
        --output host_embeddings/florin_calinescu.npy

To process multiple hosts at once, run this script for each one;
each host needs 5–10 clear images.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def generate_embedding_from_folder(folder_path: str, app: FaceAnalysis) -> np.ndarray:
    """Compute the L2-normalized mean embedding from all valid images in a folder."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder does not exist: {folder_path}")

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not files:
        raise ValueError(f"No images found in {folder_path}")

    print(f"[host] Found {len(files)} images in '{folder_path}'")

    embeddings = []
    for filename in files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"          Could not read: {filename}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"          No face detected in: {filename}")
            continue

        # Use the largest detected face (assumed to be the host)
        main_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        embeddings.append(main_face.embedding)
        print(f"         {filename}")

    if not embeddings:
        raise RuntimeError("No valid faces extracted from any image")

    embeddings_matrix = np.array(embeddings)
    mean_embedding = np.mean(embeddings_matrix, axis=0)

    # L2 normalization (critical for cosine similarity comparisons later)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    print(f"[host] Used {len(embeddings)} / {len(files)} images for the mean embedding")
    return mean_embedding


def main():
    parser = argparse.ArgumentParser(
        description="Extract a face embedding from a folder of host images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Folder containing images of the host's face (.jpg/.png/...)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the .npy embedding file",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for InsightFace (auto picks CUDA if available)",
    )
    args = parser.parse_args()

    # Set up InsightFace
    if args.device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"[insightface] Loading buffalo_l (providers={providers})")
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    embedding = generate_embedding_from_folder(args.image_dir, app)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, embedding)
    print(f"[host]  Saved embedding to {args.output}")


if __name__ == "__main__":
    main()
