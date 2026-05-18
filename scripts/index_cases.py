import os
import io
import argparse
import sys
import tempfile
from tqdm import tqdm
import faiss
import pickle
import numpy as np
import pydicom
from PIL import Image

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analyzer.clip_analyzer import MedicalCLIPAnalyzer

RASTER_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
DICOM_EXTS = (".dcm", ".dicom", ".ima")


def dicom_bytes_to_png_bytes(raw_bytes: bytes) -> bytes:
    """Convert DICOM file bytes to normalized PNG bytes for BiomedCLIP."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        ds = pydicom.dcmread(tmp_path)
        arr = ds.pixel_array.astype(float)
        if arr.ndim == 3:
            arr = arr[0]
        low, high = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip(arr, low, high)
        if high > low:
            arr = (arr - low) / (high - low) * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        os.remove(tmp_path)


def collect_image_paths(image_dir: str) -> list:
    """Collect all raster and DICOM image paths under image_dir (recursive)."""
    paths = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in RASTER_EXTS or ext in DICOM_EXTS:
                paths.append(os.path.join(root, fname))
    return paths


def index_directory(image_dir, output_dir="./data/rag_index"):
    """
    Scans a directory of images (PNG/JPG or DICOM), extracts CLIP embeddings,
    and saves a FAISS index.
    """
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    analyzer = MedicalCLIPAnalyzer(cache_dir=output_dir)

    image_paths = collect_image_paths(image_dir)

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Indexing {len(image_paths)} images (raster + DICOM)...")

    embeddings = []
    metadata = []

    for path in tqdm(image_paths):
        try:
            with open(path, "rb") as f:
                raw = f.read()

            ext = os.path.splitext(path)[1].lower()
            img_bytes = dicom_bytes_to_png_bytes(raw) if ext in DICOM_EXTS else raw

            emb = analyzer.get_image_embedding(img_bytes)
            embeddings.append(emb[0])
            metadata.append({"path": path})
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not embeddings:
        print("No embeddings were successfully extracted.")
        return

    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]

    # Create FAISS index (Inner Product for cosine similarity since vectors are normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    # Save index and metadata
    index_path = os.path.join(output_dir, "medical_cases.index")
    meta_path = os.path.join(output_dir, "medical_cases.pkl")
    
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nSuccessfully indexed {len(metadata)} cases.")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index historical MRI images for similarity search.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing MRI images.")
    parser.add_argument("--out", type=str, default="./data/rag_index", help="Output directory for index.")
    
    args = parser.parse_args()
    index_directory(args.dir, args.out)
