"""
Preprocesses the Mendeley Lumbar Spine MRI Dataset for SpineAI integration.

Pipeline:
  1. Scans data/mendeley_lumbar/raw/ for all DICOM files recursively
  2. Converts each DICOM to a normalized 8-bit PNG
  3. Generates a training JSONL for CLIP fine-tuning
  4. Builds/updates the FAISS similarity index used by the app

Usage:
  python scripts/prepare_mendeley_dataset.py
  python scripts/prepare_mendeley_dataset.py --raw_dir /custom/path --no_index
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pydicom
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analyzer.clip_analyzer import MedicalCLIPAnalyzer

RAW_DIR = Path("data/mendeley_lumbar/raw")
PROCESSED_DIR = Path("data/mendeley_lumbar/processed")
JSONL_PATH = Path("data/mendeley_lumbar/training_data.jsonl")
INDEX_DIR = Path("data/embeddings_cache")

DICOM_EXTS = {".dcm", ".dicom", ".ima"}


# ─── DICOM helpers ────────────────────────────────────────────────────────────

def apply_window(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply DICOM WindowCenter/WindowWidth, or auto-window from percentiles."""
    arr = pixel_array.astype(float)

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        wc = float(wc[0]) if hasattr(wc, "__iter__") else float(wc)
        ww = float(ww[0]) if hasattr(ww, "__iter__") else float(ww)
        low, high = wc - ww / 2.0, wc + ww / 2.0
    else:
        low = np.percentile(arr, 1)
        high = np.percentile(arr, 99)

    arr = np.clip(arr, low, high)
    if high > low:
        arr = (arr - low) / (high - low) * 255.0
    return arr.astype(np.uint8)


def dicom_to_png(dcm_path: Path, out_path: Path) -> pydicom.Dataset:
    ds = pydicom.dcmread(str(dcm_path))
    pixel_array = ds.pixel_array

    if pixel_array.ndim == 3:
        pixel_array = pixel_array[0]

    windowed = apply_window(pixel_array, ds)
    img = Image.fromarray(windowed).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG", optimize=False)
    return ds


def is_dicom(path: Path) -> bool:
    if path.suffix.lower() in DICOM_EXTS:
        return True
    if path.suffix == "":
        try:
            with open(path, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except OSError:
            return False
    return False


def find_dicoms(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and is_dicom(p)]


# ─── Report text generation ───────────────────────────────────────────────────

def make_report_text(ds: pydicom.Dataset) -> str:
    """Generate a structured radiology-style text label from DICOM tags."""
    series_desc = str(getattr(ds, "SeriesDescription", "")).lower()
    body_part = str(getattr(ds, "BodyPartExamined", "lumbar spine")).lower()
    modality = str(getattr(ds, "Modality", "MR"))

    view = "sagittal"
    if any(k in series_desc for k in ("axial", " ax ", "_ax_", "tra")):
        view = "axial"
    elif "cor" in series_desc:
        view = "coronal"

    seq = "t2-weighted"
    if "t1" in series_desc:
        seq = "t1-weighted"
    elif any(k in series_desc for k in ("stir", "pd", "flair")):
        seq = "pd-weighted"

    return (
        f"{view} {seq} lumbar spine {modality} showing the {body_part}, "
        "including vertebral bodies, intervertebral discs, spinal canal, "
        "thecal sac, and neural foramina. Assessment for disc degeneration, "
        "canal stenosis, and foraminal narrowing at L3-S1 levels."
    )


def make_report_text_from_path(path: Path) -> str:
    name = str(path).lower()
    view = "axial" if any(k in name for k in ("ax", "tra")) else "sagittal"
    return (
        f"{view} t2-weighted lumbar spine MRI showing vertebral bodies, "
        "intervertebral discs, spinal canal, and neural foramina at L3-S1 levels."
    )


# ─── FAISS index builder ──────────────────────────────────────────────────────

def build_faiss_index(png_paths: list[Path], index_dir: Path):
    print(f"\nBuilding FAISS index from {len(png_paths)} images...")
    analyzer = MedicalCLIPAnalyzer(cache_dir=str(index_dir))

    embeddings, metadata, errors = [], [], 0

    for png_path in tqdm(png_paths, desc="Embedding"):
        try:
            with open(png_path, "rb") as f:
                img_bytes = f.read()
            emb = analyzer.get_image_embedding(img_bytes)
            embeddings.append(emb[0])
            metadata.append({"path": str(png_path), "source": "mendeley_lumbar_v2"})
        except Exception:
            errors += 1

    if not embeddings:
        print("No embeddings extracted — skipping index build.")
        return

    emb_np = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "medical_cases.index"))
    with open(index_dir / "medical_cases.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"FAISS index: {len(metadata)} cases indexed  (errors: {errors})")
    print(f"Saved to: {index_dir}/")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(raw_dir: Path, processed_dir: Path, jsonl_path: Path, index_dir: Path, build_index: bool):
    print(f"Scanning: {raw_dir}")
    dcm_files = find_dicoms(raw_dir)

    if not dcm_files:
        print(f"\nNo DICOM files found in {raw_dir}")
        print("Run the download script first:")
        print("  python scripts/download_mendeley_dataset.py")
        sys.exit(1)

    print(f"Found {len(dcm_files)} DICOM files. Converting to PNG...\n")

    jsonl_entries, processed_pngs, skipped, errors = [], [], 0, 0

    for dcm_path in tqdm(dcm_files, desc="DICOM → PNG"):
        try:
            rel = dcm_path.relative_to(raw_dir)
        except ValueError:
            rel = Path(dcm_path.name)

        out_path = processed_dir / rel.with_suffix(".png")

        if out_path.exists():
            skipped += 1
            processed_pngs.append(out_path)
            jsonl_entries.append({
                "image": str(out_path),
                "report": make_report_text_from_path(dcm_path),
            })
            continue

        try:
            ds = dicom_to_png(dcm_path, out_path)
            jsonl_entries.append({
                "image": str(out_path),
                "report": make_report_text(ds),
            })
            processed_pngs.append(out_path)
        except Exception as e:
            errors += 1

    print(f"\nConverted: {len(processed_pngs) - skipped}  |  Skipped (cached): {skipped}  |  Errors: {errors}")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Training JSONL: {jsonl_path}  ({len(jsonl_entries)} entries)")

    if build_index and processed_pngs:
        build_faiss_index(processed_pngs, index_dir)
    elif not build_index:
        print("\nSkipped FAISS index build (--no_index flag).")
        print("To build later:")
        print(f"  python scripts/index_cases.py --dir {processed_dir}")

    print("\nIntegration complete.")
    print(f"  Processed images : {processed_dir}/")
    print(f"  Training JSONL   : {jsonl_path}")
    print(f"  FAISS index      : {index_dir}/")
    print("\nTo fine-tune BiomedCLIP on this dataset:")
    print(f"  python scripts/clip_training.py --train_data {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Mendeley Lumbar MRI for SpineAI fine-tuning and RAG"
    )
    parser.add_argument("--raw_dir", default=str(RAW_DIR), help="Directory with raw DICOM files")
    parser.add_argument("--processed_dir", default=str(PROCESSED_DIR), help="Output directory for PNGs")
    parser.add_argument("--jsonl_out", default=str(JSONL_PATH), help="Output path for training JSONL")
    parser.add_argument("--index_dir", default=str(INDEX_DIR), help="Directory for FAISS index output")
    parser.add_argument("--no_index", action="store_true", help="Skip FAISS index building (faster)")
    args = parser.parse_args()

    run(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        jsonl_path=Path(args.jsonl_out),
        index_dir=Path(args.index_dir),
        build_index=not args.no_index,
    )


if __name__ == "__main__":
    main()
