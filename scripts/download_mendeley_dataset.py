"""
Downloads the Mendeley Lumbar Spine MRI Dataset (DOI: 10.17632/k57fr854j2.2)
to data/mendeley_lumbar/raw/

Usage (three modes):

  Auto (Mendeley token):
    python scripts/download_mendeley_dataset.py
    -- requires MENDELEY_TOKEN in .env or --token flag

  Manual ZIP:
    python scripts/download_mendeley_dataset.py --zip PATH_TO_DOWNLOADED.zip

  Instructions only:
    python scripts/download_mendeley_dataset.py --help-download
"""

import os
import sys
import argparse
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATASET_ID = "k57fr854j2"
DATASET_VERSION = 2
RAW_DIR = Path("data/mendeley_lumbar/raw")
MENDELEY_API = "https://api.mendeley.com"


def get_dataset_files(token: str) -> list:
    url = f"{MENDELEY_API}/datasets/{DATASET_ID}/versions/{DATASET_VERSION}"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("files", [])


def stream_download(url: str, dest: Path, token: str = None, label: str = ""):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=label or dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.infolist()
        print(f"  Extracting {len(members)} entries...")
        for m in tqdm(members, desc="Extracting", leave=False):
            z.extract(m, extract_to)
    print(f"  Extracted to: {extract_to}")


def run_api_download(token: str):
    print("Querying Mendeley Data API for file list...")
    try:
        files = get_dataset_files(token)
    except requests.HTTPError as e:
        if e.response.status_code == 401:
            print("ERROR: Authentication failed — check your MENDELEY_TOKEN.")
        else:
            print(f"ERROR: API returned {e.response.status_code}: {e}")
        print_manual_instructions()
        sys.exit(1)

    if not files:
        print("No files returned by API. The dataset structure may have changed.")
        print_manual_instructions()
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} file(s). Downloading to {RAW_DIR}/\n")

    for f in files:
        fname = f.get("filename", "file")
        size_mb = f.get("size", 0) / (1024 * 1024)
        content = f.get("content_details", {})
        dl_url = content.get("download_url", "")

        if not dl_url:
            print(f"  Skipping {fname}: no download URL in API response.")
            continue

        dest = RAW_DIR / fname
        if dest.exists():
            print(f"  Already exists: {fname} ({size_mb:.1f} MB) — skipping.")
            continue

        print(f"  Downloading: {fname} ({size_mb:.1f} MB)")
        stream_download(dl_url, dest, token=token, label=fname)

        if fname.lower().endswith(".zip"):
            folder = RAW_DIR / fname[:-4]
            print(f"  Unzipping {fname}...")
            extract_zip(dest, folder)

    print(f"\nDownload complete. Files are in: {RAW_DIR}")
    print("Next step:")
    print("  python scripts/prepare_mendeley_dataset.py")


def run_zip_import(zip_path_str: str):
    zip_path = Path(zip_path_str)
    if not zip_path.exists():
        print(f"ERROR: ZIP not found: {zip_path}")
        sys.exit(1)

    print(f"Extracting {zip_path.name} → {RAW_DIR}/")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    extract_zip(zip_path, RAW_DIR)

    dcm_count = sum(
        1
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".dcm", ".dicom", ".ima"}
    )
    print(f"\nExtracted. DICOM files found: {dcm_count}")
    print("Next step:")
    print("  python scripts/prepare_mendeley_dataset.py")


def print_manual_instructions():
    print()
    print("=" * 64)
    print("MANUAL DOWNLOAD - Mendeley Lumbar Spine MRI Dataset")
    print("=" * 64)
    print()
    print("OPTION A  - Browser download (no account needed for CC BY):")
    print("  1. Open: https://data.mendeley.com/datasets/k57fr854j2/2")
    print("  2. Click the 'Download All' button (downloads a ZIP)")
    print(f"  3. Run:")
    print(f"       python scripts/download_mendeley_dataset.py --zip <path_to_zip>")
    print()
    print("OPTION B  - API with token (fully automatic):")
    print("  1. Sign up at: https://www.mendeley.com/")
    print("  2. Go to: https://dev.mendeley.com/ -> Register Application")
    print("  3. Generate a Personal Access Token")
    print("  4. Add to .env:  MENDELEY_TOKEN=<your_token>")
    print("  5. Run: python scripts/download_mendeley_dataset.py")
    print()
    print("Dataset DOI: 10.17632/k57fr854j2.2  |  License: CC BY 4.0")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Download Mendeley Lumbar Spine MRI Dataset for SpineAI"
    )
    parser.add_argument("--token", type=str, help="Mendeley API access token")
    parser.add_argument("--zip", type=str, metavar="PATH", help="Path to manually downloaded ZIP file")
    parser.add_argument("--help-download", action="store_true", help="Print download instructions and exit")
    args = parser.parse_args()

    if args.help_download:
        print_manual_instructions()
        return

    if args.zip:
        run_zip_import(args.zip)
        return

    token = args.token or os.getenv("MENDELEY_TOKEN")
    if token:
        run_api_download(token)
    else:
        print("No MENDELEY_TOKEN found in environment and no --zip provided.")
        print_manual_instructions()


if __name__ == "__main__":
    main()
