"""
Generates a synthetic lumbar-spine DICOM series for Phase 3 testing.
Produces 20 axial slices: bright oval = vertebra body, dark gap = disc space.
Run: python scripts/generate_test_dicom.py
"""
import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime

OUT_DIR = os.path.join("data", "test_dicom_series")
os.makedirs(OUT_DIR, exist_ok=True)

N_SLICES = 20
IMG_SIZE = 256

# Spinal-level labels attached to slice groups (for tooltip verification)
LEVEL_LABELS = ["L1", "L1-L2", "L2", "L2-L3", "L3", "L3-L4",
                "L4", "L4-L5", "L5", "L5-S1"]


def make_vertebra_slice(idx, n_slices, img_size=256):
    """Draw a bright oval (vertebra body) with variable size and disc gaps."""
    arr = np.zeros((img_size, img_size), dtype=np.uint16)
    cx, cy = img_size // 2, img_size // 2

    # Vertebra bodies are slightly bigger at L3/L4, smaller at L1/L5
    pos = idx / n_slices  # 0 = top (L1 region), 1 = bottom (S1 region)
    rx = int(img_size * 0.18 + img_size * 0.06 * np.sin(np.pi * pos))
    ry = int(img_size * 0.14)

    # Disc space = every ~2 slices a thin dark gap
    is_disc = (idx % 2 == 1)

    Y, X = np.ogrid[:img_size, :img_size]
    mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
    arr[mask] = 800 if is_disc else 3000   # T2: disc bright-ish, bone darker

    # Add posterior arch (thin bright arc behind vertebra body)
    arch_r = int(img_size * 0.28)
    arch_mask = (((X - cx) ** 2 + (Y - cy) ** 2) >= (arch_r - 4) ** 2) & \
                (((X - cx) ** 2 + (Y - cy) ** 2) <= arch_r ** 2) & \
                (Y >= cy)
    arr[arch_mask] = 2500

    # Thecal sac (bright CSF behind vertebra)
    thecal_cx, thecal_cy = cx, cy + int(img_size * 0.10)
    tr = int(img_size * 0.05)
    t_mask = ((X - thecal_cx) ** 2 + (Y - thecal_cy) ** 2) <= tr ** 2
    arr[t_mask] = 4000

    return arr


def make_dcm(slice_idx, n_slices, series_uid, study_uid):
    img_size = IMG_SIZE
    pixel_data = make_vertebra_slice(slice_idx, n_slices, img_size)

    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = slice_idx + 1
    ds.Modality = "MR"
    ds.SeriesDescription = "Lumbar Spine T2 SAG"
    ds.PatientName = "TestPatient^SpineAI"
    ds.PatientID = "SPINE001"
    ds.PatientAge = "045Y"
    ds.PatientSex = "M"
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")

    ds.Rows = img_size
    ds.Columns = img_size
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Spatial metadata for 3D reconstruction
    z_pos = float(slice_idx) * 4.5  # 4.5 mm slice gap
    ds.ImagePositionPatient = [0.0, 0.0, z_pos]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [0.9, 0.9]
    ds.SliceThickness = 4.0
    ds.SliceLocation = z_pos

    ds.PixelData = pixel_data.astype(np.uint16).tobytes()

    return ds


def generate_series():
    study_uid = generate_uid()
    series_uid = generate_uid()

    paths = []
    for i in range(N_SLICES):
        ds = make_dcm(i, N_SLICES, series_uid, study_uid)
        path = os.path.join(OUT_DIR, f"slice_{i:03d}.dcm")
        pydicom.dcmwrite(path, ds, write_like_original=False)
        paths.append(path)

    print(f"Generated {N_SLICES} DICOM slices -> {OUT_DIR}/")
    return paths


if __name__ == "__main__":
    generate_series()
