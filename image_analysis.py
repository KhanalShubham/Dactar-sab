import io
import os
import tempfile
import numpy as np
import pydicom
import cv2
from PIL import Image, ImageDraw
import monai
from monai.transforms import LoadImage, ScaleIntensity, Resize
import logging

logger = logging.getLogger("SpineImageAnalysis")

def extract_dicom_metadata(file_bytes):
    """
    Extracts real patient and study metadata from DICOM headers.
    Returns a dict of patient info.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        ds = pydicom.dcmread(tmp_path)
        os.remove(tmp_path)
        
        return {
            "name": str(ds.get("PatientName", "Unknown")),
            "id": str(ds.get("PatientID", "N/A")),
            "age": str(ds.get("PatientAge", "N/A")),
            "sex": str(ds.get("PatientSex", "N/A")),
            "date": str(ds.get("StudyDate", "N/A")),
            "modality": str(ds.get("Modality", "MR")),
            "description": str(ds.get("SeriesDescription", "Spine MRI"))
        }
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        return None

def get_signal_intensity_profile(image_np):
    """
    Analyzes the vertical signal intensity of the spine.
    Useful for detecting disc desiccation (T2 signal loss).
    """
    # Focusing on the central 20% of the image (where the spine usually is)
    h, w = image_np.shape
    roi_w = int(w * 0.2)
    start_x = (w // 2) - (roi_w // 2)
    roi = image_np[:, start_x:start_x + roi_w]
    
    # Average intensity per horizontal slice
    profile = np.mean(roi, axis=1)
    
    # Normalize profile for plotting
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-6)
    return profile_norm

def process_medical_image(file_bytes, filename="study.png"):
    """
    Processes MRI (DICOM or standard) for real-time analysis.
    Returns: (processed_image_bytes, intensity_profile, metadata)
    """
    is_dicom = filename.lower().endswith(".dcm") or filename.upper().startswith("IM")
    
    try:
        if is_dicom:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            ds = pydicom.dcmread(tmp_path)
            pixel_array = ds.pixel_array.astype(float)
            os.remove(tmp_path)
            
            # Normalize intensity for display
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6) * 255
            image_np = pixel_array.astype(np.uint8)
            metadata = extract_dicom_metadata(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert("L")
            image_np = np.array(image)
            metadata = None

        # 1. Get real intensity curve
        intensity_profile = get_signal_intensity_profile(image_np)

        # 2. Create visual output with real-time guide
        img = Image.fromarray(image_np).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Draw automated ROI guide (Central Axis)
        draw.line([(w//2, 0), (w//2, h)], fill="cyan", width=1)
        draw.text((10, 10), "REAL-TIME SIGNAL ANALYSIS ACTIVE", fill="cyan")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        
        return buf.getvalue(), intensity_profile, metadata
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        # Fallback to standard processing if complex logic fails
        return file_bytes, None, None
