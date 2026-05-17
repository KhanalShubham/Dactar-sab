import io
import os
import tempfile
import numpy as np
import pydicom
from PIL import Image, ImageDraw
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
    # Robust DICOM detection: check extension OR internal DICM signature at byte 128
    is_dicom = filename.lower().endswith((".dcm", ".dicom", ".ima")) or filename.upper().startswith("IM")
    if not is_dicom and len(file_bytes) > 132:
        is_dicom = file_bytes[128:132] == b"DICM"
    
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

def process_dicom_volume(files_data):
    """
    Processes a stack of DICOM files as a 3D volume.
    files_data: List of (bytes, filename) tuples.
    Returns (list_of_processed_bytes, volumetric_profile, shared_metadata).
    """
    processed_slices = []
    profiles = []
    shared_meta = None
    
    # Sort files by filename to ensure correct spatial order
    files_data = sorted(files_data, key=lambda x: x[1])
    
    for b, f in files_data:
        p_bytes, p_profile, p_meta = process_medical_image(b, f)
        processed_slices.append(p_bytes)
        if p_profile is not None:
            profiles.append(p_profile)
        if p_meta and not shared_meta:
            shared_meta = p_meta
            
    vol_profile = None
    if profiles:
        # Standardize profile lengths to the first profile's length using interpolation
        # This prevents ValueError if slices have inhomogeneous heights.
        target_len = len(profiles[0])
        standardized = []
        for p in profiles:
            if len(p) != target_len:
                x_old = np.linspace(0, 1, len(p))
                x_new = np.linspace(0, 1, target_len)
                p = np.interp(x_new, x_old, p)
            standardized.append(p)
            
        vol_profile = np.mean(np.array(standardized), axis=0)
        
    return processed_slices, vol_profile, shared_meta

def load_dicom_3d_volume(files_data):
    """
    Loads a list of DICOM files into a single 3D NumPy array.
    Ensures correct spatial ordering and returns metadata for aspect ratios.
    """
    # 1. Collect slices (DICOM or standard images)
    slices = []
    standard_images = []
    
    for b, f in files_data:
        is_dicom = f.lower().endswith((".dcm", ".dicom", ".ima")) or f.upper().startswith("IM")
        if is_dicom:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(b)
                    tmp_path = tmp.name
                ds = pydicom.dcmread(tmp_path)
                os.remove(tmp_path)
                slices.append(ds)
            except:
                pass
        else:
            try:
                img = Image.open(io.BytesIO(b)).convert("L")
                standard_images.append(np.array(img))
            except:
                pass
    
    # 2. Handle DICOM volume (Preferred)
    if slices:
        # Sort slices by ImagePositionPatient (Z coordinate)
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            slices.sort(key=lambda x: int(x.get("InstanceNumber", 0)))

        # Pixel spacing and slice thickness
        try:
            ps = slices[0].PixelSpacing
            st = slices[0].SliceThickness
            spacing = (float(ps[0]), float(ps[1]), float(st))
        except:
            spacing = (1.0, 1.0, 1.0)

        # Filter by shape
        shapes = [s.pixel_array.shape for s in slices]
        most_common = max(set(shapes), key=shapes.count)
        valid_pixel_arrays = [s.pixel_array for s in slices if s.pixel_array.shape == most_common]
        volume = np.stack(valid_pixel_arrays)
        
    # 3. Handle Standard Image stack (Fallback)
    elif standard_images:
        # Filter by shape
        shapes = [img.shape for img in standard_images]
        most_common = max(set(shapes), key=shapes.count)
        valid_images = [img for img in standard_images if img.shape == most_common]
        volume = np.stack(valid_images)
        spacing = (1.0, 1.0, 1.0) # Assume isotropic for screenshots
    else:
        return None, None

    # Normalize to 0-1
    try:
        volume = volume.astype(float)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
        
        return volume, spacing
    except Exception as e:
        return None, None
