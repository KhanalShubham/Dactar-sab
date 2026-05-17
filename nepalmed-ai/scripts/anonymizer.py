import pydicom
import hashlib
import os

def generate_hash(text):
    """Generates a consistent hash for anonymization."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def scrub_dicom(input_path, output_path):
    """
    Scrubs Personally Identifiable Information (PII) from a DICOM file.
    Follows the NepalMed AI 'Scrubber' protocol.
    """
    try:
        ds = pydicom.dcmread(input_path)
        
        # Original UID for hashing (to maintain longitudinal consistency)
        original_uid = ds.SOPInstanceUID
        anon_id = generate_hash(original_uid)
        
        # Mandatory Redactions
        ds.PatientName = "ANONYMOUS^NEPALMED"
        ds.PatientID = anon_id
        ds.PatientBirthDate = "" # Or keep year only if clinical
        ds.PatientSex = ds.get("PatientSex", "U") # Keep for clinical context
        
        # Clear sensitive institution/physician tags
        ds.InstitutionName = "REDACTED_HOSPITAL"
        ds.ReferringPhysicianName = "REDACTED_PHYSICIAN"
        
        # Save anonymized version
        ds.save_as(output_path)
        print(f"✅ Anonymized: {os.path.basename(input_path)} -> {anon_id}")
        return True
    except Exception as e:
        print(f"❌ Error scrubbing {input_path}: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    print("NepalMed AI: DICOM Scrubber Initialized")
