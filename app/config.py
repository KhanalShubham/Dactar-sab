class ConfigManager:
    """Handles application settings and environment variables."""
    @staticmethod
    def get_default_config():
        return {
            "app_name": "SpineAI + CLIP",
            "version": "1.0.0",
            "hf_model": "Qwen/Qwen2.5-7B-Instruct",
            "clip_model": "openai/clip-vit-base-patch32",
            "safety_gate": True
        }

REPORT_SECTIONS_BY_MODALITY = {
    "SPINE_MRI": [
        "TECHNIQUE", "VERTEBRAL ALIGNMENT", "BONE MARROW SIGNAL", "CONUS",
        "DISC ASSESSMENT", "SPINAL CANAL & THECAL SAC", "FORAMINAL ASSESSMENT",
        "FACET JOINTS", "IMPRESSION"
    ],
    "CHEST_XRAY": [
        "TECHNIQUE", "LUNGS & PLEURA", "HEART & MEDIASTINUM", "BONES & SOFT TISSUES", "IMPRESSION"
    ],
    "EXTREMITY_XRAY": [
        "TECHNIQUE", "BONES", "JOINTS", "SOFT TISSUES", "IMPRESSION"
    ]
}

# Legacy support for parts of the code still using the global list
REPORT_SECTIONS = REPORT_SECTIONS_BY_MODALITY["SPINE_MRI"]
