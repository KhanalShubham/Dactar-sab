# 📇 SpineAI Reference Card

Quick reference for the SpineAI + CLIP API and common operations.

## 🧠 Core API: `clip_integration.py`

### Analyzing an MRI
```python
from clip_integration import MedicalCLIPAnalyzer

analyzer = MedicalCLIPAnalyzer()
results = analyzer.auto_tag_findings("mri_scan.png")

print(results['tags'])       # ['disc_degeneration', 'l4_l5_bulge']
print(results['confidence']) # 0.89
```

### Searching Similar Cases
```python
similar_cases = analyzer.find_similar_cases(query_image="mri_scan.png", top_k=5)
# Returns list of historical MRI paths with highest visual similarity
```

---

## 📝 Clinical Reporting: `utils.py`

### Generating a PDF
```python
from utils import ReportExporter

exporter = ReportExporter()
pdf_bytes = exporter.generate_clinical_pdf(
    patient_name="John Doe",
    findings="Mild L4-L5 disc bulge noted.",
    status="FINAL"
)
```

---

## 📋 Terminal Cheatsheet

| Command | Description |
| :--- | :--- |
| `streamlit run spineai_clip_app.py` | Start the main clinical UI |
| `python clip_training.py --data ./mri_data` | Start fine-tuning CLIP on your data |
| `python utils.py --validate-dataset` | Verify medical data integrity |

---

## ⚠️ Safety Protocols
- **CES Gate**: Always call `check_ces_risk(symptoms_list)` before allowing AI analysis.
- **Human-in-the-Loop**: AI findings must be mapped to a `DRAFT` state for manual radiologist verification.
