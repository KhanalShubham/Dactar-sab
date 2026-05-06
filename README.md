# SpineAI Prototype: AI-Assisted Lumbar MRI Reporting

## 🌟 Overview
**SpineAI Prototype** is a sophisticated, clinical-grade decision support tool designed to streamline the lumbar spine MRI reporting process. It combines advanced medical image analysis (MONAI), state-of-the-art LLMs (Hugging Face Qwen), and a strict safety-first workflow to help radiologists generate accurate, structured reports in minutes.

---

## 🛠️ Core Technology Stack
- **Frontend & Orchestration**: [Streamlit](https://streamlit.io/)
- **Medical Image Processing**: [MONAI](https://monai.io/) & [NumPy](https://numpy.org/)
- **Large Language Model**: [Hugging Face Inference API](https://huggingface.co/) (Model: `Qwen2.5-7B-Instruct`)
- **Computer Vision**: [Pillow (PIL)](https://python-pillow.org/) for dynamic image overlays.
- **Reporting & Export**: [FPDF](https://pyfpdf.github.io/fpdf2/) & [ReportLab](https://www.reportlab.com/) for clinical PDF generation.

---

## 🚦 Features & Workflow Implementation

### 1. Clinical Intake & Safety Gate (State: `INTAKE`)
To ensure medical safety, the application starts with a mandatory **Cauda Equina Syndrome (CES) Screening**.
- **Red Flag Detection**: Evaluates for saddle anesthesia, bladder dysfunction, and lower extremity weakness.
- **Fail-Safe Mechanism**: If any red flags are identified, the system enters a `CES_BLOCKED` state, disabling all AI features and mandating immediate neurosurgical referral.

### 2. Intelligent MRI Upload (State: `UPLOAD`)
- **Multi-Format Support**: Accepts standard medical image formats (`JPG`, `PNG`, `JPEG`).
- **Real-Time Feedback**: Provides immediate visual confirmation of the upload before processing.

### 3. Automated Image Analysis (`image_analysis.py`)
This module handles the "heavy lifting" of the image evaluation:
- **MONAI Integration**: Uses medical-grade loaders to handle MRI data and metadata.
- **Feature Extraction**: Calculates mean pixel intensities and identifies potential disc degeneration at the L4-L5 level.
- **Visual Evidence**: Automatically draws bounding boxes and diagnostic labels onto the source MRI to show the "reasoning" behind the AI findings.

### 4. AI Report Synthesis
The extracted features are passed to a highly tuned LLM prompt:
- **Clinical Persona**: The AI acts as an expert neuroradiologist.
- **Structured Output**: Generates a standard report including **FINDINGS**, **IMPRESSION**, and **Clinical Correlation**.
- **Conservative Language**: Uses medically appropriate hedging (e.g., "suggestive of") to maintain clinical accuracy.

### 5. Radiologist Review & Finalization (State: `DRAFT` & `FINAL`)
- **Human-in-the-Loop**: Radiologists can manually edit every word of the AI-generated draft.
- **Dynamic Indicators**: UI highlights findings with color-coded status icons (🟢 Normal, 🟡 Mild, 🟠 Moderate, 🔴 Severe).
- **Electronic Sign-off**: Once reviewed, the report is timestamped and signed off with the attending physician's credentials.

### 6. Clinical PDF Export
- **Professional Formatting**: Generates a headered PDF including hospital branding, patient metadata (Name, Age, ID, Study Date), and the finalized report text.
- **Legal Compliance**: Automatically includes mandatory disclaimers regarding AI-generated content.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed and create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies
```bash
pip install streamlit monai huggingface_hub fpdf reportlab pillow requests numpy
```

### 3. Launch the Application
```bash
streamlit run app.py
```

### 4. API Configuration
- Obtain a free API Token from [Hugging Face](https://huggingface.co/settings/tokens).
- Paste the token into the **Configuration** section of the app sidebar.

---

## 📂 Project Structure
- `app.py`: Main Streamlit application and UI logic.
- `image_analysis.py`: MONAI-based image processing and feature extraction.
- `test_model.py`: Utility script for testing LLM connectivity.
- `.streamlit/config.toml`: Custom theme and UI configuration.
- `.venv/`: Project dependencies and environment.

---

## 🛑 Important Disclaimer
*This application is a **prototype** intended for demonstration and research purposes only. It is not a medical device and should not be used for actual clinical diagnosis.*
