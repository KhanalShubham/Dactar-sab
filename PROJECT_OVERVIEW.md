# SpineAI Pro: Expert-Tier Neuroradiology Diagnostic Engine

## Project Overview
SpineAI Pro is a high-fidelity, clinical-grade diagnostic assistant designed to streamline the radiology reporting workflow. It leverages state-of-the-art vision-language models (BiomedCLIP) and large language models (LLMs) to provide automated, yet human-verifiable, medical imaging analysis.

The system is built with a "Radiologist-in-the-Loop" philosophy, ensuring that while the AI performs the heavy lifting of feature extraction and narrative drafting, the final clinical responsibility remains with the board-certified professional.

---

## Key Achievements & Features

### 1. Multi-Modal AI Analysis
- **BiomedCLIP Integration**: Utilizes zero-shot learning for anatomical and pathological tagging across multiple modalities:
    - **Spine MRI** (Primary focus)
    - **Chest X-ray**
    - **Extremity X-ray**
- **Volumetric / 3D Consensus**: Native support for DICOM stacks, performing consensus-based analysis across entire 3D volumes rather than single slices.

### 2. Clinical Reasoning Framework
- **SIR (Structured Intermediate Representation)**: A proprietary intermediate format that translates raw visual markers into structured clinical data.
- **Anatomical Constraint Engine (ACE)**: Enforces anatomical logic (e.g., ensuring findings at L5-S1 are valid for that specific level).
- **Causal Reasoning Engine**: Automatically derives the pathophysiological relationship between findings (e.g., linking facet hypertrophy to foraminal narrowing).
- **Consistency Verification**: Detects and flags logical contradictions in the analysis (e.g., "Severe stenosis" vs "Normal canal diameter").

### 3. Professional UI/UX (Clinical Dashboard)
- **Glassmorphic Design**: A premium, modern interface optimized for clinical environments.
- **WCAG AA Compliance**: High-contrast accessibility and professional typography (Plus Jakarta Sans).
- **Volumetric Viewer**: Interactive slice navigator for reviewing 3D studies.
- **Section-by-Section Review**: A rigorous workflow requiring radiologists to confirm each part of the report (Technique, Alignment, Disc Assessment, etc.).

### 4. Safety & Triage
- **CES Safety Gate**: Mandatory screening for Cauda Equina Syndrome (CES) red flags, which disables AI reporting and triggers emergency referral protocols.
- **Severity Indexing**: Real-time 1-10 scoring of cases to triage urgent findings for immediate review.

### 5. Reporting & Export
- **LLM Narrative Generation**: Uses Qwen2.5-72B-Instruct to draft professional, telegraphic-style reports that sound like a human neuroradiologist.
- **Multi-Format Export**:
    - Professional **PDF** Clinical Reports.
    - Portable **Microsoft Word (.docx)** documents.
- **Annotation Pipeline**: Automated export of images and metadata to JSONL format, ready for Label Studio/Labelbox to support future model fine-tuning.

---

## Technical Stack
- **Frontend**: Streamlit (Python-based Web Framework)
- **Vision Model**: BiomedCLIP (OpenCLIP-Torch)
- **LLM**: Qwen2.5-72B-Instruct (via Hugging Face Inference API)
- **Medical Imaging**: Pydicom, Pillow, NumPy
- **Vector DB**: FAISS (for RAG-based similar case retrieval)
- **Environment**: Containerized for easy deployment (Hugging Face Spaces, AWS)

---

## Current Status
- [x] Multi-modal support (Spine, CXR, Extremity)
- [x] Volumetric DICOM processing
- [x] Clinical UI overhaul (Pro Theme)
- [x] Causal reasoning and symbolic validation
- [x] Word & PDF export functionality
- [x] CES Safety screening implementation
- [/] FAISS-based RAG integration (Refining indexer)
- [/] Fine-tuning pipeline for custom CLIP adapters (In progress)

---

## Future Roadmap
1. **Temporal Reasoning**: Comparing current studies with prior exams to detect interval changes.
2. **Quantitative Metrics**: Automated measurement of canal diameter and listhesis millimeters.
3. **Institutional Integration**: HL7/DICOM node integration for direct PACS connectivity.
