---
title: Dactar Sab
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.1
app_file: app.py
pinned: false
---

# SpineAI Pro: Expert-Tier Neuroradiology Reasoning Engine

SpineAI Pro is a clinically-grounded artificial intelligence platform designed for the advanced interpretation of Lumbar Spine MRI. Unlike generic generative AI, SpineAI Pro utilizes a **Neuro-Symbolic Architecture** that prioritizes anatomical consistency, pathophysiological causality, and professional neuroradiological standards.

## 🧠 Core Architecture: The SIR-Driven Pipeline

The system has transitioned from a basic "image-to-text" model to a sophisticated 4-stage reasoning pipeline:

1.  **Volumetric Feature Extraction**: BiomedCLIP-powered consensus analysis across 3D MRI stacks (Slices Agreement Validation).
2.  **Anatomical Constraint Engine (ACE)**: Enforces hard neuroanatomical rules (e.g., re-mapping "spinal cord" to "thecal sac" below L2) and maps pathology to specific exiting/traversing nerve roots.
3.  **Symbolic Causal Validator (SCV)**: Cross-references findings to ensure clinical consistency. It enforces the rule that **Stenosis requires a cause** (e.g., if severe stenosis is detected without an underlying bulge or facet hypertrophy, the finding is suppressed).
4.  **SIR-Driven Narrative Generation**: Reports are generated strictly from a **Structured Intermediate Representation (SIR)** JSON, ensuring 100% internal coherence between the data and the final clinical report.

## 🏥 Key Clinical Features

-   **Expert Root Mapping**: Precise differentiation between **Exiting** and **Traversing** nerve roots at each spinal level.
-   **Lateral Recess Assessment**: Dedicated logic for identifying narrowing in the lateral recess, a critical zone for traversing nerve root pathology.
-   **Quantitative Evidence Suite**: Provides pseudo-measurable clinical metrics:
    - AP Canal Diameter (mm)
    - Dural Sac Cross-Sectional Area (CSA)
    - CSF Preservation (%)
    - Foraminal Height Reduction (%)
-   **Temporal Reasoning**: A "Compare to Prior MRI" mode that phrases findings in terms of interval stability or progression.
-   **Silent AI Tone**: Professional, telegraphic radiology narrative purged of AI meta-language (e.g., "AI suggests") to mirror real neuroradiology reports.

## 🚀 Hosting & Deployment

SpineAI is built using **Streamlit** and **Python**. It can be hosted on platforms like:
- **Streamlit Community Cloud** (Easiest)
- **Hugging Face Spaces** (Recommended for AI)
- **Render / Railway / AWS**

For a detailed breakdown of the technologies used and a step-by-step hosting guide, see:
👉 **[HOSTING_GUIDE.md](./HOSTING_GUIDE.md)**

## 🛠 Setup & Tech Stack

- **Core:** Python 3.9+
- **Frontend:** Streamlit
- **AI Models:** Hugging Face Inference API (BiomedCLIP, Qwen 2.5)
- **Medical Imaging:** pydicom, MONAI

### Quick Local Start
1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with your `HF_TOKEN`.
3. Run the app: `streamlit run app.py`

## 📋 Clinical Workflow

1.  **Phase 1: Triage**: Patient data entry and multi-plane MRI upload.
2.  **Phase 2: 3D Analysis**: Volumetric consensus analysis and causal validation.
3.  **Phase 3: Radiologist Review**: Level-by-level verification, editing, and confirmation of each report section.
4.  **Phase 4: Finalization**: Digital signature and export to EMR/PACS-ready JSON or PDF formats.

## ⚠️ Medical Disclaimer

SpineAI Pro is a **Clinical Decision Support (CDS)** tool. It is designed to assist radiologists, not replace them. All findings must be reviewed and verified by a board-certified radiologist. The quantitative metrics are estimated based on visual confidence and should be used for screening purposes only.
