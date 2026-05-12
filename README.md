# 🏥 SpineAI Pro: Comprehensive Project Documentation

## 🌟 Project Vision & Core Concept
**SpineAI Pro** (also known as *Dactar Sab*) is a clinically-grounded artificial intelligence platform designed for the advanced interpretation of Lumbar Spine MRI. Unlike generic generative AI, SpineAI Pro utilizes a **Neuro-Symbolic Architecture** that prioritizes anatomical consistency, pathophysiological causality, and professional neuroradiological standards.

The project bridges the gap between raw medical pixels and professional clinical language, providing a "silent AI" experience that mirrors real-world neuroradiology workflows.

---

## 🧠 Core Architecture: The SIR-Driven Pipeline
The system utilizes a sophisticated 4-stage reasoning pipeline to ensure accuracy and clinical relevance:

1.  **Volumetric Feature Extraction**: BiomedCLIP-powered consensus analysis across 3D MRI stacks (Slices Agreement Validation).
2.  **Anatomical Constraint Engine (ACE)**: Enforces hard neuroanatomical rules (e.g., mapping pathology to specific exiting/traversing nerve roots) and validates boundaries.
3.  **Symbolic Causal Validator (SCV)**: Cross-references findings to ensure clinical consistency. It enforces the rule that **Stenosis requires a cause** (e.g., suppressing "severe stenosis" if no underlying bulge or facet hypertrophy is detected).
4.  **SIR-Driven Narrative Generation**: Reports are generated strictly from a **Structured Intermediate Representation (SIR)** JSON, ensuring 100% internal coherence between the data and the final clinical report.

### The 3-Tier Execution Layer
- **Tier 1: Safety & Intake**: Deterministic screening for **Cauda Equina Syndrome (CES)**. If clinical red flags exist, the AI engine is physically prevented from running to ensure patient safety.
- **Tier 2: Analysis Engine**: MRI analysis via the `MedicalCLIPAnalyzer`, performing **Zero-Shot Tagging** against a library of 50+ spine findings.
- **Tier 3: Narrative Generator**: Transforms raw tags into a flowing, professional medical narrative using LLMs (Qwen-2.5) with specialized radiologist system prompts.

---

## 🏥 Key Clinical Features
- **Expert Root Mapping**: Precise differentiation between **Exiting** and **Traversing** nerve roots at each spinal level.
- **Lateral Recess Assessment**: Dedicated logic for identifying narrowing in the lateral recess.
- **Quantitative Evidence Suite**: Provides pseudo-measurable clinical metrics:
    - AP Canal Diameter (mm)
    - Dural Sac Cross-Sectional Area (CSA)
    - CSF Preservation (%)
    - Foraminal Height Reduction (%)
- **Temporal Reasoning**: A "Compare to Prior MRI" mode for assessing interval stability or progression.
- **Silent AI Tone**: Professional, telegraphic radiology narrative purged of AI meta-language (e.g., "AI suggests").

---

## 📂 Project Structure
The project follows a modular, scalable architecture to separate concerns and prepare for advanced features like RAG.

```text
SpineAI_Pro/                      # Project root
├── .env                          # Environment variables (HF_TOKEN, etc.)
├── .gitignore
├── README.md                     # This file
├── requirements.txt
├── app/                          # Streamlit frontend & main application
│   ├── app.py                    # Main Streamlit entry point
│   └── config.py                 # Configuration (paths, model names)
├── src/                          # Core backend modules
│   ├── analyzer/                 # Vision & tagging (BiomedCLIP)
│   ├── constraints/              # ACE, SCV, consistency engines
│   ├── rag/                      # Nepal-specific RAG module (NEW)
│   ├── narrative/                # LLM report generation
│   ├── export/                   # PDF & JSON export
│   └── utils/                    # Helpers (DICOM, metrics, etc.)
├── data/                         # All static data & indexes
│   ├── nepal_guidelines/         # Nepal-specific clinical sources
│   ├── rag_index/                # FAISS indexes
│   ├── embeddings_cache/         # CLIP cache
│   └── annotation_export/        # Case data for labeling
├── scripts/                      # Utility scripts (Training, Indexing)
├── tests/                        # Unit & integration tests
├── models/                       # Fine-tuned model weights
└── docs/                         # Detailed documentation
```

---

## 🛠 Setup & Quick Start

### Prerequisites
- Python 3.9+
- Hugging Face API Token (`HF_TOKEN`)

### Installation
1.  **Clone the repository.**
2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment:**
    Create a `.env` file and add your Hugging Face Token:
    ```text
    HF_TOKEN=your_huggingface_token_here
    ```

### Running the App
```bash
streamlit run app/app.py
```

---

## ☁️ Hosting & Deployment

### 1. Hugging Face Spaces (Recommended)
- Direct integration with models.
- Uses `app.py` as the entry point.

### 2. AWS EC2 (Virtual Machine)
- **Instance Type:** `t3.medium` or `t3.large` (2-4GB RAM minimum).
- **Process Management:** Use `tmux` or `pm2` to keep the Streamlit process alive.
- **Security:** Open ports 80 (HTTP) and 8501 (Streamlit).

### 3. AWS App Runner (Containerized)
- Fully managed scaling.
- Use the provided `Dockerfile` (Python 3.10-slim base).

---

## 📈 Future Roadmap
- [ ] **Native DICOM Support**: Full metadata and pixel data ingestion.
- [ ] **Multi-Plane Co-Registration**: Synchronized Sagittal/Axial view coordination.
- [ ] **Human-in-the-Loop Fine-Tuning**: Capturing radiologist edits to update CLIP weights.
- [ ] **Regulatory Audit Trail**: Hard-logging of all reasoning events for compliance.

---

## ⚠️ Medical Disclaimer
SpineAI Pro is a **Clinical Decision Support (CDS)** tool. It is designed to assist radiologists, not replace them. All findings must be reviewed and verified by a board-certified radiologist. The quantitative metrics are estimates and should be used for screening purposes only.
