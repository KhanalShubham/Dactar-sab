# 🚀 Quick Start Guide

Get the **SpineAI + CLIP** system running on your local machine in under 30 minutes.

## 🛠️ Step 1: Environment Setup

We recommend using a virtual environment (Python 3.9+).

```bash
# Create environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

## 📦 Step 2: Install Dependencies

Install the core medical AI stack:

```bash
pip install -r requirements.txt
```

*Note: This includes `torch`, `transformers`, `streamlit`, `monai`, and `fpdf2`.*

## 🏃 Step 3: Run the Clinical App

Launch the Streamlit interface to test the MRI analysis workflow.

```bash
streamlit run spineai_clip_app.py
```

## 🧪 Step 4: Test Your First MRI

1. **Safety Gate**: Pass the Clinical Intake form (CES screening).
2. **Upload**: Select an MRI scan (e.g., `test.png`).
3. **Analyze**: Watch CLIP auto-tag findings and the LLM generate a draft report.
4. **Export**: Review, sign, and download your finalized PDF report.

---

## 🔧 Troubleshooting

- **Memory Issues?** If you are running on a machine with < 8GB RAM, the CLIP model might load slowly. Ensure no other heavy applications are running.
- **API Key Error?** Ensure you have entered your Hugging Face Token in the sidebar for the text-generation features.
- **CUDA Errors?** The system automatically falls back to CPU if no NVIDIA GPU is detected. You don't need a GPU for the demo!

---

## 📖 What's Next?
- Read **[REFERENCE_CARD.md](file:///c:/Users/shubh/Downloads/SpineAI_Prototype/REFERENCE_CARD.md)** for API usage examples.
- Explore **[clip_integration.py](file:///c:/Users/shubh/Downloads/SpineAI_Prototype/clip_integration.py)** to see the AI logic.
