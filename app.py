import streamlit as st
import datetime
import re
import os
import json
import numpy as np
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from src.analyzer.clip_analyzer import MedicalCLIPAnalyzer
from src.export.pdf_generator import ReportExporter
from app.config import ConfigManager, REPORT_SECTIONS
from huggingface_hub import InferenceClient
from src.training.feedback_manager import FeedbackManager
from src.analyzer.lab_interpreter import LabInterpreter
from src.triage.opd_engine import OPDTriageEngine
from src.analyzer.diff_diag_visualizer import DiffDiagVisualizer
from src.export.referral_generator import ReferralGenerator
from src.surveillance.outbreak_tracker import OutbreakTracker
from src.ui.voice_assistant import VoiceAssistant
from src.training.academy_engine import AcademyEngine
from src.inventory.stock_manager import StockManager
from src.communication.notifier import ClinicalNotifier
from src.surveillance.epidemic_predictor import EpidemicPredictor
from src.data.patient_history import PatientHistory
from src.utils.dicom_loader import process_medical_image, process_dicom_volume, load_dicom_3d_volume
from src.utils.volume_renderer import create_3d_isosurface, create_mpr_slice
from src.utils.spine_localizer import build_level_finding_map, level_to_slice_index

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpineAI Pro | Clinical Reporting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

config = ConfigManager.get_default_config()

# Report sections the doctor must review individually
REPORT_SECTIONS = [
    "TECHNIQUE",
    "VERTEBRAL ALIGNMENT",
    "BONE MARROW SIGNAL",
    "CONUS",
    "DISC ASSESSMENT",
    "SPINAL CANAL & THECAL SAC",
    "FORAMINAL ASSESSMENT",
    "FACET JOINTS",
    "IMPRESSION",
]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "state": "INTAKE",
        "patient_data": {"name": "", "age": "", "id": "", "referrer": ""},
        "analysis_results": None,
        "section_texts": {},       # {section_name: edited_text}
        "section_confirmed": {},   # {section_name: bool}
        "raw_report": "",
        "logs": [],
        "signed_by": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")

def parse_report_sections(report_text, visual_tags=None):
    """
    Ultra-robust parser for radiology reports. 
    Uses keyword anchors and a 'Smart Fallback' if sections are missing.
    """
    sections = {}
    report_text_upper = report_text.upper()
    
    # 1. Find the start index of each section
    indices = []
    for s in REPORT_SECTIONS:
        idx = report_text_upper.find(s)
        if idx != -1:
            indices.append((idx, s))
    
    indices.sort() # Sort by appearance in text
    
    # 2. Extract content between section headers
    for i in range(len(indices)):
        start_idx, current_s = indices[i]
        # Content starts after the section name (plus optional colon/space)
        content_start = start_idx + len(current_s)
        
        if i + 1 < len(indices):
            end_idx = indices[i+1][0]
            content = report_text[content_start:end_idx]
        else:
            content = report_text[content_start:]
        
        # Clean up the content (remove leading colons, stars, etc)
        content = re.sub(r"^[ \t\*\-:#]*", "", content).strip()
        sections[current_s] = content

    # 3. SMART FALLBACK: If LLM missed a section, use Visual Tags to populate it
    if visual_tags:
        # Mapping tags to sections
        mapping = {
            "VERTEBRAL ALIGNMENT": ["alignment", "listhesis", "lordosis"],
            "BONE MARROW SIGNAL": ["marrow", "modic", "fracture", "hemangioma"],
            "CONUS": ["conus", "medullaris", "terminal"],
            "DISC ASSESSMENT": ["disc", "desiccation", "protrusion", "extrusion", "height"],
            "SPINAL CANAL & THECAL SAC": ["canal", "stenosis", "thecal", "crowding", "compression"],
            "FORAMINAL ASSESSMENT": ["foramina", "narrowing", "exit"],
            "FACET JOINTS": ["facet", "arthropathy", "hypertrophy", "ligamentum"],
            "IMPRESSION": ["severe", "stenosis", "compression", "urgent"]
        }
        
        for s in REPORT_SECTIONS:
            if s not in sections or len(sections[s]) < 5:
                # Find matching tags
                keywords = mapping.get(s, [])
                matching_tags = [t['label'] for t in visual_tags if any(k in t['label'].lower() for k in keywords)]
                
                if matching_tags:
                    sections[s] = f"AI visual analysis suggests: {', '.join(matching_tags)}. Clinical correlation and radiologist verification required."
                else:
                    sections[s] = f"No significant degenerative markers detected for {s.lower()} on visual analysis."

    return sections

def all_sections_confirmed():
    return all(
        st.session_state.section_confirmed.get(s, False) for s in REPORT_SECTIONS
    )

def build_final_report():
    lines = []
    p = st.session_state.patient_data
    lines.append(f"PATIENT: {p['name']}  |  AGE: {p['age']}  |  ID: {p['id']}")
    lines.append(f"REFERRING PHYSICIAN: {p.get('referrer', 'N/A')}")
    lines.append(f"STUDY DATE: {datetime.date.today().strftime('%d %B %Y')}")
    lines.append("=" * 70)
    for s in REPORT_SECTIONS:
        lines.append(f"\n{s}:")
        lines.append(st.session_state.section_texts.get(s, "").strip())
    lines.append("\n" + "=" * 70)
    lines.append(f"Digitally signed by: {st.session_state.signed_by}")
    lines.append(f"Signed at: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}")
    lines.append("STATUS: FINALIZED")
    return "\n".join(lines)
# ─────────────────────────────────────────────────────────────────────────────
# AI ENGINE CACHING & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_analyzer(version="3D_v1.0"):
    """Loads and caches the BiomedCLIP analyzer in GPU memory."""
    return MedicalCLIPAnalyzer()

def generate_narrative(hf_token, sir_json, tags, patient_info, comparison_mode=False):
    if not hf_token:
        return "ERROR: No Hugging Face API token provided."
    try:
        client = InferenceClient(api_key=hf_token)
        sir_str = json.dumps(sir_json, indent=2)
        
        system_msg = """You are a Board-Certified Neuroradiologist.
STRICT REPORTING RULES:
1. SILENT AI: NEVER mention 'AI', 'visual markers', 'inference', or 'analysis suggests'. Speak as a human consultant.
2. TELEGRAPHIC STYLE: Use concise, professional medical shorthand. Remove fluff.
3. CAUSAL CONSISTENCY: Every finding must have a pathophysiological cause. Use terms like 'Degenerative spondylosis' or 'Hypertrophic changes'.
4. COMPARISON GATE: ONLY mention prior studies if comparison_mode is TRUE. If FALSE, comparison language is FORBIDDEN.
5. NEURO-ANATOMY: No 'spinal cord' below L2. Use 'thecal sac' or 'cauda equina'.
6. CLINICAL CORRELATION: Keep to a single sentence: 'Clinical correlation advised for radicular symptoms.'
"""
        user_msg = f"""Patient: {patient_info}
STRUCTURED FINDINGS (SIR):
{sir_str}
COMPARISON_MODE_ENABLED: {comparison_mode}

TASK: Generate a professional, unified radiology report. 
Sections: TECHNIQUE, ALIGNMENT, BONE MARROW, CONUS, DISC ASSESSMENT, SPINAL CANAL, FORAMINA, FACET JOINTS, IMPRESSION.
"""
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=2500,
            temperature=0.1
        )
        raw_content = response.choices[0].message.content
        log(f"AI Success: {len(raw_content)} chars generated.")
        return raw_content
    except Exception as e:
        log(f"LLM Error: {e}")
        return f"Report generation failed: {e}"

def ensemble_verify_findings(hf_token, tags):
    """Secondary reasoning stage to ensemble-verify findings for clinical consistency."""
    if not hf_token: return tags
    try:
        client = InferenceClient(api_key=hf_token)
        tags_str = ", ".join([t['label'] for t in tags])
        prompt = f"Review these radiology findings: {tags_str}. Identify any anatomical or logical contradictions. Respond with 'VALID' or describe the error."
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=100
        )
        verdict = response.choices[0].message.content
        if "VALID" not in verdict.upper():
            log(f"Ensemble Verification Alert: {verdict}")
        return tags
    except:
        return tags

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION & DATASET UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def export_for_annotation(image_bytes, tags, report_text, patient_id):
    """Saves image and metadata in a format ready for Label Studio/Labelbox."""
    base_dir = "annotation_export"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    case_id = f"{patient_id}_{timestamp}"
    
    # Save Image
    img_path = os.path.join(base_dir, f"{case_id}.png")
    with open(img_path, "wb") as f:
        f.write(image_bytes)
        
    # Save Metadata (JSONL format compatible with clip_training.py)
    meta = {
        "case_id": case_id,
        "image": f"{case_id}.png",
        "tags": tags,
        "report": report_text,
        "status": "ready_for_review"
    }
    meta_path = os.path.join(base_dir, f"{case_id}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return meta_path

def calculate_severity_index(tags):
    """Calculates a 1-10 severity score based on finding keywords and scores."""
    score = 1.0
    critical_keywords = {
        "stenosis": 2.0,
        "compression": 3.0,
        "fracture": 4.5,
        "protrusion": 1.5,
        "extrusion": 2.5,
        "narrowing": 1.0,
        "spondylolisthesis": 2.0
    }
    
    for tag in tags:
        label = tag['label'].lower()
        conf = tag['score']
        for kw, weight in critical_keywords.items():
            if kw in label:
                score += weight * conf
                
    return min(10.0, round(score, 1))

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ═══════════════════════════════════════════════════════════════
   DESIGN TOKENS  — automatically flip for dark / light mode
   ═══════════════════════════════════════════════════════════════ */
:root {
  --c-primary:     #0EA5E9;
  --c-primary-dk:  #0284C7;
  --c-primary-lt:  #E0F2FE;
  --c-surface:     #FFFFFF;
  --c-surface-2:   #F8FAFC;
  --c-surface-3:   #F1F5F9;
  --c-border:      #E2E8F0;
  --c-text-1:      #0F172A;
  --c-text-2:      #475569;
  --c-text-3:      #94A3B8;
  --c-success:     #10B981;
  --c-success-bg:  #ECFDF5;
  --c-success-tx:  #065F46;
  --c-warning:     #F59E0B;
  --c-warning-bg:  #FFFBEB;
  --c-danger:      #EF4444;
  --c-danger-bg:   #FEF2F2;
  --shadow-sm:     0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md:     0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -1px rgba(0,0,0,.04);
  --r-sm: 8px;  --r-md: 12px;  --r-lg: 16px;  --r-xl: 20px;
}

/* Dark mode overrides via Streamlit's data-theme attribute */
[data-theme="dark"] {
  --c-primary-lt:  #0C2A40;
  --c-surface:     #1E2A3A;
  --c-surface-2:   #141E2E;
  --c-surface-3:   #0D1623;
  --c-border:      #2D3E53;
  --c-text-1:      #E8F0FE;
  --c-text-2:      #8BA5C0;
  --c-text-3:      #4D6A88;
  --c-success-bg:  #052E16;
  --c-success-tx:  #6EE7B7;
  --c-warning-bg:  #1C1108;
  --c-danger-bg:   #1F0808;
  --shadow-sm:     0 1px 4px rgba(0,0,0,.4);
  --shadow-md:     0 4px 16px rgba(0,0,0,.55);
}

/* ═══════════════════════════════════════════════════════════════
   BASE
   ═══════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
  background-color: var(--c-surface-2) !important;
}

.block-container {
  padding-top: 1.5rem !important;
  max-width: 1440px !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR TOGGLE BUTTONS
   ═══════════════════════════════════════════════════════════════ */

/* ">" button shown when sidebar is COLLAPSED */
[data-testid="collapsedControl"] {
  position: fixed !important;
  top: 12px !important;
  left: 0 !important;
  z-index: 99999 !important;
  display: flex !important;
  visibility: visible !important;
  align-items: center !important;
}
[data-testid="collapsedControl"] button {
  background: var(--c-brand, #0EA5E9) !important;
  border: none !important;
  border-radius: 0 10px 10px 0 !important;
  width: 2.6rem !important;
  height: 2.6rem !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  box-shadow: 2px 2px 10px rgba(0,0,0,0.30) !important;
}
[data-testid="collapsedControl"] button svg {
  fill: #ffffff !important;
  width: 1.2rem !important;
  height: 1.2rem !important;
}

/* "×" button shown inside sidebar when it is OPEN */
[data-testid="stSidebarCollapseButton"] {
  display: flex !important;
  visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"] button {
  background: rgba(14,165,233,0.12) !important;
  border: 1.5px solid var(--c-brand, #0EA5E9) !important;
  border-radius: 8px !important;
  width: 2.2rem !important;
  height: 2.2rem !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
}
[data-testid="stSidebarCollapseButton"] button svg {
  fill: var(--c-brand, #0EA5E9) !important;
  width: 1.1rem !important;
  height: 1.1rem !important;
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background-color: var(--c-surface) !important;
  border-right: 1px solid var(--c-border) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown strong {
  color: var(--c-text-1) !important;
}

[data-testid="stSidebar"] [data-testid="stCaption"] p {
  color: var(--c-text-2) !important;
  font-size: 0.8rem !important;
}

/* ═══════════════════════════════════════════════════════════════
   INPUTS
   ═══════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox [data-baseweb="select"] > div {
  background-color: var(--c-surface) !important;
  color: var(--c-text-1) !important;
  border: 1.5px solid var(--c-border) !important;
  border-radius: var(--r-md) !important;
  font-size: 0.9375rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--c-primary) !important;
  box-shadow: 0 0 0 3px rgba(14,165,233,.15) !important;
  outline: none !important;
}

.stTextInput label, .stTextArea label, .stSelectbox label {
  color: var(--c-text-2) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: .03em !important;
}

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, var(--c-primary), var(--c-primary-dk)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  padding: 0.6rem 1.25rem !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  letter-spacing: .01em !important;
  box-shadow: 0 2px 8px rgba(14,165,233,.3) !important;
  transition: transform .15s, box-shadow .15s !important;
}

.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(14,165,233,.45) !important;
}

.stButton > button:active {
  transform: translateY(0) !important;
}

/* ═══════════════════════════════════════════════════════════════
   METRICS
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stMetricValue"] {
  font-weight: 800 !important;
  color: var(--c-primary) !important;
  letter-spacing: -0.04em !important;
}

[data-testid="stMetricLabel"] {
  color: var(--c-text-2) !important;
  font-weight: 500 !important;
  font-size: 0.8125rem !important;
}

/* ═══════════════════════════════════════════════════════════════
   ALERTS / CALLOUTS
   ═══════════════════════════════════════════════════════════════ */
.stAlert {
  border-radius: var(--r-md) !important;
  border-left-width: 4px !important;
  border-top: none !important;
  border-right: none !important;
  border-bottom: none !important;
}

/* ═══════════════════════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════════════════════ */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--c-primary), var(--c-primary-dk)) !important;
  border-radius: 99px !important;
}

/* ═══════════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--c-surface-3) !important;
  border-radius: var(--r-md) !important;
  padding: 4px !important;
  gap: 3px !important;
  border: 1px solid var(--c-border) !important;
}

.stTabs [data-baseweb="tab"] {
  border-radius: var(--r-sm) !important;
  color: var(--c-text-2) !important;
  font-weight: 500 !important;
  padding: 0.4rem 1rem !important;
  background: transparent !important;
}

.stTabs [aria-selected="true"] {
  background: var(--c-surface) !important;
  color: var(--c-primary) !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ═══════════════════════════════════════════════════════════════
   EXPANDERS
   ═══════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border) !important;
  border-radius: var(--r-md) !important;
  overflow: hidden !important;
}

[data-testid="stExpander"] summary {
  font-weight: 600 !important;
  color: var(--c-text-1) !important;
}

/* ═══════════════════════════════════════════════════════════════
   CHECKBOXES
   ═══════════════════════════════════════════════════════════════ */
.stCheckbox label {
  color: var(--c-text-1) !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
}

/* ═══════════════════════════════════════════════════════════════
   SELECTBOX DROPDOWN
   ═══════════════════════════════════════════════════════════════ */
[data-baseweb="popover"] {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border) !important;
  border-radius: var(--r-md) !important;
}

[data-baseweb="menu"] li {
  color: var(--c-text-1) !important;
}

[data-baseweb="menu"] li:hover {
  background: var(--c-primary-lt) !important;
}

/* ═══════════════════════════════════════════════════════════════
   DIVIDERS
   ═══════════════════════════════════════════════════════════════ */
hr {
  border-color: var(--c-border) !important;
  opacity: 1 !important;
}

/* ═══════════════════════════════════════════════════════════════
   ── CUSTOM COMPONENTS ──
   ═══════════════════════════════════════════════════════════════ */

/* App Branding Header */
.spine-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-radius: var(--r-xl);
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
}

.spine-logo {
  width: 46px; height: 46px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--c-primary), var(--c-primary-dk));
  border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.35rem; font-weight: 800; color: #fff;
  box-shadow: 0 4px 12px rgba(14,165,233,.25);
}

.spine-wordmark { display: flex; flex-direction: column; }

.spine-title {
  font-size: 1.45rem;
  font-weight: 800;
  color: var(--c-text-1) !important;
  margin: 0;
  letter-spacing: -0.03em;
  line-height: 1.1;
}

.spine-sub {
  font-size: 0.8125rem;
  color: var(--c-text-2) !important;
  margin: 0.15rem 0 0;
  font-weight: 500;
}

/* Phase / Section Header */
.report-header {
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-left: 5px solid var(--c-primary);
  border-radius: var(--r-lg);
  padding: 1.125rem 1.5rem;
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow-sm);
}

.report-header h2 {
  margin: 0 0 0.25rem !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
  color: var(--c-text-1) !important;
}

.report-header p {
  margin: 0 !important;
  font-size: 0.875rem !important;
  color: var(--c-text-2) !important;
  line-height: 1.5;
}

/* Patient Identity Strip */
.patient-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem 2.5rem;
  background: var(--c-surface-3);
  border: 1px solid var(--c-border);
  border-radius: var(--r-md);
  padding: 0.875rem 1.25rem;
  margin-bottom: 1.25rem;
}

.ps-item { display: flex; flex-direction: column; gap: 1px; }

.ps-label {
  font-size: 0.67rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: .07em !important;
  color: var(--c-text-3) !important;
}

.ps-value {
  font-size: 0.9375rem !important;
  font-weight: 600 !important;
  color: var(--c-text-1) !important;
}

/* Section Review Card */
.section-card {
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-left: 4px solid var(--c-border);
  border-radius: var(--r-md);
  padding: 0.75rem 1rem;
  margin-bottom: 0.375rem;
  transition: border-left-color .2s;
}

.section-card.confirmed {
  border-left-color: var(--c-success) !important;
  background: var(--c-success-bg) !important;
}

.section-card.pending {
  border-left-color: var(--c-warning) !important;
}

.sec-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8125rem !important;
  font-weight: 700 !important;
  color: var(--c-text-1) !important;
  margin: 0;
  text-transform: uppercase;
  letter-spacing: .04em;
}

/* Status Badges */
.badge-confirmed {
  background: var(--c-success-bg) !important;
  color: var(--c-success-tx) !important;
  padding: 3px 10px !important;
  border-radius: 99px !important;
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  border: 1px solid rgba(16,185,129,.25);
  white-space: nowrap;
}

.badge-pending {
  background: var(--c-warning-bg) !important;
  color: #92400E !important;
  padding: 3px 10px !important;
  border-radius: 99px !important;
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  white-space: nowrap;
}

/* Digital Sign-off Box */
.sign-box {
  background: var(--c-surface);
  border: 2px dashed var(--c-primary);
  border-radius: var(--r-xl);
  padding: 1.5rem;
  margin-top: 1.5rem;
}

/* Triage Priority Display */
.triage-IMMEDIATE,
.triage-URGENT,
.triage-ROUTINE {
  padding: 1.25rem 1.5rem;
  border-radius: var(--r-lg);
  text-align: center;
  margin-bottom: 0.75rem;
}

.triage-IMMEDIATE { background: #EF4444; }
.triage-URGENT    { background: #F59E0B; }
.triage-ROUTINE   { background: var(--c-primary); }

.triage-IMMEDIATE h1,
.triage-URGENT h1,
.triage-ROUTINE h1 {
  margin: 0 !important;
  color: #ffffff !important;
  font-size: 1.75rem !important;
  font-weight: 800 !important;
  letter-spacing: -.02em;
}

.triage-IMMEDIATE p,
.triage-URGENT p,
.triage-ROUTINE p {
  margin: 0.2rem 0 0 !important;
  color: rgba(255,255,255,.8) !important;
  font-size: 0.75rem !important;
  text-transform: uppercase;
  letter-spacing: .1em;
  font-weight: 600;
}

/* Differential Diagnosis Rows */
.diff-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.55rem 0.875rem;
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-radius: var(--r-sm);
  margin-bottom: 0.4rem;
}

.diff-bar {
  width: 4px;
  align-self: stretch;
  border-radius: 2px;
  flex-shrink: 0;
}

.diff-name { font-weight: 600; color: var(--c-text-1) !important; flex: 1; font-size: 0.9rem; }
.diff-pct  { font-weight: 700; color: var(--c-text-2) !important; font-size: 0.875rem; }

/* Mobile responsive */
@media (max-width: 768px) {
  .spine-header  { flex-direction: column; align-items: flex-start; }
  .patient-strip { gap: 0.5rem 1.5rem; }
  .report-header h2 { font-size: 1.05rem !important; }
  .sec-title { flex-direction: column; align-items: flex-start; gap: 0.3rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar toggle button (JS-injected, CSS selectors unreliable across versions) ──
import streamlit.components.v1 as _components
_components.html("""
<script>
(function() {
  var par = window.parent;

  function clickNativeToggle() {
    // Try every known Streamlit sidebar button selector
    var selectors = [
      '[data-testid="stSidebarCollapseButton"] button',
      '[data-testid="collapsedControl"] button',
      '[data-testid="stSidebar"] button[kind="header"]',
      'section[data-testid="stSidebar"] > div > div > button',
      '[data-testid="stSidebarNav"] + div button',
    ];
    for (var i = 0; i < selectors.length; i++) {
      var btn = par.document.querySelector(selectors[i]);
      if (btn) { btn.click(); return true; }
    }
    return false;
  }

  function isSidebarVisible() {
    var sb = par.document.querySelector('[data-testid="stSidebar"]');
    if (!sb) return false;
    var rect = sb.getBoundingClientRect();
    return rect.left > -10;
  }

  function updateIcon(btn) {
    btn.textContent = isSidebarVisible() ? '✕' : '☰';
    btn.title = isSidebarVisible() ? 'Close sidebar' : 'Open sidebar';
  }

  function createToggle() {
    if (par.document.getElementById('spineai-sb-toggle')) return;
    var btn = par.document.createElement('button');
    btn.id = 'spineai-sb-toggle';
    btn.style.cssText = [
      'position:fixed', 'top:14px', 'left:14px', 'z-index:2147483647',
      'width:42px', 'height:42px', 'border-radius:10px', 'border:none',
      'background:#0EA5E9', 'color:#fff', 'font-size:19px', 'cursor:pointer',
      'box-shadow:0 2px 10px rgba(0,0,0,0.28)', 'display:flex',
      'align-items:center', 'justify-content:center', 'transition:background .15s'
    ].join(';');
    btn.onmouseenter = function() { btn.style.background = '#0284C7'; };
    btn.onmouseleave = function() { btn.style.background = '#0EA5E9'; };
    btn.onclick = function() {
      clickNativeToggle();
      setTimeout(function() { updateIcon(btn); }, 350);
    };
    updateIcon(btn);
    par.document.body.appendChild(btn);
    // Keep icon in sync on Streamlit re-renders
    setInterval(function() { updateIcon(btn); }, 800);
  }

  if (par.document.readyState === 'loading') {
    par.document.addEventListener('DOMContentLoaded', createToggle);
  } else {
    setTimeout(createToggle, 300);
  }
})();
</script>
""", height=0)

def branding_header():
    st.markdown("""
        <div class="spine-header">
            <div class="spine-logo">S</div>
            <div class="spine-wordmark">
                <span class="spine-title">SpineAI Pro</span>
                <span class="spine-sub">Advanced Neuroradiology Diagnostic Infrastructure &middot; Nepal</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

branding_header()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 SpineAI Pro")
    st.caption(f"v{config['version']} · Clinical Decision Support")
    st.markdown("---")

    st.markdown("**🔑 API Authentication**")
    hf_token = st.text_input("Hugging Face Token", type="password",
                              value=os.getenv("HF_TOKEN", ""),
                              label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**👤 Patient Demographics**")
    p = st.session_state.patient_data
    p["name"]     = st.text_input("Full Name",           value=p["name"])
    p["id"]       = st.text_input("Medical Record No.",  value=p["id"])
    p["age"]      = st.text_input("Age / DOB",           value=p["age"])
    p["referrer"] = st.text_input("Referring Physician", value=p.get("referrer",""))

    st.markdown("---")
    # Workflow progress
    stage_map = {"INTAKE": 0, "UPLOAD": 1, "DRAFT": 2, "REVIEW": 3, "FINAL": 4, "CES_BLOCKED": 0}
    stage_val = stage_map.get(st.session_state.state, 0)
    st.markdown("**Workflow Progress**")
    stages = ["Intake", "Upload", "AI Analysis", "Doctor Review", "Final"]
    for i, s in enumerate(stages):
        icon = "✅" if i < stage_val else ("🔵" if i == stage_val else "⬜")
        st.caption(f"{icon} {s}")

    st.markdown("---")
    st.markdown("**📋 Confirmed Findings**")
    if "section_confirmed" in st.session_state:
        confirmed = [s for s, v in st.session_state.section_confirmed.items() if v]
        if confirmed:
            for s in confirmed:
                st.caption(f"✅ {s}")
        else:
            st.caption("No sections confirmed yet.")

    st.markdown("---")
    st.markdown("**📋 Session Log**")
    for entry in st.session_state.logs[-6:]:
        st.caption(entry)

    st.markdown("---")
    st.markdown("**📊 Dataset Dashboard**")
    if os.path.exists("annotation_export"):
        count = len([f for f in os.listdir("annotation_export") if f.endswith(".png")])
        st.metric("Images Collected", count)
    if os.path.exists("priority_review"):
        hard_count = len([f for f in os.listdir("priority_review") if f.endswith(".png")])
        st.metric("Hard Cases (Priority)", hard_count)

    st.markdown("---")
    st.markdown("**🎓 Education & Logistics**")
    if st.button("Open NepalMed Academy", help="Training scenarios for health workers"):
        st.session_state.state = "ACADEMY_MODE"
        st.rerun()
    
    if st.button("Check Pharmacy Stock", help="View essential medicine availability"):
        st.session_state.state = "INVENTORY_MODE"
        st.rerun()

    if st.button("📈 Surveillance Dashboard", help="Real-time outbreak monitoring"):
        st.session_state.state = "SURVEILLANCE_MODE"
        st.rerun()

    st.markdown("---")
    st.markdown("**🏥 Community Health Portal (FCHV)**")
    if st.button("Open FCHV Interface", help="Simplified interface for community health volunteers"):
        st.session_state.state = "FCHV_MODE"
        st.rerun()

    st.markdown("---")
    with st.expander("🛠️ Developer Console"):
        st.caption("Admin & System Management")
        if st.button("🔄 Re-index Vector DB", help="Scans historical_cases/ and rebuilds FAISS index"):
            import subprocess
            try:
                subprocess.run([".venv/Scripts/python.exe", "scripts/index_cases.py", "--dir", "historical_cases"], check=True)
                st.success("FAISS Index Rebuilt!")
            except Exception as e:
                st.error(f"Re-indexing failed: {e}")
        
        if st.button("🧹 Clear Logs"):
            st.session_state.logs = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CLINICAL INTAKE & CES SCREENING
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.state == "INTAKE":
    st.markdown("""
    <div class="report-header">
        <h2>Phase 1 — Clinical Intake & Safety Screening</h2>
        <p>Complete the mandatory Cauda Equina Syndrome screening before proceeding.</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_info = st.columns([2, 1])
    with col_form:
        with st.form("intake_form"):
            st.markdown("#### Cauda Equina Syndrome (CES) Red-Flag Checklist")
            st.caption("Check all symptoms that apply to the patient:")
            saddle   = st.checkbox("Saddle anesthesia (perineal/groin numbness)")
            bladder  = st.checkbox("New urinary retention or incontinence")
            bowel    = st.checkbox("New bowel incontinence")
            weakness = st.checkbox("Rapidly progressive bilateral leg weakness")
            st.markdown("---")
            study_type = st.selectbox("Study Type", ["Lumbar Spine MRI — T2 Sagittal & Axial", "Lumbar Spine MRI — T1 & T2", "Other"])
            clinical_q = st.text_area("Clinical Question / Indication", placeholder="e.g. Low back pain radiating to left leg, rule out disc prolapse.")
            submitted = st.form_submit_button("Proceed with Screening →", type="primary")

        if submitted:
            if saddle or bladder or bowel or weakness:
                st.session_state.state = "CES_BLOCKED"
                log("CES red flags detected — AI blocked.")
                st.rerun()
            else:
                st.session_state.patient_data["study_type"]  = study_type
                st.session_state.patient_data["clinical_q"]  = clinical_q
                log("CES screening passed.")
                st.session_state.state = "UPLOAD"
                st.rerun()

    with col_info:
        st.info("**Why this screen?**\n\nCauda Equina Syndrome is a surgical emergency. SpineAI is designed for elective, non-emergency reporting only. Any red-flag symptoms disable AI assistance.")

# ─────────────────────────────────────────────────────────────────────────────
# CES BLOCKED
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "CES_BLOCKED":
    st.error("🚨 **CRITICAL: POTENTIAL CAUDA EQUINA SYNDROME DETECTED**")
    st.warning("**AI reporting is DISABLED.** Refer immediately to Neurosurgery / Emergency Department.")
    st.markdown("**Do not delay.** CES requires surgical decompression within hours of onset.")
    if st.button("← New Patient"):
        for k in ["state","section_texts","section_confirmed","raw_report","analysis_results"]:
            st.session_state.pop(k, None)
        init_state()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — MRI UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "UPLOAD":
    st.markdown("""
    <div class="report-header">
        <h2>Phase 2 — MRI Image Upload</h2>
        <p>Upload a standard image (PNG/JPG) or a clinical **DICOM (.dcm)** file for real-time analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Select MRI files (Stack/Volume)", type=["png","jpg","jpeg","dcm", "dicom", "ima"], accept_multiple_files=True)

    if uploaded_files:
        # 1. DATA PROCESSING (Unified for both tabs)
        files_data = [(f.getvalue(), f.name) for f in uploaded_files]
        if len(uploaded_files) > 1:
            processed_slices, vol_profile, dcm_meta = process_dicom_volume(files_data)
            st.session_state.processed_volume = processed_slices
            active_image = processed_slices[len(processed_slices)//2] 
            intensity_profile = vol_profile
            study_type_label = f"Volumetric Study ({len(uploaded_files)} slices)"
        else:
            f = uploaded_files[0]
            processed_bytes, intensity_profile, dcm_meta = process_medical_image(f.getvalue(), f.name)
            active_image = processed_bytes
            st.session_state.processed_volume = [processed_bytes]
            study_type_label = "Single-Slice Study"

        # Auto-fill Metadata
        if dcm_meta:
            st.session_state.patient_data["name"] = dcm_meta["name"]
            st.session_state.patient_data["id"] = dcm_meta["id"]
            st.session_state.patient_data["age"] = dcm_meta["age"]
            st.toast("✅ Medical metadata extracted.")

        # 2. UI LAYOUT (Tabs)
        tab_preview, tab_3d = st.tabs(["🖼️ Slice Preview", "🧊 3D Interactive Viewer"])
        
        with tab_preview:
            col_prev, col_action = st.columns([1, 1])
            with col_prev:
                st.image(active_image, caption=f"{study_type_label} - Key Slice Preview", use_container_width=True)
                if intensity_profile is not None and len(intensity_profile) > 0:
                    try:
                        # Only show chart if there is data variation to avoid Vega-Lite warnings
                        if np.max(intensity_profile) > np.min(intensity_profile):
                            st.line_chart(intensity_profile, height=150, use_container_width=True)
                            st.caption("📏 3D Mean T2 Signal Intensity Profile")
                    except:
                        pass
            
            with col_action:
                st.markdown("**Volumetric Study Insights**")
                p = st.session_state.patient_data
                st.write(f"- **Patient:** {p['name']}")
                st.write(f"- **ID:** {p['id']}")
                st.write(f"- **Volume Size:** {len(uploaded_files)} slices")
                
                st.markdown("---")
                st.info("Clicking **Run 3D AI Analysis** will perform a consensus-based interpretation across the entire volume.")

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    comp_mode = st.checkbox("⏳ Compare to Prior MRI", help="Enables temporal reasoning for interval changes")
                
                if st.button("🧠 Run 3D AI Consensus Analysis", type="primary"):
                    with st.spinner(f"Analyzing {len(uploaded_files)} slices for volumetric consensus…"):
                        from src.constraints.anatomical_engine import AnatomicalConstraintEngine
                        from src.constraints.consistency_engine import ConsistencyEngine
                        from src.constraints.causal_validator import CausalReasoningEngine
                        
                        log("STAGE 1: Volumetric feature extraction...")
                        analyzer = get_analyzer()
                        if len(uploaded_files) > 1:
                            tags = analyzer.analyze_volume_consensus(processed_slices)
                        else:
                            tags = analyzer.auto_tag_findings(active_image)
                        
                        log("STAGE 2: Applying Anatomical Constraints (ACE)...")
                        tags = AnatomicalConstraintEngine.apply_constraints(tags)
                        
                        log("STAGE 3: Consistency & Causal Reasoning...")
                        from src.constraints.causal_validator import SymbolicCausalValidator
                        
                        conflicts = ConsistencyEngine.detect_contradictions(tags)
                        causality = CausalReasoningEngine.derive_causality(tags)
                        
                        # Generate Structured Intermediate Representation (SIR)
                        sir_json = analyzer.get_sir_map(tags)
                        sir_json["CAUSALITY_MAP"] = causality
                        
                        # Phase 10: Symbolic Causal Validation
                        sir_json = SymbolicCausalValidator.validate_and_refine(sir_json)
                        for level, data in sir_json.items():
                            if isinstance(data, dict) and data.get("_flag"):
                                log(f"🧠 CAUSAL SUPPRESSION: {level} - {data['_flag']}")
                        
                        # Ensemble cross-check
                        tags = ensemble_verify_findings(hf_token, tags)

                        log("STAGE 4: Professional Narrative Generation...")
                        similar = analyzer.find_similar_cases(active_image)
                        p_info = f"{p['name']}, Age {p['age']}, ID {p['id']}"
                        
                        raw = generate_narrative(hf_token, sir_json, tags, p_info, comparison_mode=comp_mode)
                        
                        sections = parse_report_sections(raw, visual_tags=tags)
                        st.session_state.section_texts     = {s: sections.get(s, "") for s in REPORT_SECTIONS}
                        st.session_state.section_confirmed = {s: False for s in REPORT_SECTIONS}
                        st.session_state.raw_report        = raw
                        st.session_state.analysis_results  = {
                            "tags": tags, "image": active_image, "similar": similar, "profile": intensity_profile,
                            "is_3d": len(uploaded_files) > 1, "conflicts": conflicts, "sir_json": sir_json
                        }
                        
                        # Phase 7.5: Quantitative Analysis
                        from src.utils.metrics import QuantitativeAnalyzer
                        st.session_state.analysis_results["metrics"] = QuantitativeAnalyzer.estimate_metrics(tags)

                        # Pre-build 3D volume so it's ready in the REVIEW phase viewer
                        if len(uploaded_files) > 1:
                            if ("volume_3d" not in st.session_state or
                                    st.session_state.get("last_uploaded_files") != uploaded_files):
                                v3d, s3d = load_dicom_3d_volume(files_data)
                                st.session_state.volume_3d = v3d
                                st.session_state.spacing_3d = s3d
                                st.session_state.last_uploaded_files = uploaded_files
                        st.session_state.pop("review_target_slice", None)

                        st.session_state.state = "REVIEW"
                        st.rerun()
        
        with tab_3d:
            if len(uploaded_files) > 1:
                # Cache the 3D volume to prevent expensive re-loading on every slider change
                if "volume_3d" not in st.session_state or st.session_state.get("last_uploaded_files") != uploaded_files:
                    with st.spinner("Reconstructing 3D Volume..."):
                        volume_3d, spacing_3d = load_dicom_3d_volume(files_data)
                        st.session_state.volume_3d = volume_3d
                        st.session_state.spacing_3d = spacing_3d
                        st.session_state.last_uploaded_files = uploaded_files
                
                volume_3d = st.session_state.volume_3d
                spacing_3d = st.session_state.spacing_3d
                    
                if volume_3d is not None:
                    col_mpr, col_3d_view = st.columns([1, 1.2])
                    
                    with col_mpr:
                        st.markdown("#### Multi-Planar Reconstruction (MPR)")
                        st.caption("Slice through anatomical planes in real-time.")
                        
                        z_max = volume_3d.shape[0] - 1
                        y_max = volume_3d.shape[1] - 1
                        x_max = volume_3d.shape[2] - 1

                        z_idx = st.slider("Axial (Z)", 0, z_max, z_max//2) if z_max > 0 else 0
                        y_idx = st.slider("Coronal (Y)", 0, y_max, y_max//2) if y_max > 0 else 0
                        x_idx = st.slider("Sagittal (X)", 0, x_max, x_max//2) if x_max > 0 else 0
                        
                        mpr_tab_1, mpr_tab_2, mpr_tab_3 = st.tabs(["Axial", "Coronal", "Sagittal"])
                        with mpr_tab_1:
                            fig_ax = create_mpr_slice(volume_3d, 'axial', z_idx, spacing_3d)
                            st.plotly_chart(fig_ax, use_container_width=True, config={'displayModeBar': False})
                        with mpr_tab_2:
                            fig_cor = create_mpr_slice(volume_3d, 'coronal', y_idx, spacing_3d)
                            st.plotly_chart(fig_cor, use_container_width=True, config={'displayModeBar': False})
                        with mpr_tab_3:
                            fig_sag = create_mpr_slice(volume_3d, 'sagittal', x_idx, spacing_3d)
                            st.plotly_chart(fig_sag, use_container_width=True, config={'displayModeBar': False})

                    with col_3d_view:
                        st.markdown("#### Interactive 3D Rendering")
                        st.caption("Rotate, zoom, and adjust tissue density.")
                        
                        threshold_val = st.select_slider("Tissue Density Threshold", 
                                                   options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                                                   value=0.4,
                                                   help="Lower values show soft tissue; higher values isolate bone.")
                        
                        fig_3d = create_3d_isosurface(volume_3d, spacing_3d, threshold_val)
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.error("3D rendering failed for this volume.")
                else:
                    st.error("#### 🧊 3D Reconstruction Unavailable")
                    st.warning("The system could not build a 3D volume from these files. This is usually because:")
                    st.write("- Files are standard images (PNG/JPG) rather than clinical **DICOM** files.")
                    st.write("- Slices have inconsistent dimensions or are missing spatial metadata.")
                    st.info("💡 **Clinical Tip:** For full 3D functionality, upload the original DICOM series exported from the hospital PACS.")
            else:
                st.warning("3D Visualization requires a volumetric study (multiple DICOM slices).")
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — DOCTOR REVIEW (Section by Section)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "REVIEW":
    res = st.session_state.analysis_results
    confirmed_count = sum(1 for s in REPORT_SECTIONS if st.session_state.section_confirmed.get(s))
    total = len(REPORT_SECTIONS)

    st.markdown("""
    <div class="report-header">
        <h2>Phase 3 — Radiologist Review & Editing</h2>
        <p>Review, edit and confirm each section individually. All sections must be confirmed before signing.</p>
    </div>
    """, unsafe_allow_html=True)

    # Patient strip
    p = st.session_state.patient_data
    st.markdown(f"""
    <div class="patient-strip">
        <div class="ps-item"><span class="ps-label">Patient</span><span class="ps-value">{p['name'] or '—'}</span></div>
        <div class="ps-item"><span class="ps-label">ID</span><span class="ps-value">{p['id'] or '—'}</span></div>
        <div class="ps-item"><span class="ps-label">Age</span><span class="ps-value">{p['age'] or '—'}</span></div>
        <div class="ps-item"><span class="ps-label">Date</span><span class="ps-value">{datetime.date.today().strftime('%d %b %Y')}</span></div>
        <div class="ps-item"><span class="ps-label">Referring Physician</span><span class="ps-value">{p.get('referrer','—') or '—'}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Progress
    st.progress(confirmed_count / total, text=f"Sections Reviewed: {confirmed_count} / {total}")

    # Advanced Triage: Severity Index
    severity = calculate_severity_index(res["tags"])
    col_sev, col_warn = st.columns([1, 2])
    with col_sev:
        st.metric("Clinical Severity Index", f"{severity} / 10")
    with col_warn:
        if severity > 7.0:
            st.error("🚨 **URGENT:** High severity findings detected. Prioritize clinical correlation and immediate review.")
        elif severity > 4.0:
            st.warning("⚠️ **MODERATE:** Significant degenerative changes detected.")
        else:
            st.info("ℹ️ **LOW:** Routine degenerative or normal findings.")
            
    if res.get("conflicts"):
        with st.expander("🚨 **REASONING ALERT: Logical Contradictions Detected**", expanded=True):
            for conflict in res["conflicts"]:
                st.error(conflict)
            st.caption("The AI has detected anatomical or clinical contradictions in the visual markers. Please verify these sections with extra care.")

    # Two-column layout: image + tags | sections

    # Two-column layout: image + tags | sections
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Volumetric Study Viewer**")

        volume_3d = st.session_state.get("volume_3d")
        spacing_3d = st.session_state.get("spacing_3d", (1.0, 1.0, 1.0))

        if res.get("is_3d") and volume_3d is not None:
            n_slices = volume_3d.shape[0]

            # ── Level jump buttons ──────────────────────────────────────────
            level_map = build_level_finding_map(st.session_state.get("section_texts", {}))
            if level_map:
                st.caption("**Jump to spinal level:**")
                sorted_levels = sorted(
                    level_map.keys(),
                    key=lambda lev: level_to_slice_index(lev, n_slices)
                )
                btn_cols = st.columns(min(len(sorted_levels), 5))
                for i, lev in enumerate(sorted_levels):
                    tooltip = "; ".join(level_map[lev][:2])
                    with btn_cols[i % len(btn_cols)]:
                        if st.button(lev, key=f"lvl_{lev}", help=tooltip):
                            st.session_state.review_target_slice = level_to_slice_index(lev, n_slices)
                            st.rerun()

            # ── Axial MPR slice viewer ──────────────────────────────────────
            default_slice = st.session_state.get("review_target_slice", n_slices // 2)
            slice_idx = st.slider("Axial Slice", 0, n_slices - 1, value=default_slice)
            st.session_state.review_target_slice = slice_idx

            fig_ax = create_mpr_slice(volume_3d, "axial", slice_idx, spacing_3d)
            if fig_ax:
                st.plotly_chart(fig_ax, use_container_width=True,
                                config={"displayModeBar": False})
        elif res.get("is_3d") and "processed_volume" in st.session_state:
            vol = st.session_state.processed_volume
            slice_idx = st.slider("Slice Navigator (3D Stack)", 0, len(vol) - 1, len(vol) // 2)
            st.image(vol[slice_idx],
                     caption=f"Slice {slice_idx + 1} of {len(vol)} (3D Consensus)",
                     use_container_width=True)
        else:
            st.image(res["image"], caption="Key Study Slice", use_container_width=True)

        if res.get("profile") is not None:
            st.line_chart(res["profile"], height=100)
            st.caption("📏 3D Volumetric Signal Intensity Map")

        st.markdown("**3D Consensus Findings & Mapping**")
        for tag in res["tags"]:
            label = tag['label']
            modifier = tag.get('modifier', 'Possible')
            root_info = tag.get('root_info', '')
            score = tag['score']
            
            # Clinical Color Coding
            color = "blue"
            if modifier == "Definite": color = "green"
            elif modifier == "Likely": color = "orange"
            elif modifier == "Probable": color = "red"
            
            display_text = f":{color}[**[{modifier}]**] {label}"
            if root_info: display_text += f" → {root_info}"
            
            st.progress(min(score, 1.0), text=display_text)

        if res.get("metrics"):
            st.markdown("**📏 Quantitative Evidence**")
            cols = st.columns(2)
            metrics = list(res["metrics"].items())
            for i, (k, v) in enumerate(metrics):
                cols[i % 2].metric(label=k, value=v)

        with st.expander("🔍 **Full Reasoning Audit (SIR JSON)**"):
            st.json(res.get("sir_json", {}))

        with st.expander("Similar Historical Patterns"):
            for case in res.get("similar", []):
                st.write(f"📄 `{case['path']}` — {case['similarity']:.1%} match")

    with col_right:
        st.markdown("**Section-by-Section Review**")
        st.caption("Edit the AI-drafted text, then check the confirmation box to mark each section as reviewed.")

        for section in REPORT_SECTIONS:
            is_confirmed = st.session_state.section_confirmed.get(section, False)
            card_class   = "confirmed" if is_confirmed else "pending"
            badge        = '<span class="badge-confirmed">✓ Confirmed</span>' if is_confirmed else '<span class="badge-pending">Pending Review</span>'

            st.markdown(f"""
            <div class="section-card {card_class}">
                <div class="sec-title">{section} &nbsp; {badge}</div>
            </div>
            """, unsafe_allow_html=True)

            edited = st.text_area(
                label=f"Edit: {section}",
                value=st.session_state.section_texts.get(section, ""),
                height=150,
                key=f"edit_{section}",
                label_visibility="collapsed"
            )
            st.session_state.section_texts[section] = edited

            confirmed = st.checkbox(
                f"✅ I have reviewed and confirm this section is accurate",
                value=is_confirmed,
                key=f"chk_{section}"
            )
            st.session_state.section_confirmed[section] = confirmed
            st.markdown("---")

    # ── Sign-off box ──
    if all_sections_confirmed():
        st.markdown('<div class="sign-box">', unsafe_allow_html=True)
        st.success("✅ All sections reviewed and confirmed. Ready for digital sign-off.")
        col_name, col_btn = st.columns([2, 1])
        with col_name:
            signer = st.text_input("Radiologist Full Name & Credentials",
                                   placeholder="e.g. Dr. Jane Smith, MD, FRCR",
                                   value=st.session_state.get("signed_by",""))
            st.session_state.signed_by = signer
        with col_btn:
            st.write("")
            if st.button("🖊 Finalize & Sign Report", type="primary", disabled=not signer.strip()):
                log(f"Report signed by {signer}.")
                st.session_state.state = "FINAL"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        remaining = [s for s in REPORT_SECTIONS if not st.session_state.section_confirmed.get(s)]
        st.warning(f"⚠️ {len(remaining)} section(s) still need review: **{', '.join(remaining)}**")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FINALIZED REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "FINAL":
    final_text = build_final_report()
    res = st.session_state.analysis_results

    st.markdown("""
    <div class="report-header">
        <h2>Phase 4 — Final Signed Clinical Report</h2>
        <p>Report is finalized and digitally signed. Ready for PACS archiving and download.</p>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✅ Signed by **{st.session_state.signed_by}** on {datetime.date.today().strftime('%d %B %Y')}")

    col_img, col_rep = st.columns([1, 2])
    with col_img:
        st.markdown("**Reference Image**")
        st.image(res["image"], use_container_width=True)

    with col_rep:
        st.markdown("**Final Radiology Report**")
        for section in REPORT_SECTIONS:
            with st.expander(f"📄 {section}", expanded=(section == "IMPRESSION")):
                st.markdown(st.session_state.section_texts.get(section, "").replace("\n","  \n"))

    st.markdown("---")
    
    # ── Self-Training Feedback Loop ──
    st.markdown("#### 🧠 Clinical Contribution Decision")
    st.info("Your clinical edits are valuable. By sharing your finalized report, you help train Nepal's own medical AI (RLHF Loop).")
    
    col_contribute, col_skip = st.columns(2)
    
    with col_contribute:
        if st.button("🚀 YES - Contribute My Edits", type="primary", use_container_width=True):
            mgr = FeedbackManager()
            for section in REPORT_SECTIONS:
                original = "AI Draft"
                final = st.session_state.section_texts.get(section, "")
                if original != final:
                    mgr.log_correction(
                        original_input=f"Analyze spine MRI for {section}",
                        ai_output=original,
                        doctor_correction=final,
                        clinical_context={"modality": "Spine MRI", "radiologist": st.session_state.signed_by}
                    )
            st.success("✅ Expertise added to National Dataset.")

    with col_skip:
        if st.button("📥 NO - Download Only", use_container_width=True):
            st.toast("Proceeding to download only.")

    st.markdown("---")
    # PDF Export
    export_patient_data = {**st.session_state.patient_data, "signed_by": st.session_state.signed_by}
    exporter = ReportExporter(REPORT_SECTIONS)
    pdf_bytes = exporter.generate_clinical_pdf(
        export_patient_data,
        final_text,
        findings_summary=[t['label'] for t in res["tags"] if t['score'] > 0.2],
        image_bytes=res["image"]
    )

    col_dl, col_new = st.columns([1, 1])
    with col_dl:
        st.download_button("Download Official PDF", data=pdf_bytes, file_name=f"SpineAI_{st.session_state.patient_data['id']}.pdf", mime="application/pdf", type="primary", use_container_width=True)

    with col_new:
        if st.button("Start New Patient Case", use_container_width=True):
            for k in ["state","section_texts","section_confirmed","raw_report","analysis_results","signed_by"]:
                st.session_state.pop(k, None)
            init_state()
            st.rerun()

    st.caption("⚠️ Clinical Governance: This report was drafted by an AI and finalized by the signing radiologist.")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — COMMUNITY HEALTH (FCHV MODE)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "FCHV_MODE":
    st.markdown("""
    <div class="report-header">
        <h2>🏥 Community Health Portal</h2>
        <p>Simplified Symptom Checker & Triage for Nepal's Health Volunteers (FCHVs).</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Main App"):
        st.session_state.state = "INTAKE"
        st.rerun()

    # Patient History Lookup
    patient_id = st.text_input("🔍 Search Patient ID (History Lookup)", placeholder="e.g. P-001")
    history_mgr = PatientHistory()
    if patient_id:
        history = history_mgr.get_history(patient_id)
        if history:
            with st.expander(f"📜 Past Visits for {patient_id} ({len(history)})"):
                for h in history[-5:]: # Show last 5
                    st.markdown(f"- **{h['timestamp'][:10]}**: {h['data'].get('symptoms', 'No symptoms recorded')[:50]}...")
        else:
            st.caption("No prior history found for this ID.")

    col_input, col_advice = st.columns([1, 1])

    with col_input:
        st.markdown("### 📋 Symptom Entry")
        if st.button("🎤 Start Voice Recording (Simulation)"):
            with st.spinner("Listening (Nepali STT)..."):
                time.sleep(2)
                st.session_state.fchv_symptoms = "fever for 10 days and jaundice"
                st.success("Transcribed: 'fever for 10 days and jaundice'")
        
        symptoms = st.text_area("Patient Symptoms", 
                                value=st.session_state.get("fchv_symptoms", ""),
                                height=200)
        
        if st.button("🧠 Analyze Symptoms", type="primary"):
            if patient_id:
                history_mgr.log_visit(patient_id, {"symptoms": symptoms})
            st.session_state.run_fchv_analysis = True
        
    if st.session_state.get("run_fchv_analysis") and symptoms:
        with col_advice:
            st.markdown("### 🏥 AI Clinical Guidance")
            triage_eng = OPDTriageEngine()
            triage_res = triage_eng.analyze_symptoms(symptoms)
            level = triage_res["triage_level"]
            safe_level = level if level in ("IMMEDIATE", "URGENT", "ROUTINE") else "ROUTINE"

            st.markdown(f"""<div class="triage-{safe_level}">
                <h1>{level}</h1><p>PRIORITY LEVEL</p>
            </div>""", unsafe_allow_html=True)
            
            st.warning(f"**Reasoning:** {triage_res['reason']}")
            
            with st.container(border=True):
                st.markdown("**📋 Recommended Action Plan**")
                st.write(triage_res['advice'])
                
                va = VoiceAssistant()
                nepali_voice_text = va.get_nepali_instructions(level)
                if st.button("🔊 Listen (Nepali Voice)"):
                    st.info(f"🔊 AI Speaking: '{nepali_voice_text}'")

            # Differential Diagnosis
            viz = DiffDiagVisualizer()
            diffs = viz.calculate_differentials(symptoms)
            if diffs:
                tracker = OutbreakTracker()
                tracker.log_case("Unknown", diffs[0]["condition"], level)
                st.markdown("**📊 Differential Diagnosis**")
                bar_colors = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#6B7280"}
                for d in diffs[:3]:
                    c = bar_colors.get(d["confidence"], "#6B7280")
                    st.markdown(f"""<div class="diff-row">
                        <div class="diff-bar" style="background:{c}"></div>
                        <span class="diff-name">{d['condition']}</span>
                        <span class="diff-pct">{d['likelihood']}%</span>
                    </div>""", unsafe_allow_html=True)
            
            if level in ["IMMEDIATE", "URGENT"]:
                st.markdown("---")
                if st.button("📄 Generate Referral Note"):
                    ref_gen = ReferralGenerator()
                    note = ref_gen.generate_note({"name": "Patient", "symptoms": symptoms}, triage_res)
                    st.download_button("💾 Download Note", note, file_name="referral.txt")
                    
                    # Notify Referral Center
                    notifier = ClinicalNotifier()
                    success, msg = notifier.notify_referral("Unknown", "Patient", diffs[0]["condition"], level)
                    if success:
                        st.success(f"📡 Notification sent to Referral Center!")
                        with st.expander("View Alert Message"):
                            st.code(msg)

    st.markdown("---")
    with st.expander("🧪 Laboratory Data Interpretation"):
        alt = st.number_input("ALT (U/L)", min_value=0, value=0)
        if st.button("Analyze Labs"):
            lab = LabInterpreter()
            res = lab.interpret_lft({"ALT": alt})
            st.write(res["summary"])
            st.success(res["nepali_explanation"])

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — NEPALMED ACADEMY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "ACADEMY_MODE":
    st.markdown("## 🎓 NepalMed Academy")
    if st.button("← Back "):
        st.session_state.state = "INTAKE"
        st.rerun()
    academy = AcademyEngine()
    if "current_scenario" not in st.session_state:
        st.session_state.current_scenario = academy.get_random_scenario()
    s = st.session_state.current_scenario
    st.info(f"**Scenario:** {s['title']}\n\n{s['symptoms']}")
    choice = st.radio("What is the correct action?", s['options'])
    if st.button("Submit Answer"):
        is_correct, expl = academy.evaluate_answer(s['id'], s['options'].index(choice))
        if is_correct: st.success("Correct!")
        else: st.error("Incorrect.")
        st.write(f"**Explanation:** {expl}")
        if st.button("Next Case"):
            st.session_state.current_scenario = academy.get_random_scenario()
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — PHARMACY INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "INVENTORY_MODE":
    st.markdown("## 💊 National Pharmacy Inventory")
    if st.button("← Back  "):
        st.session_state.state = "INTAKE"
        st.rerun()
    sm = StockManager()
    stock = sm.get_stock()
    st.markdown("### Essential Stock Levels")
    for item, level in stock.items():
        status = sm.check_availability(item)
        st.metric(item, f"{level} units", delta=status)
    
    st.markdown("---")
    new_qty = st.number_input("Adjust Stock (+/-)", value=0)
    item_upd = st.selectbox("Item", list(stock.keys()))
    if st.button("Update Stock"):
        if sm.update_stock(item_upd, new_qty):
            st.success("Stock updated.")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — NATIONAL SURVEILLANCE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "SURVEILLANCE_MODE":
    st.markdown("""
    <div class="report-header">
        <h2>📈 National Surveillance Dashboard</h2>
        <p>Real-time monitoring of disease outbreaks and clinical priorities across Nepal.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back"):
        st.session_state.state = "INTAKE"
        st.rerun()

    tracker = OutbreakTracker()
    stats = tracker.get_district_stats()
    
    if stats:
        st.markdown("### 🗺️ Case Distribution by District")
        st.bar_chart(stats)
        
        # Epidemic Predictions
        st.markdown("---")
        st.markdown("### 🔮 AI Epidemic Predictions")
        predictor = EpidemicPredictor()
        predictions = predictor.analyze_trends()
        if predictions:
            for p in predictions:
                st.error(f"**RISK: {p['condition']} ({p['risk_level']})**")
                st.write(f"- **Reason:** {p['reason']}")
                st.write(f"- **Districts Involved:** {', '.join(p['districts'])}")
        else:
            st.success("No active outbreak risks detected by AI.")

        st.markdown("---")
        st.markdown("### 📋 Recent Surveillance Alerts")
        if os.path.exists(tracker.log_path):
            with open(tracker.log_path, "r") as f:
                logs = [json.loads(line) for line in f.readlines()]
                st.table(logs[-10:]) 
    else:
        st.info("No surveillance data recorded yet. Data is automatically populated as cases are triaged.")

