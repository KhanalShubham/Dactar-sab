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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --primary: #0EA5E9;
    --primary-dark: #0284C7;
    --primary-light: #F0F9FF;
    --bg-main: #F8FAFC;
    --text-main: #0F172A;
    --text-muted: #64748B;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --border: #E2E8F0;
    --card-bg: #FFFFFF;
}

/* Force light theme for clinical clarity */
.stApp {
    background-color: var(--bg-main) !important;
}

/* Global Font & Color Overrides */
html, body, [class*="css"], .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main) !important;
}

/* Header Branding */
.main-header {
    display: flex;
    align-items: center;
    margin-bottom: 2rem;
    gap: 1.2rem;
    padding: 1rem 0;
}

.logo-box {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white !important;
    padding: 10px;
    border-radius: 14px;
    font-weight: 800;
    font-size: 1.6rem;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
}

.report-header {
    margin-bottom: 2.5rem;
    padding: 1.5rem 2rem;
    background: white;
    border-radius: 18px;
    border: 1px solid var(--border);
    border-left: 6px solid var(--primary);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.report-header h2 {
    margin: 0 !important;
    color: var(--primary-dark) !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.02em;
}

.report-header p {
    margin: 0.6rem 0 0 0 !important;
    color: var(--text-muted) !important;
    font-size: 1rem !important;
}

/* Inputs & Forms */
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
    background-color: white !important;
    color: var(--text-main) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.85rem !important;
    font-size: 1rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.15) !important;
}

/* Checkboxes */
.stCheckbox label {
    font-weight: 600 !important;
    color: var(--text-main) !important;
    font-size: 0.95rem !important;
}

[data-testid="stCheckbox"] {
    background-color: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stCheckbox"]:has(input:checked) {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}

[data-testid="stCheckbox"] input:checked + div {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 25px -5px rgba(14, 165, 233, 0.3) !important;
    opacity: 0.95;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: white !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding-top: 2rem !important;
}

/* Section Cards in Review */
.section-card {
    background: white !important;
    border: 1px solid var(--border) !important;
    padding: 1.2rem 1.5rem !important;
    border-radius: 16px !important;
    margin-bottom: 0.8rem !important;
    transition: all 0.2s ease !important;
}

.section-card.confirmed {
    border-left: 6px solid var(--success) !important;
    background: #F0FDF4 !important;
}

.section-card.pending {
    border-left: 6px solid var(--warning) !important;
}

.sec-title {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Badges */
.badge-confirmed { 
    background: #DCFCE7 !important; 
    color: #166534 !important; 
    padding: 4px 12px !important; 
    border-radius: 99px !important; 
    font-size: 0.75rem !important; 
    font-weight: 700 !important; 
}

.badge-pending { 
    background: #FEF3C7 !important; 
    color: #92400E !important; 
    padding: 4px 12px !important; 
    border-radius: 99px !important; 
    font-size: 0.75rem !important; 
    font-weight: 700 !important; 
}

/* Sign-off Box */
.sign-box {
    background: #F8FAFC !important;
    padding: 2rem !important;
    border-radius: 20px !important;
    border: 2px dashed var(--primary) !important;
    margin-top: 2.5rem !important;
}

/* Hide Streamlit default components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Utility Classes */
.patient-strip {
    display: flex;
    justify-content: space-between;
    background: #F1F5F9;
    padding: 1.2rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 1.2rem;
}

.patient-strip span {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-main) !important;
}

/* Severity Metrics */
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: var(--primary) !important;
    letter-spacing: -0.03em !important;
}

/* Info/Warning Boxes */
.stAlert {
    border-radius: 14px !important;
    border: none !important;
    background-color: #F1F5F9 !important;
}

.stAlert [data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
}

/* Mobile Adjustments */
@media (max-width: 768px) {
    .main-header { flex-direction: column; align-items: flex-start; }
    .patient-strip { flex-direction: column; gap: 0.5rem; }
}
</style>
""", unsafe_allow_html=True)

def branding_header():
    st.markdown("""
        <div class="main-header">
            <div class="logo-box">S</div>
            <div>
                <h1 style='margin:0; font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;'>SpineAI Pro</h1>
                <p style='margin:0; color: #64748B; font-size: 1rem; font-weight: 500;'>Advanced Neuroradiology Diagnostic Infrastructure</p>
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
                            tags = analyzer.analyze_volume_consensus([f.getvalue() for f in uploaded_files])
                        else:
                            tags = analyzer.auto_tag_findings(uploaded_files[0].getvalue())
                        
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
                        # Use middle slice for search/reference
                        ref_image = uploaded_files[len(uploaded_files)//2].getvalue()
                        similar = analyzer.find_similar_cases(ref_image)
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
        <span><strong>Patient:</strong> {p['name']}</span>
        <span><strong>ID:</strong> {p['id']}</span>
        <span><strong>Age:</strong> {p['age']}</span>
        <span><strong>Date:</strong> {datetime.date.today().strftime('%d %b %Y')}</span>
        <span><strong>Referring:</strong> {p.get('referrer','N/A')}</span>
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
        if res.get("is_3d") and "processed_volume" in st.session_state:
            vol = st.session_state.processed_volume
            slice_idx = st.slider("Slice Navigator (3D Stack)", 0, len(vol)-1, len(vol)//2)
            st.image(vol[slice_idx], caption=f"Slice {slice_idx+1} of {len(vol)} (3D Consensus)", use_container_width=True)
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
            color = {"IMMEDIATE": "#EF4444", "URGENT": "#F59E0B", "ROUTINE": "#0EA5E9"}.get(level, "gray")
            
            st.markdown(f"""<div style="background:{color}; color:white; padding:20px; border-radius:16px; text-align:center;">
                <h1 style="margin:0; color:white;">{level}</h1><p style="margin:0; color:white;">PRIORITY LEVEL</p>
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
                for d in diffs[:3]:
                    c = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#6B7280"}.get(d["confidence"])
                    st.markdown(f"<div style='border-left: 5px solid {c}; padding-left: 10px;'>{d['condition']} ({d['likelihood']}%)</div>", unsafe_allow_html=True)
            
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

