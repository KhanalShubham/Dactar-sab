import streamlit as st
import datetime
import re
import os
import json
import numpy as np
from dotenv import load_dotenv
from clip_integration import MedicalCLIPAnalyzer
from utils import ReportExporter, ConfigManager, REPORT_SECTIONS
from huggingface_hub import InferenceClient

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
        "stenosis": 2.0,       # Reduced from 3.0
        "compression": 3.0,    # Reduced from 4.0
        "fracture": 4.5,       # Reduced from 5.0
        "protrusion": 1.5,     # Reduced from 2.0
        "extrusion": 2.5,      # Reduced from 3.5
        "narrowing": 1.0,      # Reduced from 1.5
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
# MODERN CLINICAL THEME (Pro Version)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --primary: #0EA5E9;
    --primary-dark: #0284C7;
    --bg-light: #F8FAFC;
    --text-main: #1E293B;
    --text-muted: #64748B;
    --glass-bg: rgba(255, 255, 255, 0.75);
    --glass-border: rgba(255, 255, 255, 0.4);
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-main);
}

.stApp {
    background: radial-gradient(circle at top right, #F0F9FF, #F8FAFC);
}

/* Header Branding */
.main-header {
    display: flex;
    align-items: center;
    margin-bottom: 2rem;
    gap: 1rem;
}

.logo-box {
    background: var(--primary);
    color: white;
    padding: 10px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 1.5rem;
    width: 45px;
    height: 45px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Glassmorphism Section Cards */
.section-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.section-card.confirmed {
    border-left: 6px solid var(--success);
    background: rgba(255, 255, 255, 0.9);
}

.section-card.pending {
    border-left: 6px solid var(--warning);
}

.sec-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Status Badges */
.badge-confirmed { 
    background: #DCFCE7; 
    color: #166534; 
    padding: 4px 12px; 
    border-radius: 99px; 
    font-size: 0.75rem; 
    font-weight: 600; 
    text-transform: uppercase;
}

.badge-pending { 
    background: #FEF3C7; 
    color: #92400E; 
    padding: 4px 12px; 
    border-radius: 99px; 
    font-size: 0.75rem; 
    font-weight: 600; 
    text-transform: uppercase;
}

/* Professional Buttons */
.stButton>button {
    border-radius: 12px;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    border: none !important;
    background-color: var(--primary) !important;
    color: white !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.3);
    background-color: var(--primary-dark) !important;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #E2E8F0;
}

/* Text Areas for Radiology Reports */
.stTextArea textarea {
    font-family: 'Source Serif 4', serif;
    font-size: 1rem;
    line-height: 1.6;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background: white !important;
}

/* Hide Streamlit default components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def branding_header():
    st.markdown("""
        <div class="main-header">
            <div class="logo-box">S</div>
            <div>
                <h1 style='margin:0; font-size: 1.8rem;'>SpineAI Pro</h1>
                <p style='margin:0; color: #64748B; font-size: 0.9rem;'>Expert-Tier Neuroradiology Diagnostic Engine</p>
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
    with st.expander("🛠️ Developer Console"):
        st.caption("Admin & System Management")
        if st.button("🔄 Re-index Vector DB", help="Scans historical_cases/ and rebuilds FAISS index"):
            import subprocess
            try:
                subprocess.run([".venv/Scripts/python.exe", "index_cases.py", "--dir", "historical_cases"], check=True)
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
        # Real-Time 3D Volumetric Processing
        from image_analysis import process_medical_image, process_dicom_volume
        
        if len(uploaded_files) > 1:
            files_data = [(f.getvalue(), f.name) for f in uploaded_files]
            processed_slices, vol_profile, dcm_meta = process_dicom_volume(files_data)
            st.session_state.processed_volume = processed_slices
            active_image = processed_slices[len(processed_slices)//2] # Middle slice as preview
            intensity_profile = vol_profile
            study_type_label = f"Volumetric Study ({len(uploaded_files)} slices)"
        else:
            f = uploaded_files[0]
            processed_bytes, intensity_profile, dcm_meta = process_medical_image(f.getvalue(), f.name)
            active_image = processed_bytes
            st.session_state.processed_volume = [processed_bytes]
            study_type_label = "Single-Slice Study"

        # Auto-fill Metadata if DICOM
        if dcm_meta:
            st.session_state.patient_data["name"] = dcm_meta["name"]
            st.session_state.patient_data["id"] = dcm_meta["id"]
            st.session_state.patient_data["age"] = dcm_meta["age"]
            st.toast("✅ Volumetric metadata extracted.")

        col_prev, col_action = st.columns([1, 1])
        with col_prev:
            st.image(active_image, caption=f"{study_type_label} - Key Slice Preview", use_container_width=True)
            if intensity_profile is not None:
                st.line_chart(intensity_profile, height=150, use_container_width=True)
                st.caption("📏 3D Mean T2 Signal Intensity Profile")
        
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
                    from clip_integration import AnatomicalConstraintEngine, ConsistencyEngine, CausalReasoningEngine
                    
                    log("STAGE 1: Volumetric feature extraction...")
                    analyzer = get_analyzer()
                    if len(uploaded_files) > 1:
                        tags = analyzer.analyze_volume_consensus([f.getvalue() for f in uploaded_files])
                    else:
                        tags = analyzer.auto_tag_findings(uploaded_files[0].getvalue())
                    
                    log("STAGE 2: Applying Anatomical Constraints (ACE)...")
                    tags = AnatomicalConstraintEngine.apply_constraints(tags)
                    
                    log("STAGE 3: Consistency & Causal Reasoning...")
                    from clip_integration import SymbolicCausalValidator
                    
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
                    from utils import QuantitativeAnalyzer
                    st.session_state.analysis_results["metrics"] = QuantitativeAnalyzer.estimate_metrics(tags)
                    
                    st.session_state.state = "REVIEW"
                    st.rerun()

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
            st.write("")
            if st.button("🖊 Finalize & Sign Report", type="primary", disabled=not signer.strip()):
                log(f"Report signed by {signer}.")
                st.session_state.state = "FINAL"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
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
    # Disclaimer
    st.caption("⚠️ This report was drafted by an AI (SpineAI + CLIP) and reviewed, edited, and finalized by the signing radiologist. It does not replace a full clinical assessment.")

    # PDF Export — inject signed_by so it appears in the signature block
    export_patient_data = {**st.session_state.patient_data,
                           "signed_by": st.session_state.signed_by}
    exporter = ReportExporter()
    pdf_bytes = exporter.generate_clinical_pdf(
        export_patient_data,
        final_text,
        findings_summary=[t['label'] for t in res["tags"] if t['score'] > 0.2],
        image_bytes=res["image"]
    )


    col_dl, col_new = st.columns([1, 3])
    with col_dl:
        st.download_button(
            " Download PDF",
            data=pdf_bytes,
            file_name=f"SpineAI_{st.session_state.patient_data['id']}_{datetime.date.today()}.pdf",
            mime="application/pdf",
            type="primary"
        )
    
    with col_dl:
        # Phase 7.5: Structured JSON Export
        structured_json = json.dumps({
            "patient": st.session_state.patient_data,
            "findings": res["tags"],
            "metrics": res.get("metrics", {}),
            "report_sections": st.session_state.section_texts,
            "signed_by": st.session_state.signed_by
        }, indent=2)
        
        st.download_button(
            " Download JSON",
            data=structured_json,
            file_name=f"SpineAI_Structured_{st.session_state.patient_data['id']}.json",
            mime="application/json",
            help="Export clinical-grade structured data for PACS/EMR integration"
        )

    with col_dl:
        if st.button("📤 Export for Annotation", help="Saves image + tags for Label Studio"):
            path = export_for_annotation(
                res["image"], res["tags"], final_text, st.session_state.patient_data['id']
            )
            st.toast(f"Exported to {path}")
            log(f"Case exported for annotation: {st.session_state.patient_data['id']}")

    with col_new:
        if st.button("Start New Patient Case"):
            for k in ["state","section_texts","section_confirmed","raw_report","analysis_results","signed_by"]:
                st.session_state.pop(k, None)
            init_state()
            st.rerun()
