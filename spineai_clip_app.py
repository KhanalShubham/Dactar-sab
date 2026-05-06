import streamlit as st
import datetime
import re
import os
import json
import numpy as np
from clip_integration import MedicalCLIPAnalyzer
from utils import ReportExporter, ConfigManager, REPORT_SECTIONS
from huggingface_hub import InferenceClient

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
@st.cache_resource
def get_analyzer():
    """Loads and caches the BiomedCLIP analyzer in GPU memory."""
    # This prevents redundant model reloads, ensuring real-time performance.
    return MedicalCLIPAnalyzer()

def generate_narrative(hf_token, tags, patient_info, intensity_profile=None, similar_cases=None):
    if not hf_token:
        return "ERROR: No Hugging Face API token provided."
    try:
        client = InferenceClient(api_key=hf_token)
        tags_str = "\n".join([f"  - {t['label']} (Confidence: {t['score']:.2f})" for t in tags])
        
        intensity_summary = "Not available"
        if intensity_profile is not None:
            avg_sig = np.mean(intensity_profile)
            intensity_summary = f"Mean Signal: {avg_sig:.2f}. "
            if avg_sig < 0.4:
                intensity_summary += "Signal loss suggests multilevel disc desiccation."

        sim_str = "No similar cases found."
        if similar_cases:
            sim_str = "\n".join([f"- Case: {c['path']} (Match: {c['similarity']:.2f})" for c in similar_cases[:3]])

        system_msg = """You are a Senior Neuroradiology Consultant. Provide a high-precision, multi-section report based on AI visual markers.
RULES:
1. BE DECISIVE: Do not use "cannot be assessed". Use the AI markers to describe findings.
2. EXPERT TERMINOLOGY: Use terms like "thecal sac effacement", "foraminal patency", "lordosis", "Modic changes".
3. NO EMPTY SECTIONS: Every section below MUST be filled with clinical interpretation.

Sections:
TECHNIQUE:
VERTEBRAL ALIGNMENT:
BONE MARROW SIGNAL:
DISC ASSESSMENT:
  L1-L2, L2-L3, L3-L4, L4-L5, L5-S1:
SPINAL CANAL & THECAL SAC:
FORAMINAL ASSESSMENT:
FACET JOINTS:
IMPRESSION:
"""
        user_msg = f"""Patient: {patient_info}
Visual Markers: {tags_str}
Intensity: {intensity_summary}
Similar Historical Patterns: {sim_str}

TASK: YOU MUST GENERATE A COMPLETE, PROFESSIONAL RADIOLOGY REPORT. 
- EVERY section from TECHNIQUE to IMPRESSION must have at least 2-3 sentences of clinical description.
- For 'DISC ASSESSMENT', you MUST describe each level (L1-L2 through L5-S1) individually.
- If no abnormal markers are present for a section, describe the anatomy as normal (e.g., 'Normal vertebral body signal', 'Alignment is within normal limits').
- DO NOT use generic placeholders. Use the data provided above to create a bespoke interpretation."""

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=2500,
            temperature=0.3
        )
        raw_content = response.choices[0].message.content
        log(f"AI Success: {len(raw_content)} chars generated.")
        return raw_content
    except Exception as e:
        log(f"LLM Error: {e}")
        return f"Report generation failed: {e}"

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

def collect_hard_case(image_bytes, tags, reason="low_confidence"):
    """Flags cases where the AI is uncertain for priority review."""
    hard_dir = "priority_review"
    if not os.path.exists(hard_dir):
        os.makedirs(hard_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    case_id = f"hard_{timestamp}"
    
    with open(os.path.join(hard_dir, f"{case_id}.png"), "wb") as f:
        f.write(image_bytes)
    
    with open(os.path.join(hard_dir, f"{case_id}.json"), "w") as f:
        json.dump({"tags": tags, "reason": reason}, f)
    
    log(f"Hard case flagged: {reason}")

def calculate_severity_index(tags):
    """Calculates a 1-10 severity score based on finding keywords and scores."""
    score = 1.0
    critical_keywords = {
        "stenosis": 3.0,
        "compression": 4.0,
        "fracture": 5.0,
        "protrusion": 2.0,
        "extrusion": 3.5,
        "narrowing": 1.5,
        "spondylolisthesis": 2.5
    }
    
    for tag in tags:
        label = tag['label'].lower()
        conf = tag['score']
        for kw, weight in critical_keywords.items():
            if kw in label:
                score += weight * conf
                
    return min(10.0, round(score, 1))

# ─────────────────────────────────────────────────────────────────────────────
# ACADEMIC CSS THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f4f6f9;
}

/* Header band */
.report-header {
    background: linear-gradient(135deg, #1a2b4a 0%, #243b67 100%);
    color: white;
    padding: 18px 28px;
    border-radius: 10px;
    margin-bottom: 22px;
}
.report-header h2 { margin: 0; font-family: 'Source Serif 4', serif; font-size: 1.5rem; }
.report-header p  { margin: 4px 0 0; font-size: 0.85rem; opacity: 0.75; }

/* Section card */
.section-card {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.section-card.confirmed {
    border-left: 5px solid #2e7d32;
}
.section-card.pending {
    border-left: 5px solid #e0a500;
}

/* Section title */
.sec-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a2b4a;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Status badge */
.badge-confirmed { background: #e8f5e9; color: #2e7d32; padding: 2px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-pending   { background: #fff8e1; color: #e0a500; padding: 2px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }

/* Finalize button */
.sign-box {
    background: #e8f0fe;
    border: 1px solid #b3c3f4;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 20px;
}

/* Patient strip */
.patient-strip {
    background: #1a2b4a;
    color: white;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 0.88rem;
    margin-bottom: 16px;
    display: flex;
    gap: 30px;
}

/* Progress bar colours */
div[data-testid="stProgress"] > div { background-color: #d0e4f7; }
div[data-testid="stProgress"] > div > div { background-color: #1a2b4a; }

/* Text area */
.stTextArea textarea {
    font-family: 'Source Serif 4', serif;
    font-size: 0.93rem;
    line-height: 1.65;
    border: 1px solid #c8d3e0 !important;
    border-radius: 6px !important;
}

/* Warning / info */
.stAlert { border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

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

    uploaded_file = st.file_uploader("Select MRI file", type=["png","jpg","jpeg","dcm"])

    if uploaded_file:
        # Real-Time Processing
        from image_analysis import process_medical_image
        processed_bytes, intensity_profile, dcm_meta = process_medical_image(
            uploaded_file.getvalue(), uploaded_file.name
        )

        # Auto-fill Metadata if DICOM
        if dcm_meta:
            st.session_state.patient_data["name"] = dcm_meta["name"]
            st.session_state.patient_data["id"] = dcm_meta["id"]
            st.session_state.patient_data["age"] = dcm_meta["age"]
            st.toast("✅ Real-time DICOM metadata extracted.")

        col_prev, col_action = st.columns([1, 1])
        with col_prev:
            st.image(processed_bytes, caption="Active Study Analysis", use_container_width=True)
            if intensity_profile is not None:
                st.line_chart(intensity_profile, height=150, use_container_width=True)
                st.caption("📏 Real-time T2 Signal Intensity Profile (Vertical Axis)")
        
        with col_action:
            st.markdown("**Real-Time Study Insights**")
            p = st.session_state.patient_data
            st.write(f"- **Patient Name:** {p['name']}")
            st.write(f"- **Medical ID:** {p['id']}")
            if dcm_meta:
                st.write(f"- **Modality:** {dcm_meta['modality']}")
                st.write(f"- **Description:** {dcm_meta['description']}")
            
            st.markdown("---")
            st.info("Clicking **Run AI Analysis** will use the cached BiomedCLIP engine for sub-second precise tagging.")

            if st.button("🧠 Run Precision AI Analysis", type="primary"):
                with st.spinner("Processing visual features…"):
                    log("Precision inference started.")
                    analyzer = get_analyzer()
                    tags = analyzer.auto_tag_findings(uploaded_file.getvalue())
                    log(f"Tagged {len(tags)} clinical findings.")

                    similar = analyzer.find_similar_cases(uploaded_file.getvalue())
                    p_info = f"{p['name']}, Age {p['age']}, ID {p['id']}"
                    raw = generate_narrative(hf_token, tags, p_info, intensity_profile=intensity_profile, similar_cases=similar)
                    
                    sections = parse_report_sections(raw, visual_tags=tags)
                    st.session_state.section_texts     = {s: sections.get(s, "") for s in REPORT_SECTIONS}
                    st.session_state.section_confirmed = {s: False for s in REPORT_SECTIONS}
                    st.session_state.raw_report        = raw
                    st.session_state.analysis_results  = {
                        "tags": tags, "image": processed_bytes, "similar": similar, "profile": intensity_profile
                    }
                    
                    max_score = max([t['score'] for t in tags]) if tags else 0
                    if max_score < 0.15:
                        collect_hard_case(uploaded_file.getvalue(), tags, reason=f"low_conf_{max_score:.2f}")
                    
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

    # Two-column layout: image + tags | sections

    # Two-column layout: image + tags | sections
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Source Image**")
        st.image(res["image"], use_container_width=True)

        st.markdown("**AI Visual Tags**")
        for tag in res["tags"]:
            st.progress(min(tag['score'], 1.0), text=f"{tag['label']} ({tag['score']:.1%})")

        with st.expander("Similar Historical Cases"):
            for case in res["similar"]:
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
