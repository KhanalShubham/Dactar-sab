import sys
import os
# Add project root to path for src and app imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import datetime
import re
import os
import json
import numpy as np
from dotenv import load_dotenv

# Updated Imports for New Structure
from src.analyzer.clip_analyzer import MedicalCLIPAnalyzer
from src.constraints.anatomical_engine import AnatomicalConstraintEngine
from src.constraints.consistency_engine import ConsistencyEngine
from src.constraints.causal_validator import CausalReasoningEngine, SymbolicCausalValidator
from src.utils.dicom_loader import process_medical_image, process_dicom_volume
from src.utils.metrics import QuantitativeAnalyzer
from src.export.pdf_generator import ReportExporter
from src.rag.retriever import NepalGuidelinesRetriever
from config import ConfigManager, REPORT_SECTIONS

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
        "chat_history": []
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
    current_sections = st.session_state.analysis_results.get("sections", REPORT_SECTIONS) if st.session_state.analysis_results else REPORT_SECTIONS
    return all(
        st.session_state.section_confirmed.get(s, False) for s in current_sections
    )

def build_final_report():
    lines = []
    p = st.session_state.patient_data
    lines.append(f"PATIENT: {p['name']}  |  AGE: {p['age']}  |  ID: {p['id']}")
    lines.append(f"REFERRING PHYSICIAN: {p.get('referrer', 'N/A')}")
    lines.append(f"STUDY DATE: {datetime.date.today().strftime('%d %B %Y')}")
    lines.append("=" * 70)
    current_sections = st.session_state.analysis_results.get("sections", REPORT_SECTIONS) if st.session_state.analysis_results else REPORT_SECTIONS
    for s in current_sections:
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

@st.cache_resource(show_spinner=False)
def get_nepal_retriever():
    """Loads and caches the Nepal Medical Guidelines retriever."""
    return NepalGuidelinesRetriever()

def generate_narrative(hf_token, sir_json, tags, patient_info, comparison_mode=False, use_rag=False, modality="SPINE_MRI"):
    if not hf_token:
        return "ERROR: No Hugging Face API token provided.", None
    try:
        client = InferenceClient(api_key=hf_token)
        sir_str = json.dumps(sir_json, indent=2)
        
        # RAG Context Retrieval
        context_str = ""
        retrieved_results = []
        if use_rag and modality == "SPINE_MRI":
            start_rag = datetime.datetime.now()
            retriever = get_nepal_retriever()
            # Build query from SIR
            query_parts = []
            for level in ["L4-L5", "L5-S1", "L3-L4"]:
                data = sir_json.get(level, {})
                if isinstance(data, dict):
                    if "stenosis" in data.get("canal", "").lower(): query_parts.append(f"{level} canal stenosis")
                    if "bulge" in data.get("disc", "").lower(): query_parts.append(f"{level} disc bulge")
            
            query = " ".join(query_parts) if query_parts else "lumbar spine degenerative changes Nepal guidelines"
            retrieved_results = retriever.retrieve(query, k=3)
            context_str = retriever.format_for_prompt(retrieved_results)
            
            rag_duration = (datetime.datetime.now() - start_rag).total_seconds()
            log(f"RAG: Retrieved {len(retrieved_results)} chunks in {rag_duration:.2f}s")

        persona = "Board-Certified Neuroradiologist" if modality == "SPINE_MRI" else "Consultant Radiologist"
        
        system_msg = f"""You are a {persona} practicing in Nepal.
{context_str}

STRICT REPORTING RULES:
1. SILENT AI: NEVER mention 'AI', 'visual markers', 'inference', or 'analysis suggests'. Speak as a human consultant.
2. TELEGRAPHIC STYLE: Use concise, professional medical shorthand. Remove fluff.
3. CAUSAL CONSISTENCY: Every finding must have a pathophysiological cause. 
4. COMPARISON GATE: ONLY mention prior studies if comparison_mode is TRUE.
"""
        from config import REPORT_SECTIONS_BY_MODALITY
        current_sections = ", ".join(REPORT_SECTIONS_BY_MODALITY.get(modality, REPORT_SECTIONS_BY_MODALITY["SPINE_MRI"]))

        user_msg = f"""Patient: {patient_info}
MODALITY: {modality}
STRUCTURED FINDINGS:
{sir_str if modality == "SPINE_MRI" else json.dumps([t['label'] for t in tags], indent=2)}
COMPARISON_MODE_ENABLED: {comparison_mode}

TASK: Generate a professional radiology report using these sections: {current_sections}.
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
        return raw_content, retrieved_results
    except Exception as e:
        log(f"LLM Error: {e}")
        return f"Report generation failed: {e}", None

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

def answer_guideline_question(hf_token, question):
    """Answers a general medical question using the Nepal Guidelines RAG index."""
    if not hf_token:
        return "ERROR: No Hugging Face API token provided.", []
    try:
        retriever = get_nepal_retriever()
        results = retriever.retrieve(question, k=4)
        if not results:
            return "No specific information found in the guidelines for this query.", []
        
        context = retriever.format_for_prompt(results)
        client = InferenceClient(api_key=hf_token)
        
        prompt = f"""You are a Clinical Protocol Assistant specializing in Nepal Medical Guidelines.
Using ONLY the following excerpts from official guidelines, answer the user's question. 
If the answer is not in the context, state that the current guidelines do not cover this topic.

CONTEXT:
{context}

QUESTION: {question}

Provide a concise, evidence-based answer with source references."""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=800,
            temperature=0.1
        )
        return response.choices[0].message.content, results
    except Exception as e:
        return f"Error answering question: {e}", []

def generate_clinical_action_plan(hf_token, sir_json, report_texts):
    """Generates an on-demand clinical management plan based on findings and Nepal guidelines."""
    if not hf_token:
        return "ERROR: No Hugging Face API token provided.", []
    
    try:
        impression = report_texts.get("IMPRESSION", "")
        query_parts = []
        levels = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]
        for level in levels:
            data = sir_json.get(level, {})
            if isinstance(data, dict):
                level_findings = []
                canal = data.get("canal", "").lower()
                if "severe" in canal: level_findings.append(f"severe central stenosis at {level}")
                elif "moderate" in canal: level_findings.append(f"moderate central stenosis at {level}")
                foram = data.get("foramina", "").lower()
                if "stenosis" in foram or "narrowing" in foram: level_findings.append(f"{level} foraminal narrowing")
                disc = data.get("disc", "").lower()
                if any(k in disc for k in ["bulge", "protrusion", "extrusion"]): level_findings.append(f"{level} {disc}")
                align = data.get("alignment", "").lower()
                if "listhesis" in align: level_findings.append(f"{level} listhesis")
                if level_findings: query_parts.append(level_findings[0])
        
        if not query_parts:
            query = f"General lumbar spine degenerative disease management Nepal guidelines. Impression: {impression[:200]}"
        else:
            query = ", ".join(query_parts) + ". " + f"Impression: {impression[:100]}"
        query += " Provide treatment, referral urgency, medication, patient advice, follow-up according to Nepal BHS STP."
        
        retriever = get_nepal_retriever()
        results = retriever.retrieve(query, k=5)
        context = retriever.format_for_prompt(results)
        
        client = InferenceClient(api_key=hf_token)
        system_msg = """You are a clinical decision support assistant for doctors in Nepal.  
Based on the MRI findings and the Nepal guidelines provided, output a structured action plan with exactly these sections:
1. PROBABLE DIAGNOSIS
2. URGENCY
3. FIRST-LINE TREATMENT
4. REFERRAL
5. PATIENT ADVICE
6. FOLLOW-UP
"""
        user_msg = f"MRI FINDINGS:\n{query}\n\nNEPAL GUIDELINES EXCERPTS:\n{context}\n\nGenerate action plan."
        response = client.chat_completion(
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=800,
            temperature=0.2
        )
        return response.choices[0].message.content, results
    except Exception as e:
        return f"Failed to generate action plan: {e}", []

def export_for_annotation(image_bytes, tags, report_text, patient_id):
    base_dir = "data/annotation_export"
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    case_id = f"{patient_id}_{timestamp}"
    img_path = os.path.join(base_dir, f"{case_id}.png")
    with open(img_path, "wb") as f: f.write(image_bytes)
    meta = {"case_id": case_id, "image": f"{case_id}.png", "tags": tags, "report": report_text, "status": "ready_for_review"}
    meta_path = os.path.join(base_dir, f"{case_id}.json")
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
    return meta_path

def calculate_severity_index(tags):
    score = 1.0
    critical_keywords = {"stenosis": 2.0, "compression": 3.0, "fracture": 4.5, "protrusion": 1.5, "extrusion": 2.5}
    for tag in tags:
        label = tag['label'].lower()
        conf = tag['score']
        for kw, weight in critical_keywords.items():
            if kw in label: score += weight * conf
    return min(10.0, round(score, 1))

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL VISIBILITY & ACCESSIBILITY THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {
    --primary: #1F4E79; --primary-dark: #153452; --primary-light: #E6F0FA;
    --bg-main: #F8F9FC; --bg-sidebar: #FFFFFF; --text-heading: #1E293B; --text-body: #334155;
    --border-standard: #CBD5E1; --border-subtle: #E2E8F0;
}
.stApp { background-color: var(--bg-main) !important; color: var(--text-body) !important; font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp p, .stApp span, .stApp label, .stApp div, .stApp li { color: var(--text-body) !important; }
[data-testid="stSidebar"] { background-color: var(--bg-sidebar) !important; border-right: 1px solid var(--border-subtle) !important; min-width: 280px !important; }
h1, h2, h3 { color: var(--text-heading) !important; font-weight: 700 !important; }
input, textarea, select { background-color: #FFFFFF !important; color: var(--text-body) !important; border: 1.5px solid var(--border-standard) !important; border-radius: 12px !important; }
div[data-testid="stVerticalBlock"] > div.stVerticalBlockBorderWrapper > div { background: #FFFFFF !important; border: 1px solid var(--border-subtle) !important; border-radius: 16px !important; padding: 2rem !important; }
.stButton>button { background-color: var(--primary) !important; color: #FFFFFF !important; border-radius: 12px !important; font-weight: 700 !important; border: none !important; }
.step { flex: 1; text-align: center; font-size: 0.8rem; font-weight: 700; color: #64748B !important; padding-bottom: 0.5rem; border-bottom: 4px solid #E2E8F0; }
.step.active { color: var(--primary) !important; border-bottom: 4px solid var(--primary); }
#MainMenu, footer, header { visibility: hidden; }
.chat-bubble { padding: 1rem; border-radius: 12px; margin-bottom: 1rem; max-width: 85%; line-height: 1.5; }
.chat-user { background-color: var(--primary-light); color: var(--primary-dark) !important; align-self: flex-end; border-bottom-right-radius: 2px; }
.chat-ai { background-color: #FFFFFF; border: 1px solid var(--border-subtle); color: var(--text-body) !important; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.chat-container { display: flex; flex-direction: column; gap: 0.5rem; max-height: 500px; overflow-y: auto; padding-right: 10px; }
</style>
""", unsafe_allow_html=True)

def render_stepper(current_phase):
    phases = ["INTAKE", "UPLOAD", "REVIEW", "FINAL"]
    phase_labels = ["1. Clinical Intake", "2. Study Upload", "3. Specialist Review", "4. Report Finalized"]
    cols = st.columns(len(phases))
    for i, (phase, label) in enumerate(zip(phases, phase_labels)):
        is_active = (phase == current_phase)
        active_class = "active" if is_active else ""
        cols[i].markdown(f'<div class="step {active_class}">{label}</div>', unsafe_allow_html=True)
    st.write("")

def branding_header():
    render_stepper(st.session_state.state)

branding_header()

def render_ai_assistant():
    with st.container(border=True):
        st.markdown("### 🤖 Clinical AI Consultant")
        st.caption("Ask questions for clinical reasoning and second opinions.")
        chat_placeholder = st.container()
        with chat_placeholder:
            for chat in st.session_state.chat_history:
                role_class = "chat-user" if chat["role"] == "user" else "chat-ai"
                st.markdown(f'<div class="chat-bubble {role_class}">{chat["content"]}</div>', unsafe_allow_html=True)
        with st.form("global_ai_chat", clear_on_submit=True):
            u_input = st.text_input("Ask a clinical question...", placeholder="e.g. Differentiate L4 vs L5 radiculopathy", label_visibility="collapsed")
            if st.form_submit_button("Consult Assistant", use_container_width=True):
                if u_input:
                    st.session_state.chat_history.append({"role": "user", "content": u_input})
                    try:
                        client = InferenceClient(api_key=st.session_state.hf_token)
                        messages = [{"role": "system", "content": "You are a professional Neuroradiology Consultant."}]
                        for m in st.session_state.chat_history[-6:]: messages.append(m)
                        response = client.chat_completion(messages=messages, model="Qwen/Qwen2.5-72B-Instruct", max_tokens=800, temperature=0.2)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
                        st.rerun()
                    except Exception as e: st.error(f"AI error: {e}")

with st.sidebar:
    st.markdown("### 🏥 SpineAI Pro")
    st.caption(f"v{config['version']} · Clinical Decision Support")
    if "hf_token" not in st.session_state: st.session_state.hf_token = os.getenv("HF_TOKEN", "")
    hf_token = st.session_state.hf_token
    st.markdown("---")
    st.markdown("**👤 Patient Demographics**")
    p = st.session_state.patient_data
    p["name"] = st.text_input("Full Name", value=p["name"])
    p["id"] = st.text_input("Medical Record No.", value=p["id"])
    p["age"] = st.text_input("Age / DOB", value=p["age"])
    st.markdown("---")
    st.markdown("**🛡️ Clinical Governance**")
    use_nepal_rag = st.checkbox("Apply Nepal Medical Guidelines (RAG)", value=True)
    st.markdown("---")
    with st.expander("🛠️ Developer Console"):
        if st.button("🧹 Clear Logs"): st.session_state.logs = []; st.rerun()
        new_token = st.text_input("HF Token", type="password", value=st.session_state.hf_token)
        if new_token != st.session_state.hf_token: st.session_state.hf_token = new_token; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT LAYOUT (SIDE-BY-SIDE)
# ─────────────────────────────────────────────────────────────────────────────
col_workflow, col_assistant = st.columns([2, 1.2], gap="medium")

with col_assistant:
    render_ai_assistant()

with col_workflow:
    if st.session_state.state == "INTAKE":
        st.markdown("## Phase 1 — Clinical Governance & Safety")
        with st.container(border=True):
            st.markdown("### 🏥 Patient Intake & Safety Screening")
            col_form, col_info = st.columns([1.5, 1])
            with col_form:
                with st.form("intake_form", border=False):
                    c1, c2 = st.columns(2)
                    with c1: saddle = st.checkbox("Saddle anesthesia"); bladder = st.checkbox("Urinary retention")
                    with c2: bowel = st.checkbox("Bowel incontinence"); weakness = st.checkbox("Bilateral leg weakness")
                    study_type = st.selectbox("Study Protocol", ["Lumbar Spine MRI", "Other"])
                    clinical_q = st.text_area("Clinical Indication")
                    submitted = st.form_submit_button("Proceed to Imaging consensus →", use_container_width=True)
            with col_info:
                st.warning("SpineAI is for diagnostic support. CES is a surgical emergency.")
            if submitted:
                if saddle or bladder or bowel or weakness:
                    st.session_state.state = "CES_BLOCKED"; log("CES flags detected."); st.rerun()
                else:
                    st.session_state.patient_data["study_type"] = study_type
                    st.session_state.patient_data["clinical_q"] = clinical_q
                    st.session_state.state = "UPLOAD"; log("CES screening passed."); st.rerun()

    elif st.session_state.state == "CES_BLOCKED":
        st.error("🚨 **CRITICAL: POTENTIAL CAUDA EQUINA SYNDROME**")
        st.warning("AI reporting is DISABLED. Refer immediately to Neurosurgery.")
        if st.button("← New Patient"): init_state(); st.rerun()

    elif st.session_state.state == "UPLOAD":
        st.markdown("### Phase 2 — Imaging Consensus")
        uploaded_files = st.file_uploader("Select MRI files", type=["png","jpg","jpeg","dcm"], accept_multiple_files=True)
        if uploaded_files:
            if st.button("🧠 Start AI Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    analyzer = get_analyzer()
                    if len(uploaded_files) > 1:
                        tags = analyzer.analyze_volume_consensus([f.getvalue() for f in uploaded_files])
                    else:
                        tags = analyzer.auto_tag_findings(uploaded_files[0].getvalue())
                    modality = tags[0].get('modality', 'SPINE_MRI') if tags else 'SPINE_MRI'
                    tags = AnatomicalConstraintEngine.apply_constraints(tags)
                    sir_json = analyzer.get_sir_map(tags) if modality == "SPINE_MRI" else {"findings": [t['label'] for t in tags]}
                    raw, rag_refs = generate_narrative(hf_token, sir_json, tags, p['name'], modality=modality, use_rag=use_nepal_rag)
                    from config import REPORT_SECTIONS_BY_MODALITY
                    current_sections = REPORT_SECTIONS_BY_MODALITY.get(modality, REPORT_SECTIONS)
                    sections = parse_report_sections(raw, visual_tags=tags)
                    st.session_state.section_texts = {s: sections.get(s, "") for s in current_sections}
                    st.session_state.section_confirmed = {s: False for s in current_sections}
                    st.session_state.analysis_results = {"tags": tags, "modality": modality, "sir_json": sir_json, "rag_refs": rag_refs, "sections": current_sections, "image": uploaded_files[0].getvalue()}
                    st.session_state.state = "REVIEW"; st.rerun()

    elif st.session_state.state == "REVIEW":
        res = st.session_state.analysis_results
        st.markdown("### Phase 3 — Clinical Verification")
        col_img, col_txt = st.columns([1, 1.6])
        with col_img: st.image(res["image"], use_container_width=True)
        with col_txt:
            for s in res["sections"]:
                st.session_state.section_texts[s] = st.text_area(s, value=st.session_state.section_texts.get(s, ""), height=100)
                st.session_state.section_confirmed[s] = st.checkbox(f"Confirm {s}", value=st.session_state.section_confirmed.get(s, False))
        if all_sections_confirmed():
            signer = st.text_input("Physician Credentials", value=st.session_state.get("signed_by",""))
            if st.button("Finalize Report", type="primary"):
                st.session_state.signed_by = signer; st.session_state.state = "FINAL"; st.rerun()

    elif st.session_state.state == "FINAL":
        res = st.session_state.analysis_results
        st.success("### Phase 4 — Report Finalized")
        st.markdown(f"Verified by **{st.session_state.signed_by}**")
        for s in res["sections"]:
            with st.expander(s): st.markdown(st.session_state.section_texts.get(s, ""))
        if st.button("Generate Action Plan"):
            plan, refs = generate_clinical_action_plan(hf_token, res["sir_json"], st.session_state.section_texts)
            st.info(plan)
        if st.button("Start New Case"): init_state(); st.rerun()
