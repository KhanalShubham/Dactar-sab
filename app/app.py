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
        # 1. Build Management Query (Prioritized Level-by-Level)
        impression = report_texts.get("IMPRESSION", "")
        query_parts = []
        
        # Priority levels: L5-S1, L4-L5, L3-L4, L2-L3, L1-L2
        levels = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]
        for level in levels:
            data = sir_json.get(level, {})
            if isinstance(data, dict):
                level_findings = []
                # Canal Stenosis
                canal = data.get("canal", "").lower()
                if "severe" in canal: level_findings.append(f"severe central stenosis at {level}")
                elif "moderate" in canal: level_findings.append(f"moderate central stenosis at {level}")
                # Foraminal
                foram = data.get("foramina", "").lower()
                if "stenosis" in foram or "narrowing" in foram: level_findings.append(f"{level} foraminal narrowing")
                # Disc
                disc = data.get("disc", "").lower()
                if any(k in disc for k in ["bulge", "protrusion", "extrusion"]): level_findings.append(f"{level} {disc}")
                # Alignment
                align = data.get("alignment", "").lower()
                if "listhesis" in align: level_findings.append(f"{level} listhesis")
                
                if level_findings:
                    query_parts.append(level_findings[0]) # Prioritize first finding per level
        
        if not query_parts:
            query = f"General lumbar spine degenerative disease management Nepal guidelines. Impression: {impression[:200]}"
        else:
            query = ", ".join(query_parts) + ". " + f"Impression: {impression[:100]}"
            
        query += " Provide treatment, referral urgency, medication, patient advice, follow-up according to Nepal BHS STP."
        
        # 2. Retrieve Guidelines (Top 5 for management)
        retriever = get_nepal_retriever()
        results = retriever.retrieve(query, k=5)
        context = retriever.format_for_prompt(results)
        
        # 3. Call LLM with Spec System Prompt
        client = InferenceClient(api_key=hf_token)
        system_msg = """You are a clinical decision support assistant for doctors in Nepal.  
Based on the MRI findings and the Nepal guidelines provided, output a structured action plan with exactly these sections:

1. PROBABLE DIAGNOSIS (from the MRI findings)
2. URGENCY (Routine / Urgent / Emergency) – use emergency only for cauda equina or rapidly progressive deficit
3. FIRST-LINE TREATMENT (non-pharmacological and pharmacological) – include drug names, doses, duration only if found in guidelines
4. REFERRAL (specialty and timing, e.g., orthopaedics, neurosurgery, physiotherapy, pain clinic)
5. PATIENT ADVICE (what to tell the patient, including red flags to return immediately)
6. FOLLOW-UP (when to reassess, e.g., 4 weeks, 6 weeks)

Use the retrieved excerpts as primary source. If a section is not covered by the excerpts, state "Not specified in available Nepal guidelines."  
Keep language professional and concise. Do not mention "AI" or "assistant" in the output. Do not add extra sections."""

        user_msg = f"MRI FINDINGS (SIR and Impression):\n{query}\n\nNEPAL GUIDELINES EXCERPTS:\n{context}\n\nGenerate action plan."

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=800,
            temperature=0.2
        )
        return response.choices[0].message.content, results
    except Exception as e:
        return f"Failed to generate action plan: {e}", []

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION & DATASET UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def export_for_annotation(image_bytes, tags, report_text, patient_id):
    """Saves image and metadata in a format ready for Label Studio/Labelbox."""
    base_dir = "data/annotation_export"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    case_id = f"{patient_id}_{timestamp}"
    
    # Save Image
    img_path = os.path.join(base_dir, f"{case_id}.png")
    with open(img_path, "wb") as f:
        f.write(image_bytes)
        
    # Save Metadata
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
# CLINICAL VISIBILITY & ACCESSIBILITY THEME
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL VISIBILITY & ACCESSIBILITY THEME (WCAG 2.1 AA)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    /* Primary brand colors - Psychological Trust */
    --primary: #1F4E79;        /* Deep Blue (Stability) */
    --primary-dark: #153452;
    --primary-light: #E6F0FA;  /* Tranquil Background */
    --accent: #2C7A7B;         /* Muted Teal (Growth/Health) */
    
    /* Content Backgrounds */
    --bg-main: #F8F9FC;        /* Clean Off-white */
    --bg-sidebar: #FFFFFF;     /* Pure White Sidebar */
    --bg-card: #FFFFFF;
    
    /* High-Contrast Typography */
    --text-heading: #1E293B;   /* Dark Slate */
    --text-body: #334155;      /* Charcoal Charcoal */
    --text-muted: #64748B;
    
    /* Layout Accents */
    --border-standard: #CBD5E1;
    --border-subtle: #E2E8F0;
    
    /* Semantic Feedback */
    --success-bg: #DCFCE7;
    --success-text: #166534;
    --warning-bg: #FEF3C7;
    --warning-text: #92400E;
    --error-bg: #FEE2E2;
    --error-text: #E11D48;
}

/* Application Layout - Forced Visibility */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-body) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Force global text visibility */
.stApp p, .stApp span, .stApp label, .stApp div, .stApp li {
    color: var(--text-body) !important;
}

/* Fixed Solid Sidebar (280px) */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-subtle) !important;
    min-width: 280px !important;
    max-width: 280px !important;
}

[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: var(--text-heading) !important;
    font-weight: 500 !important;
}

/* Headings & Typography */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-heading) !important;
    font-weight: 700 !important;
    line-height: 1.3 !important;
}

h1 { font-size: 2rem !important; margin-bottom: 1rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.25rem !important; }

/* Enhanced Form Elements */
input, textarea, select {
    background-color: #FFFFFF !important;
    color: var(--text-body) !important;
    border: 1.5px solid var(--border-standard) !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
    font-size: 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

input:focus, textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-light) !important;
    outline: none !important;
}

/* Properly Aligned Checkboxes */
div[data-testid="stCheckbox"] label {
    color: var(--text-body) !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding-left: 0.5rem !important;
}

/* Clinical Component Cards */
div[data-testid="stVerticalBlock"] > div.stVerticalBlockBorderWrapper > div {
    background: #FFFFFF !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    margin-bottom: 1.5rem !important;
}

/* Action Buttons */
.stButton>button {
    background-color: var(--primary) !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-weight: 700 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important;
    width: auto !important;
}

.stButton>button:hover {
    background-color: var(--primary-dark) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 15px -3px rgba(31, 78, 121, 0.3) !important;
}

/* Secondary Button Styling */
.stButton>button[kind="secondary"] {
    background-color: #F1F5F9 !important;
    color: var(--text-heading) !important;
}

/* Stepper Indicator */
.stepper-container {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding: 0 1rem;
}

.step {
    flex: 1;
    text-align: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-muted) !important;
    padding-bottom: 0.5rem;
    border-bottom: 4px solid var(--border-subtle);
}

.step.active {
    color: var(--primary) !important;
    border-bottom: 4px solid var(--primary);
}

/* Expander Titles */
[data-testid="stExpander"] summary p {
    color: var(--text-heading) !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

def render_stepper(current_phase):
    phases = ["INTAKE", "UPLOAD", "REVIEW", "FINAL"]
    phase_labels = ["1. Clinical Intake", "2. Study Upload", "3. Specialist Review", "4. Report Finalized"]
    
    cols = st.columns(len(phases))
    for i, (phase, label) in enumerate(zip(phases, phase_labels)):
        is_active = (phase == current_phase)
        active_class = "active" if is_active else ""
        cols[i].markdown(f"""
        <div class="step {active_class}">
            {label}
        </div>
        """, unsafe_allow_html=True)
    st.write("")

def branding_header():
    # Only keep Phase indicator in main area as per request
    render_stepper(st.session_state.state)

branding_header()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 SpineAI Pro")
    st.caption(f"v{config['version']} · Clinical Decision Support")
    
    # Initialize token from session state or env
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = os.getenv("HF_TOKEN", "")
    hf_token = st.session_state.hf_token

    st.markdown("---")

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
    st.markdown("**🛡️ Clinical Governance**")
    use_nepal_rag = st.checkbox("Apply Nepal Medical Guidelines (RAG)", value=True, help="Enriches AI reasoning with local BHS/STP clinical protocols.")

    st.markdown("---")
    st.markdown("**📖 Guideline Knowledge Desk**")
    st.caption("Ask anything about Nepal Medical protocols:")
    q_desk = st.text_input("Protocol Question", placeholder="e.g. Treatment for Pott's spine", label_visibility="collapsed")
    if st.button("Query Guidelines"):
        if q_desk:
            with st.spinner("Searching Nepal guidelines..."):
                ans, refs = answer_guideline_question(hf_token, q_desk)
                st.session_state.last_q_ans = {"q": q_desk, "ans": ans, "refs": refs}
        else:
            st.warning("Please enter a question.")

    if "last_q_ans" in st.session_state:
        with st.expander("📝 Guideline Answer", expanded=True):
            st.markdown(f"**Q: {st.session_state.last_q_ans['q']}**")
            st.markdown(st.session_state.last_q_ans['ans'])
            st.markdown("---")
            st.caption("Sources:")
            for r in st.session_state.last_q_ans['refs']:
                st.caption(f"- {r['metadata']['source']} (Page {r['metadata']['page']})")
        if st.button("Clear Answer"):
            del st.session_state.last_q_ans
            st.rerun()

    st.markdown("---")
    st.markdown("**📊 Dataset Dashboard**")
    if os.path.exists("data/annotation_export"):
        count = len([f for f in os.listdir("data/annotation_export") if f.endswith(".png")])
        st.metric("Images Collected", count)
    
    st.markdown("---")
    with st.expander("🛠️ Developer Console"):
        st.caption("Admin & System Management")
        if st.button("🔄 Re-index Vector DB", help="Scans data/historical_cases/ and rebuilds FAISS index"):
            import subprocess
            try:
                # Updated path for script
                subprocess.run([".venv/Scripts/python.exe", "scripts/index_cases.py", "--dir", "data/historical_cases"], check=True)
                st.success("FAISS Index Rebuilt!")
            except Exception as e:
                st.error(f"Re-indexing failed: {e}")
        
        if st.button("📚 Reload RAG Index", help="Reloads the Nepal Guidelines RAG index from disk"):
            st.cache_resource.clear()
            st.success("RAG Index & Model Reloaded.")
            st.rerun()

        if st.button("🧹 Clear Logs"):
            st.session_state.logs = []
            st.rerun()

    st.markdown("---")
    with st.expander("🔑 Authentication", expanded=False):
        new_token = st.text_input("Hugging Face Token", type="password",
                                  value=st.session_state.hf_token,
                                  placeholder="hf_...")
        if new_token != st.session_state.hf_token:
            st.session_state.hf_token = new_token
            st.rerun()
        hf_token = new_token # Sync local variable

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CLINICAL INTAKE & CES SCREENING
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.state == "INTAKE":
    st.markdown("## Phase 1 — Clinical Governance & Safety")
    
    with st.container(border=True):
        st.markdown("### 🏥 Patient Intake & Safety Screening")
        st.markdown("Mandatory Cauda Equina Syndrome (CES) screening must be completed before AI assistance is enabled.")
        
        st.write("")
        col_form, col_info = st.columns([1.5, 1])
        
        with col_form:
            with st.form("intake_form", border=False):
                st.markdown("#### **Red-Flag Symptom Checklist**")
                st.caption("Identify any acute neurological deficits:")
                
                c1, c2 = st.columns(2)
                with c1:
                    saddle   = st.checkbox("Saddle anesthesia")
                    bladder  = st.checkbox("Urinary retention")
                with c2:
                    bowel    = st.checkbox("Bowel incontinence")
                    weakness = st.checkbox("Bilateral leg weakness")
                
                st.markdown("---")
                st.markdown("#### **Clinical Context**")
                study_type = st.selectbox("Study Protocol", ["Lumbar Spine MRI — Standard T2/T1", "Myelogram Protocol", "Other"])
                clinical_q = st.text_area("Clinical Indication", placeholder="e.g. Radiculopathy, L5 distribution.")
                
                st.write("")
                submitted = st.form_submit_button("Proceed to Imaging Consensus →", use_container_width=True)

        with col_info:
            st.warning("""
            **Safety Protocol 10.4**
            
            SpineAI is optimized for elective diagnostic support. CES is a surgical emergency requiring immediate decompression. 
            
            If red flags are present, the AI reasoning engine will be blocked to ensure immediate referral to a neurosurgical unit.
            """)
            st.image("https://img.icons8.com/fluency/96/shield.png", width=80)

        if submitted:
            if saddle or bladder or bowel or weakness:
                st.session_state.state = "CES_BLOCKED"
                log("CES flags detected — AI deactivated.")
                st.rerun()
            else:
                st.session_state.patient_data["study_type"]  = study_type
                st.session_state.patient_data["clinical_q"]  = clinical_q
                log("CES screening passed.")
                st.session_state.state = "UPLOAD"
                st.rerun()

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
    st.markdown('<p class="phase-indicator">Phase 2 — Imaging Consensus</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("### 🧬 MRI Volume Upload")
        st.caption("Standard image stacks (PNG/JPG) or clinical DICOM (.dcm) files are supported.")
        uploaded_files = st.file_uploader("Select MRI files (Stack/Volume)", type=["png","jpg","jpeg","dcm", "dicom", "ima"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        st.write("")
        col_prev, col_action = st.columns([1.2, 1])
        
        if len(uploaded_files) > 1:
            files_data = [(f.getvalue(), f.name) for f in uploaded_files]
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

        if dcm_meta:
            st.session_state.patient_data["name"] = dcm_meta["name"]
            st.session_state.patient_data["id"] = dcm_meta["id"]
            st.session_state.patient_data["age"] = dcm_meta["age"]
            st.toast("✅ DICOM metadata synced.")

        with col_prev:
            with st.container(border=True):
                st.image(active_image, caption=f"{study_type_label} - Key Slice Preview", use_container_width=True)
                if intensity_profile is not None:
                    st.line_chart(intensity_profile, height=120)
                    st.caption("📏 Volumetric T2 Signal Profile")
        
        with col_action:
            with st.container(border=True):
                st.markdown("#### 📊 Study Insight")
                p = st.session_state.patient_data
                st.markdown(f"""
                - **Patient:** `{p['name'] or 'Unknown'}`
                - **ID:** `{p['id'] or 'Not Provided'}`
                - **Composition:** `{len(uploaded_files)} slices detected`
                """)
                
                st.markdown("---")
                st.info("The 3D Consensus Analysis performs cross-slice verification to eliminate false positives.")

                comp_mode = st.checkbox("⏳ Compare to Prior Study", help="Enables temporal reasoning for interval changes")
                
                st.write("")
                if st.button("🧠 Start 3D AI Consensus Analysis", type="primary", use_container_width=True):
                    with st.spinner("Analyzing slices for volumetric consensus..."):
                        # 1. Detect Modality & Get Tags
                        log("STAGE 1: Volumetric modality detection & tagging...")
                        analyzer = get_analyzer()
                        if len(uploaded_files) > 1:
                            tags = analyzer.analyze_volume_consensus([f.getvalue() for f in uploaded_files])
                        else:
                            tags = analyzer.auto_tag_findings(uploaded_files[0].getvalue())
                        
                        modality = tags[0].get('modality', 'SPINE_MRI') if tags else 'SPINE_MRI'
                        log(f"MODALITY DETECTED: {modality}")
                        
                        # 2. Constraints & Reasoning (Skip or adapt for non-spine)
                        log("STAGE 2: Applying Clinical Constraints...")
                        tags = AnatomicalConstraintEngine.apply_constraints(tags)
                        
                        conflicts = []
                        causality = {}
                        sir_json = {}
                        
                        if modality == "SPINE_MRI":
                            log("STAGE 3: Spine Causal Reasoning...")
                            conflicts = ConsistencyEngine.detect_contradictions(tags)
                            causality = CausalReasoningEngine.derive_causality(tags)
                            sir_json = analyzer.get_sir_map(tags)
                            sir_json["CAUSALITY_MAP"] = causality
                            sir_json = SymbolicCausalValidator.validate_and_refine(sir_json)
                        else:
                            sir_json = {"findings": [t['label'] for t in tags]}

                        # 3. Narrative Generation
                        log("STAGE 4: Multimodal Narrative Generation...")
                        p_info = f"{p['name']}, Age {p['age']}, ID {p['id']}"
                        raw, rag_refs = generate_narrative(hf_token, sir_json, tags, p_info, comparison_mode=comp_mode, use_rag=use_nepal_rag, modality=modality)
                        
                        from config import REPORT_SECTIONS_BY_MODALITY
                        current_sections = REPORT_SECTIONS_BY_MODALITY.get(modality, REPORT_SECTIONS_BY_MODALITY["SPINE_MRI"])
                        
                        sections = parse_report_sections(raw, visual_tags=tags)
                        st.session_state.section_texts     = {s: sections.get(s, "") for s in current_sections}
                        st.session_state.section_confirmed = {s: False for s in current_sections}
                        st.session_state.raw_report        = raw
                        st.session_state.analysis_results  = {
                            "tags": tags, "modality": modality, "sir_json": sir_json, "rag_refs": rag_refs,
                            "sections": current_sections, "image": active_image, "profile": intensity_profile,
                            "is_3d": len(uploaded_files) > 1, "conflicts": conflicts
                        }
                        
                        st.session_state.state = "REVIEW"
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — DOCTOR REVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "REVIEW":
    res = st.session_state.analysis_results
    current_sections = res.get("sections", REPORT_SECTIONS)
    confirmed_count = sum(1 for s in current_sections if st.session_state.section_confirmed.get(s))
    total = len(current_sections)

    st.markdown('<p class="phase-indicator">Phase 3 — Clinical Verification</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        p = st.session_state.patient_data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient", p['name'] or "Unknown")
        c2.metric("Study ID", p['id'] or "Unassigned")
        c3.metric("Review Progress", f"{confirmed_count}/{total}")
        severity = calculate_severity_index(res["tags"])
        c4.metric("Severity Index", f"{severity}/10")

    if res.get("conflicts"):
        for conflict in res["conflicts"]:
            st.error(f"⚠️ **Symbolic Conflict Detected:** {conflict}")

    st.write("")
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        with st.container(border=True):
            st.markdown("#### 🖼️ Image Verification")
            if res.get("is_3d") and "processed_volume" in st.session_state:
                vol = st.session_state.processed_volume
                slice_idx = st.slider("Consensus Volume Navigator", 0, len(vol)-1, len(vol)//2)
                st.image(vol[slice_idx], use_container_width=True)
            else:
                st.image(res["image"], use_container_width=True)
            
            if res.get("profile") is not None:
                st.line_chart(res["profile"], height=100)
                st.caption("📏 T2 Signal Intensity Profile")

            st.markdown("---")
            st.markdown("#### 🏷️ 3D Consensus Tags")
            for tag in res["tags"]:
                label = tag['label']
                mod = tag.get('modifier', 'Possible')
                color = "#166534" if mod == "Definite" else ("#92400E" if mod == "Likely" else "#1E293B")
                bg_color = "#DCFCE7" if mod == "Definite" else ("#FEF3C7" if mod == "Likely" else "#F1F5F9")
                
                st.markdown(f"""
                <div style='background: {bg_color}; border: 1.5px solid {color}33; padding: 12px; border-radius: 12px; margin-bottom: 8px; border-left: 5px solid {color}'>
                    <div style='font-size: 0.75rem; font-weight: 800; color: {color}; text-transform: uppercase; margin-bottom: 2px;'>{mod}</div>
                    <div style='font-size: 0.95rem; font-weight: 600; color: #0F172A;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if res.get("metrics"):
                st.markdown("---")
                st.markdown("#### 📏 Quantitative Evidence")
                cols = st.columns(2)
                metrics = list(res["metrics"].items())
                for i, (k, v) in enumerate(metrics):
                    cols[i % 2].metric(label=k, value=v)

    with col_right:
        st.markdown("#### 📄 Section Verification")
        st.caption("Verify findings and narrative logic for each section.")
        
        for section in REPORT_SECTIONS:
            is_confirmed = st.session_state.section_confirmed.get(section, False)
            color = "#10B981" if is_confirmed else "#F59E0B"
            
            with st.container(border=True):
                st.markdown(f"**{section}**")
                edited = st.text_area(
                    label=section,
                    value=st.session_state.section_texts.get(section, ""),
                    height=130,
                    key=f"edit_{section}",
                    label_visibility="collapsed"
                )
                st.session_state.section_texts[section] = edited
                
                c_check, c_status = st.columns([1, 2])
                with c_check:
                    confirmed = st.checkbox("Confirm Section", value=is_confirmed, key=f"chk_{section}")
                    st.session_state.section_confirmed[section] = confirmed
                with c_status:
                    status_text = "✅ VERIFIED" if confirmed else "🟡 PENDING"
                    st.markdown(f"<p style='color: {color}; font-weight: 700; font-size: 0.8rem; margin-top: 5px;'>{status_text}</p>", unsafe_allow_html=True)

    if all_sections_confirmed():
        st.write("")
        with st.container(border=True):
            st.markdown("#### 🖊️ Finalization")
            col_name, col_btn = st.columns([1.5, 1])
            with col_name:
                signer = st.text_input("Physician Credentials", placeholder="Dr. Name, MD", value=st.session_state.get("signed_by",""))
                st.session_state.signed_by = signer
            with col_btn:
                st.write("")
                if st.button("Finalize & Digitally Sign", type="primary", disabled=not signer.strip(), use_container_width=True):
                    st.session_state.state = "FINAL"
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FINALIZED REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.state == "FINAL":
    res = st.session_state.analysis_results
    st.markdown('<p class="phase-indicator">Phase 4 — Clinical Finalization</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown(f"### ✅ Study Finalized & Signed")
        st.markdown(f"Verified by **{st.session_state.signed_by}** on {datetime.date.today().strftime('%d %B %Y')}")

    col_img, col_rep = st.columns([1, 1.8])
    with col_img:
        st.image(st.session_state.analysis_results["image"], use_container_width=True)
        st.caption("Key Study Slice (Reference Only)")

    with col_rep:
        for section in REPORT_SECTIONS:
            with st.expander(f"📄 {section}", expanded=(section == "IMPRESSION")):
                st.markdown(st.session_state.section_texts.get(section, "").replace("\n","  \n"))

        if "rag_refs" in st.session_state.analysis_results and st.session_state.analysis_results["rag_refs"]:
            with st.expander("📚 Nepal Clinical Guidelines Evidence"):
                for i, ref in enumerate(st.session_state.analysis_results["rag_refs"], 1):
                    st.markdown(f"**[{i}] {ref['metadata']['source']} (Page {ref['metadata']['page']})**")
                    st.caption(ref['text'])

    st.write("")
    with st.container(border=True):
        st.markdown("### 🤖 Clinical Assistant (Action Plan)")
        st.markdown("Generate evidence-based management suggestions based on official Nepal BHS STP protocols.")
        
        c_ap1, c_ap2 = st.columns(2)
        with c_ap1:
            gen_plan = st.button("🚀 Generate Management Action Plan", type="primary", use_container_width=True)
        with c_ap2:
            clear_plan = st.button("🗑️ Reset Assistant", type="secondary", use_container_width=True)

        if gen_plan:
            with st.spinner("Analyzing findings against local protocols..."):
                plan, plan_refs = generate_clinical_action_plan(hf_token, st.session_state.analysis_results["sir_json"], st.session_state.section_texts)
                st.session_state.clinical_action_plan = {"plan": plan, "refs": plan_refs}
        
        if clear_plan:
            if "clinical_action_plan" in st.session_state:
                del st.session_state.clinical_action_plan
                st.rerun()
        
        if "clinical_action_plan" in st.session_state:
            st.write("")
            st.markdown("#### 📋 Personalized Management Plan")
            st.markdown(st.session_state.clinical_action_plan["plan"])
            with st.expander("📚 Decision Sources"):
                for r in st.session_state.clinical_action_plan["refs"]:
                    st.markdown(f"- **{r['metadata']['source']} (Page {r['metadata']['page']})**")
                    st.caption(f"Excerpt: \"{r['text'][:150]}...\"")

    st.write("")
    
    # Build final report text for PDF
    final_text = ""
    for sec in REPORT_SECTIONS:
        final_text += f"{sec}\n{st.session_state.section_texts.get(sec, '')}\n\n"

    export_patient_data = {**st.session_state.patient_data, "signed_by": st.session_state.signed_by}
    exporter = ReportExporter(REPORT_SECTIONS) # Pass REPORT_SECTIONS here
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
            file_name=f"SpineAI_{st.session_state.patient_data['id']}.pdf",
            mime="application/pdf",
            type="primary"
        )
        
        structured_json = json.dumps({
            "patient": st.session_state.patient_data,
            "findings": res["tags"],
            "metrics": res.get("metrics", {}),
            "report_sections": st.session_state.section_texts,
            "rag_context": res.get("rag_refs", []),
            "clinical_assistant": {
                "generated_at": str(datetime.datetime.now()),
                "action_plan": st.session_state.get("clinical_action_plan", {}).get("plan", ""),
                "sources_used": [
                    {"file": r['metadata']['source'], "page": r['metadata']['page'], "excerpt": r['text'][:200]}
                    for r in st.session_state.get("clinical_action_plan", {}).get("refs", [])
                ]
            },
            "signed_by": st.session_state.signed_by,
            "timestamp": str(datetime.datetime.now())
        }, indent=2)
        
        st.download_button(" Download JSON", data=structured_json, file_name=f"SpineAI_{st.session_state.patient_data['id']}.json", mime="application/json")

        if st.button("📤 Export for Annotation"):
            path = export_for_annotation(res["image"], res["tags"], final_text, st.session_state.patient_data['id'])
            st.toast(f"Exported to {path}")

    with col_new:
        if st.button("Start New Case"):
            for k in ["state","section_texts","section_confirmed","raw_report","analysis_results","signed_by"]:
                st.session_state.pop(k, None)
            init_state()
            st.rerun()
