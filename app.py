import streamlit as st
import datetime
import time
import requests
import base64
import random
from fpdf import FPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from huggingface_hub import InferenceClient
from image_analysis import analyze_mri

st.set_page_config(page_title="SpineAI Prototype", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background-color: #f8f9fa;
        }
        
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .report-card {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            color: #1a1a1a;
        }
        
        .finding-item {
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 4px;
            background-color: #f1f3f5;
            display: flex;
            align-items: center;
        }
        
        .finding-normal { border-left: 4px solid #40c057; }
        .finding-mild { border-left: 4px solid #fab005; }
        .finding-severe { border-left: 4px solid #fa5252; }
        
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 700 !important;
        }
        
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()
if "report_state" not in st.session_state:
    st.session_state.report_state = "INTAKE" # States: INTAKE, UPLOAD, DRAFT, FINAL, CES_BLOCKED
if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "findings" not in st.session_state:
    st.session_state.findings = {}
if "analysis_messages" not in st.session_state:
    st.session_state.analysis_messages = []
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {"name": "", "age": "", "id": "", "date": str(datetime.date.today())}

def generate_ai_report(api_key, findings_dict):
    if not api_key:
        st.error("Please enter your Hugging Face API Token in the sidebar to generate a report.")
        return None
        
    try:
        hf_token = os.getenv("HF_TOKEN", "")
        client = InferenceClient(api_key=api_key)
        
        system_prompt = """You are an expert radiologist.

Generate a lumbar spine MRI report using:
- Conservative clinical language
- Avoid definitive diagnosis
- Use terms like "suggestive of", "may indicate"
- Include:
  - FINDINGS
  - IMPRESSION
  - Clinical correlation note

Keep it professional and realistic. Output only the report text, no extra commentary."""
        
        findings_str = "\n".join([f"- {k}: {v}" for k, v in findings_dict.items()])
        patient_info = f"Patient: {st.session_state.patient_data['name']}, Age: {st.session_state.patient_data['age']}, ID: {st.session_state.patient_data['id']}"
        
        user_prompt = f"""{patient_info}
Based on these extracted image features, write a very detailed report:
{findings_str}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat_completion(
            messages=messages, 
            model="Qwen/Qwen2.5-7B-Instruct", 
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"Hugging Face API Error: {str(e)}")
        return None

def add_confidence(findings):
    updated = {}
    for level, result in findings.items():
        confidence = random.randint(65, 90)
        updated[level] = f"{result} (Confidence: {confidence}%)"
    return updated

def generate_pdf_reportlab(text, filename="SpineAI_Report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("SpineAI Radiology Report", styles["Title"]))
    story.append(Spacer(1, 12))
    
    # Patient Details Section
    p_data = st.session_state.patient_data
    patient_info = f"<b>Patient:</b> {p_data['name']} | <b>Age:</b> {p_data['age']} | <b>ID:</b> {p_data['id']} | <b>Date:</b> {p_data['date']}"
    story.append(Paragraph(patient_info, styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<hr/>", styles["Normal"]))
    story.append(Spacer(1, 12))
    
    # Process text for reportlab (basic line breaks)
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    return filename

def generate_pdf(report_text, status):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.set_fill_color(240, 242, 246)
    pdf.rect(0, 0, 210, 40, 'F')
    
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(190, 15, txt="SPINE AI MEDICAL", ln=True, align='R')
    pdf.set_font("Arial", size=10)
    pdf.cell(190, 5, txt="Advanced Neuroradiology Diagnostics", ln=True, align='R')
    pdf.ln(10)
    
    # Reset text color
    pdf.set_text_color(0, 0, 0)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Clinical Radiology Report", ln=True, align='L')
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, 52, 200, 52)
    pdf.ln(5)
    
    # Status and Date
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(100, 8, txt=f"REPORT STATUS: {status}", ln=0)
    pdf.cell(90, 8, txt=f"DATE: {datetime.date.today().strftime('%Y-%m-%d')}", ln=1, align='R')
    pdf.ln(5)
    
    # Content
    pdf.set_font("Arial", size=12)
    for line in report_text.split('\n'):
        # Sanitize markdown and unsupported unicode chars for default FPDF font
        clean_line = line.replace('•', '-').replace('**', '')
        clean_line = clean_line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=clean_line)
    
    pdf.ln(10)
    
    # Signatures
    if status == "FINAL":
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Electronically signed by:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Dr. Jane Doe, MD", ln=True)
        pdf.cell(200, 10, txt="Attending Radiologist", ln=True)
    
    # Disclaimer
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="DISCLAIMER: This report was initially drafted by an Artificial Intelligence system. It has been reviewed and finalized by a qualified radiologist.")
    
    return pdf.output(dest='S').encode('latin-1')

st.title("SpineAI Prototype")

st.sidebar.header("Configuration")
st.sidebar.write("Get your free token at [huggingface.co](https://huggingface.co/settings/tokens)")
hf_token = st.sidebar.text_input("Hugging Face API Token", type="password", value=os.getenv("HF_TOKEN", ""))

st.sidebar.markdown("---")
st.sidebar.header("👤 Patient Details")
p_name = st.sidebar.text_input("Full Name", value=st.session_state.patient_data["name"])
p_age = st.sidebar.text_input("Age", value=st.session_state.patient_data["age"])
p_id = st.sidebar.text_input("Patient ID", value=st.session_state.patient_data["id"])
p_date = st.sidebar.date_input("Study Date", value=datetime.date.today())

# Update session state
st.session_state.patient_data = {
    "name": p_name,
    "age": p_age,
    "id": p_id,
    "date": str(p_date)
}

st.sidebar.markdown("---")
st.sidebar.header("About SpineAI")
st.sidebar.info("""
**Advanced MRI Analysis Engine**
- **Segmentation:** AI-driven vertebral mapping.
- **Classification:** Automated disc health grading.
- **Natural Language:** GPT-powered reporting.
""")
st.sidebar.caption("Prototype v2.0 | Clinical demo only")

# Top Disclaimer
st.warning("⚠️ **AI-generated draft.** Not for diagnostic use. Requires radiologist review. Explicitly excludes Cauda Equina Syndrome (CES).")
st.markdown("🟢 Normal | 🟡 Mild | 🟠 Moderate | 🔴 Severe")
st.caption("Simulated AI detection for demonstration purposes")

if st.session_state.report_state == "INTAKE":
    st.header("📋 Clinical Intake & Screening")
    st.write("Ensuring patient safety through comprehensive red-flag screening.")
    
    with st.container():
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.write("**Evaluate the patient for Cauda Equina Syndrome (CES):**")
        
        ces_saddle = st.checkbox("Saddle anesthesia (loss of sensation in buttocks/perineum)")
        ces_bladder = st.checkbox("New onset of bladder or bowel dysfunction")
        ces_weakness = st.checkbox("Rapidly progressive lower extremity weakness")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("No red-flag symptoms present. Proceed to Upload", type="primary"):
            if ces_saddle or ces_bladder or ces_weakness:
                st.error("You have checked red-flag symptoms but clicked 'No red-flag symptoms present'. Please verify.")
            else:
                st.session_state.report_state = "UPLOAD"
                st.rerun()
    with col2:
        if st.button("One or more red-flag symptoms present", type="secondary"):
            st.session_state.report_state = "CES_BLOCKED"
            st.rerun()

elif st.session_state.report_state == "CES_BLOCKED":
    st.error("🚨 **RED FLAG DETECTED: Potential Cauda Equina Syndrome**")
    st.error("**Report generation blocked. SpineAI is not designed to assist with emergency CES.**")
    st.error("**Refer the patient to neurosurgery immediately.**")
    if st.button("Start New Patient"):
        st.session_state.report_state = "INTAKE"
        st.rerun()

elif st.session_state.report_state == "UPLOAD":
    st.write("Upload a lumbar spine MRI scan to generate a draft report.")
    uploaded_file = st.file_uploader("Upload MRI (DICOM or image)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", width=300)
        if st.button("Generate AI Report", type="primary"):
            with st.spinner("Performing deep image evaluation & generating detailed report..."):
                # Real-looking processing steps
                status_placeholder = st.empty()
                status_placeholder.text("🔍 Segmenting vertebral bodies...")
                time.sleep(1)
                status_placeholder.text("📏 Measuring disc heights...")
                time.sleep(1)
                status_placeholder.text("🧠 Detecting neural foraminal narrowing...")
                time.sleep(1)
                
                # Run the simulated analysis
                processed_img, findings, messages = analyze_mri(uploaded_file.getvalue())
                
                # Add confidence scores
                findings_with_conf = add_confidence(findings)
                
                generated_draft = generate_ai_report(hf_token, findings_with_conf)
                if generated_draft:
                    st.session_state.draft_text = generated_draft
                    st.session_state.uploaded_image = processed_img
                    st.session_state.findings = findings_with_conf
                    st.session_state.analysis_messages = messages
                    st.session_state.report_state = "DRAFT"
                    st.rerun()
            
    if st.button("Cancel & Return to Intake"):
        st.session_state.report_state = "INTAKE"
        st.rerun()

elif st.session_state.report_state == "DRAFT":
    st.warning("⚠️ **PRELIMINARY REPORT** - AI-generated draft. Requires radiologist review.")
    
    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.markdown("### Source Image Analysis")
        if st.session_state.uploaded_image:
            try:
                st.image(st.session_state.uploaded_image, use_container_width=True)
            except Exception:
                st.info("Image format not supported for direct preview.")
            
            st.markdown("### 🤖 AI Analysis Findings")
            
            for level, finding in st.session_state.findings.items():
                icon = "✅"
                cls = "finding-normal"
                if "Severe" in finding or "Stenosis" in finding:
                    icon = "🚨"
                    cls = "finding-severe"
                elif "Mild" in finding or "Moderate" in finding or "Bulge" in finding:
                    icon = "⚠️"
                    cls = "finding-mild"
                
                st.markdown(f"""
                    <div class="finding-item {cls}">
                        <span style="margin-right:10px">{icon}</span>
                        <strong>{level}:</strong> &nbsp; {finding}
                    </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.analysis_messages:
                st.info("💡 **AI Insight:** " + " | ".join(st.session_state.analysis_messages))
            
    with col_text:
        st.markdown("### 📝 Draft Radiology Report")
        edited_text = st.text_area("Review and edit the AI-generated draft:", value=st.session_state.draft_text, height=600)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Sign & Finalize", type="primary"):
            st.session_state.draft_text = edited_text
            st.session_state.report_state = "FINAL"
            st.rerun()
    with col2:
        if st.button("Cancel & Discard"):
            st.session_state.report_state = "UPLOAD"
            st.session_state.draft_text = ""
            st.session_state.uploaded_image = None
            st.rerun()

elif st.session_state.report_state == "FINAL":
    st.success("✅ **FINAL REPORT** - Signed and Finalized.")
    
    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.markdown("### Reference Image")
        if st.session_state.uploaded_image:
            try:
                st.image(st.session_state.uploaded_image, use_container_width=True)
            except Exception:
                st.info("Image format not supported for direct preview.")
                
    with col_text:
        st.markdown("### 📄 Final Radiology Report")
        
        # Display Patient Header in UI
        p_data = st.session_state.patient_data
        st.markdown(f"""
            <div style="background-color: #f1f3f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #2c3e50;">
                <strong>PATIENT:</strong> {p_data['name']} | <strong>AGE:</strong> {p_data['age']} | <strong>ID:</strong> {p_data['id']} <br>
                <strong>DATE:</strong> {p_data['date']}
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<div class='report-card'>{st.session_state.draft_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.write(f"**Electronically signed by:** Dr. Jane Doe, MD")
        st.write(f"**Date:** {datetime.date.today().strftime('%B %d, %Y')}")
    
    # Generate PDF using reportlab as requested
    pdf_path = generate_pdf_reportlab(st.session_state.draft_text)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Final Report",
                data=f,
                file_name="SpineAI_Report.pdf",
                mime="application/pdf",
                type="primary"
            )
    with col2:
        if st.button("Start New Report"):
            st.session_state.report_state = "INTAKE"
            st.session_state.draft_text = ""
            st.session_state.uploaded_image = None
            st.rerun()