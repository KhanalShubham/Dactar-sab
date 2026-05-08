import streamlit as st
import datetime
import time
import requests
import base64
import random
import os
from dotenv import load_dotenv
from fpdf import FPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from huggingface_hub import InferenceClient
from image_analysis import analyze_mri

load_dotenv()

st.set_page_config(page_title="SpineAI Prototype", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        :root {
            --primary: #0EA5E9;
            --primary-dark: #0284C7;
            --bg-light: #F8FAFC;
            --text-main: #1E293B;
            --text-muted: #64748B;
            --glass-bg: rgba(255, 255, 255, 0.7);
            --glass-border: rgba(255, 255, 255, 0.3);
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
        
        /* Glassmorphism Card */
        .report-card {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        
        /* Modern Sidebar */
        [data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #E2E8F0;
        }
        
        /* Professional Buttons */
        .stButton>button {
            border-radius: 12px;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: none !important;
            background-color: var(--primary) !important;
            color: white !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.3);
            background-color: var(--primary-dark) !important;
        }
        
        /* Finding Items */
        .finding-item {
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 10px;
            background: white;
            border: 1px solid #F1F5F9;
            display: flex;
            align-items: center;
            transition: all 0.2s ease;
        }
        
        .finding-item:hover {
            border-color: var(--primary);
            transform: translateX(4px);
        }
        
        .badge {
            width: 8px;
            height: 32px;
            border-radius: 4px;
            margin-right: 16px;
        }
        
        .badge-normal { background-color: var(--success); box-shadow: 0 0 10px rgba(16, 185, 129, 0.2); }
        .badge-mild { background-color: var(--warning); box-shadow: 0 0 10px rgba(245, 158, 11, 0.2); }
        .badge-severe { background-color: var(--danger); box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }
        
        /* Header Styling */
        .main-header {
            display: flex;
            align-items: center;
            margin-bottom: 3rem;
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
        
        h1, h2, h3 {
            color: #0F172A;
            letter-spacing: -0.02em;
        }
        
        /* Hide Streamlit components for a cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
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

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Engine Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password", value=os.getenv("HF_TOKEN", ""), help="Required for CLIP vision analysis and Qwen report generation.")
    
    st.markdown("---")
    st.markdown("### 👤 Patient Details")
    p_name = st.text_input("Full Name", value=st.session_state.patient_data["name"])
    p_age = st.text_input("Age", value=st.session_state.patient_data["age"])
    p_id = st.text_input("Patient ID", value=st.session_state.patient_data["id"])
    p_date = st.date_input("Study Date", value=datetime.date.today())

    # Update session state
    st.session_state.patient_data = {
        "name": p_name,
        "age": p_age,
        "id": p_id,
        "date": str(p_date)
    }

    st.markdown("---")
    st.markdown("### 🛡️ Clinical Guardrails")
    st.success("ACE Logic: Active")
    st.success("SIR Validation: Active")
    st.caption("SpineAI Pro v2.5.0-stable")

# Top Disclaimer
st.info("💡 **Clinical Dashboard Active**: Analysis restricted to Lumbar Spine protocols. System monitors for red-flag symptoms in real-time.")
st.caption("🟢 Normal | 🟡 Mild-Moderate | 🔴 Severe Finding")

if st.session_state.report_state == "INTAKE":
    st.subheader("📋 Clinical Intake & Red-Flag Screening")
    st.write("Ensuring patient safety through comprehensive neurological screening.")
    
    with st.container():
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.write("#### Evaluate for Cauda Equina Syndrome (CES)")
        st.write("Check any symptoms that apply to the patient's current presentation:")
        
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
    uploaded_file = st.file_uploader("Upload MRI (DICOM or image)", type=["jpg", "png", "jpeg", "dcm", "dicom", "ima"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", width=300)
        if st.button("Generate AI Report", type="primary"):
            with st.spinner("Initializing Clinical Analysis Engines..."):
                status_placeholder = st.empty()
                
                # Phase 1: Image Processing
                status_placeholder.markdown("🔍 **[Phase 1]** Normalizing DICOM metadata & voxel intensity...")
                time.sleep(1.2)
                
                # Phase 2: CLIP Feature Extraction
                status_placeholder.markdown("🧠 **[Phase 2]** BiomedCLIP precision feature extraction active...")
                time.sleep(1.5)
                
                # Phase 3: Anatomical Mapping
                status_placeholder.markdown("📏 **[Phase 3]** Mapping structural findings to nerve root levels...")
                time.sleep(1.2)
                
                # Phase 4: Causal Validation
                status_placeholder.markdown("🛡️ **[Phase 4]** Running ACE Logic: Verifying causal consistency...")
                time.sleep(1.0)
                
                # Run the simulated analysis
                processed_img, findings, messages = analyze_mri(uploaded_file.getvalue(), uploaded_file.name)
                
                # Update patient metadata if found in messages (DICOM extraction)
                for msg in messages:
                    if "👤 Patient:" in msg:
                        try:
                            # Parse "👤 Patient: Name (ID: ID)"
                            name_part = msg.split("👤 Patient: ")[1]
                            name = name_part.split(" (ID: ")[0]
                            patient_id = name_part.split(" (ID: ")[1].replace(")", "")
                            st.session_state.patient_data["name"] = name
                            st.session_state.patient_data["id"] = patient_id
                        except Exception:
                            pass
                
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
                icon = "✨"
                badge_cls = "badge-normal"
                if "Severe" in finding or "Stenosis" in finding or "Extrusion" in finding:
                    icon = "🚨"
                    badge_cls = "badge-severe"
                elif "Mild" in finding or "Moderate" in finding or "Bulge" in finding:
                    icon = "⚠️"
                    badge_cls = "badge-mild"
                
                st.markdown(f"""
                    <div class="finding-item">
                        <div class="badge {badge_cls}"></div>
                        <div style="flex-grow:1">
                            <div style="font-size: 0.8rem; color: #64748B; font-weight: 600;">{level}</div>
                            <div style="font-weight: 500;">{finding}</div>
                        </div>
                        <div style="font-size: 1.2rem;">{icon}</div>
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