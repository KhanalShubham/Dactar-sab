# BLUEPRINT: Nepal National Medical AI Infrastructure (NepalMed AI)

## 1. Executive Summary
This document serves as the master technical and clinical architecture for **NepalMed AI**, a transformative national-scale medical intelligence system. Designed as a "Living Intelligence," NepalMed AI is engineered to bridge the massive specialist gap in rural Nepal by digitizing and distributing the collective clinical expertise of the nation's top medical centers.

Unlike general-purpose AI (ChatGPT, Claude), which lacks local context, NepalMed AI is built on a "Radiologist-in-the-Loop" and "Physician-as-the-Trainer" philosophy. It leverages the technical foundations of the **SpineAI Pro** engine—including zero-shot vision analysis, causal reasoning, and safety-gate architectures—to create a unified diagnostic and decision-support backbone for the entire Ministry of Health & Population (MoHP) network.


## 2. The Vision: A Living Intelligence for Nepal
The core objective is to create a system that doesn't just "know" medicine, but "understands" Nepal. 

### 2.1 The "Contextual Differential"
In traditional medical AI, a symptom like "fever with joint pain" might suggest a wide range of global diseases. NepalMed AI applies **Geo-Temporal Bayesian Filtering**:
- **Geographic Layer**: If the patient is in the Terai belt during monsoon season, the system prioritizes Dengue and Malaria.
- **Altitude Layer**: If the patient is in the Himalayan region (e.g., Solu-Khumbu), it prioritizes high-altitude pulmonary/cerebral edema (HAPE/HACE).
- **Resource Layer**: If the patient is at a Health Post with no laboratory, it suggests physical exam-based screening rather than unavailable blood tests.

---

## 3. Detailed Technical Architecture (AWS-Cloud & Edge)
To handle national-scale data securely, the system utilizes a multi-layered AWS architecture.

### 3.1 Data Ingestion & Storage Layer
- **AWS S3 (The Knowledge Lake)**: 
    - **Raw Zone**: Stores incoming DICOM imaging, scanned paper records (via OCR), and raw chat logs.
    - **Clean Zone**: Stores anonymized, structured JSON data ready for RAG and fine-tuning.
    - **Archive Zone (Glacier)**: Long-term storage of medical records for legal compliance (10+ years).
- **AWS IoT Greengrass (The Hospital Bridge)**: 
    - Deployed on-premises at major hospitals (TUTH, Bir, BPKIHS) to act as a secure gateway for DICOM (PACS) data, ensuring high-speed local processing before anonymized metadata is synced to the cloud.

### 3.2 The "Scrubber" Pipeline (Data Anonymization)
A mandatory AWS Lambda-based pipeline ensures no Personally Identifiable Information (PII) ever reaches the training models:
1. **DICOM Header Scrubbing**: Removes Patient Name, ID, and Accession Number from imaging tags.
2. **NLP PII Recognition**: Uses AWS Comprehend Medical to identify and redact names/locations from unstructured doctor notes.
3. **Synthetic ID Generation**: Replaces patient identities with unique hashes to allow longitudinal tracking (tracking a patient's recovery over time) without knowing their identity.

### 3.3 The Med-RAG (Retrieval-Augmented Generation) Engine
To ensure 100% clinical accuracy, the LLM (Qwen/Llama) is augmented by a **Medical Retrieval System**:
- **Vector Database**: Qdrant or FAISS clusters storing embeddings of every MoH Nepal guideline, NHRC research paper, and WHO Nepal country report.
- **The "Truth Check" Algorithm**: Before generating a response, the system retrieves the top 5 most relevant "Clinical Truths" from the vector DB. If the LLM's proposed answer contradicts these truths, the system automatically corrects the output and flags it for human review.

---

## 4. The Self-Learning Engine: RLHF & Active Learning
The soul of NepalMed AI is its ability to learn from the doctors who use it. This creates a "Data Flywheel."

### 4.1 Reinforced Learning from Human Feedback (RLHF)
- **The "Binary Signal"**: Every diagnostic suggestion includes a "Helpful / Not Helpful" toggle for clinicians.
- **The "Corrective Signal"**: If a doctor edits a report section (e.g., changing "Normal marrow signal" to "Modic Type 1 changes"), the system treats the *edited* version as the absolute ground truth.
- **Weighting by Expertise**: Corrections from a senior consultant at a teaching hospital carry more "Training Weight" than those from a junior resident.

### 4.2 Active Learning & Dataset Curation
The system identifies "Low Confidence" cases (e.g., an unusual spinal tumor it hasn't seen often). It automatically flags these for a "National Consensus Panel"—a group of volunteer doctors who review the case. Once they agree on a diagnosis, that case becomes a "Gold Standard" training example.

---

## 5. Specialty Modules: Beyond Radiology
While starting with SpineAI (Radiology), the architecture expands into several key medical domains:

### 5.1 The OPD Triage Module
- **Goal**: Reduce the burden on the 2000+ patients waiting daily at Bir Hospital's OPD.
- **Function**: Patients interact with a mobile/kiosk-based AI in Nepali. It collects history, suggests initial lab tests (CBC, LFT, RFT), and triages them based on urgency. When the patient finally sees the doctor, the doctor already has a structured "AI-Pre-Summary" ready.

### 5.2 The Emergency Stabilization Module
- **Goal**: Save lives in the "Golden Hour" in rural areas.
- **Function**: If the AI detects symptoms of a Heart Attack, Stroke, or Sepsis, it immediately triggers the "Safety Interlock."
- **Action**: Provides step-by-step stabilization (e.g., "Give Aspirin 300mg," "Start IV fluids at X rate") and uses GPS to find the nearest hospital with an ICU/CCU bed available.

### 5.3 The Community Health (FCHV) Module
- **Goal**: Empower the 50,000+ Female Community Health Volunteers.
- **Function**: A simple, voice-based interface in local languages. FCHVs can speak symptoms, and the system provides guidance on maternal health, child nutrition, and vaccination schedules.

---

## 6. Language & Localization Strategy
Nepal's linguistic diversity is a major hurdle for global AI.
- **Medical NLP Alignment**: Building a mapping between colloquial Nepali (e.g., "Mero pet poleko chha") and clinical terms ("Gastroesophageal Reflux" or "Gastritis").
- **Multi-Dialect Support**: Fine-tuning the model on Maithili, Bhojpuri, and Newari clinical datasets to ensure inclusivity for Terai and Valley populations.
- **Voice-to-Text**: Utilizing custom-trained Whisper models for Nepali to allow doctors to dictate notes hands-free in noisy hospital environments.

---

## 7. Governance, Ethics & Sovereignty
This is a national asset and must be protected as such.

### 7.1 Data Sovereignty
- All **Identifiable Data** must reside on servers physically located in Nepal (e.g., Government Integrated Data Center).
- Only **Anonymized Embeddings** are sent to global cloud providers (AWS/GCP) for high-compute training.

### 7.2 The Ethics Board
A permanent committee involving:
- **Nepal Health Research Council (NHRC)**: For ethical oversight.
- **Nepal Medical Council (NMC)**: For licensing and clinical accountability.
- **MoHP Digital Health Division**: For national strategic alignment.

---

## 8. Implementation Roadmap (Phases 1-4)

### Phase 1: The Foundation (Months 1-8)
- **Infrastructure**: Setup AWS landing zones and S3 Knowledge Lake.
- **Data**: Scrape 10,000+ pages of Nepal medical history and protocols.
- **MVP**: Launch "NepalMed RAG" (A search engine for doctors that knows every Nepal-specific guideline).

### Phase 2: The Specialist Expansion (Months 9-18)
- **Imaging**: Integrate SpineAI Pro into 5 major hospitals.
- **Feedback**: Launch the "Radiologist-in-the-Loop" training pipeline.
- **Language**: Release the first fully-functional Nepali Medical LLM (Fine-tuned BioMistral).

### Phase 3: The National Triage (Months 19-30)
- **OPD Launch**: Pilot kiosk-based triage at Bir Hospital and TUTH.
- **FCHV Integration**: Distribute the mobile assistant to 1,000 health volunteers.
- **Emergency Gate**: Deploy the national emergency stabilization protocol AI.

### Phase 4: The Autonomous Evolution (Months 31-48)
- **Self-Correction**: The system begins to detect new disease patterns (outbreaks) in real-time.
- **Global Leadership**: Nepal presents the "NepalMed AI Model" to the WHO as the gold standard for medical AI in developing nations.

---

## 9. Enhanced Data Resource Catalog
To bootstrap the intelligence of the system, the following public/free resources are prioritized for immediate ingestion into the "Knowledge Lake."

### 9.1 Tier 1: Government Primary Sources
- **Ministry of Health & Population (MoHP)**: [mohp.gov.np/publications](https://mohp.gov.np/category/publications/)
    - National Treatment Guidelines (TB, Malaria, Dengue).
    - HMIS Annual Reports (disease burden by district).
    - COVID-19 Nepal Management Guidelines.
- **Nepal Health Research Council (NHRC)**: [nhrc.gov.np/publications](https://nhrc.gov.np/publications/)
    - 500+ free research PDFs.
    - Ethics guidelines for clinical AI.
- **Epidemiology & Disease Control Division (EDCD)**: [edcd.gov.np/publications](https://edcd.gov.np/publications/)
    - Kala-azar surveillance and enteric fever reports.

### 9.2 Tier 2: Nepal Medical Literature (10,000+ Papers)
- **JNMA (Journal of Nepal Medical Association)**: [jnma.com.np](https://www.jnma.com.np/jnma/index.php/jnma/issue/archive) (Free full archive).
- **KUMJ (Kathmandu University Medical Journal)**: [kumj.com.np](https://www.kumj.com.np/issue/browse).
- **JNHRC (Journal of Nepal Health Research Council)**: [nepjol.info/index.php/JNHRC](https://nepjol.info/index.php/JNHRC/issue/archive).
- **PubMed Strategy**: `(Nepal[affiliation]) AND (clinical[title] OR case[title])` — yields 12,000+ targeted results.

### 9.3 Tier 3: Population Health Goldmines
- **Nepal Demographic Health Survey (NDHS) 2022**: [dhsprogram.com](https://dhsprogram.com/pubs/pdf/FR336/FR336.pdf).
- **WHO Nepal Country Office**: [who.int/nepal/publications](https://www.who.int/nepal/publications).

---

## 10. Technical Implementation Priority (Week 1)
The deployment sequence follows a high-impact, low-cost path:

- **Days 1-2: S3 Knowledge Lake Setup**
    - `nepalmed-raw/`: For DICOM, PDFs, and chat logs.
    - `nepalmed-clean/`: For anonymized JSONL.
    - `nepalmed-vectors/`: For FAISS/Qdrant embeddings.
- **Days 3-4: Lambda Anonymization Pipeline**
    - Implementation of PyDICOM scrubbers and NLP-based PII redaction.
- **Days 5-7: Med-RAG Prototype**
    - Deployment of Ollama (Llama 3.2) and ChromaDB for local, zero-cost RAG testing.

---

## 11. RLHF Training Pipeline (8-Week Cycle)
The system operates on a continuous improvement cycle:
1. **Weeks 1-4 (Collect)**: Capture "Helpful/Not Helpful" signals and report edits.
2. **Week 5 (Fine-tune)**: Use QLoRA on Kaggle T4 GPUs (Free) to update models.
3. **Week 6 (Validate)**: Test against safety gates (CES) and referral logic.
4. **Week 7 (A/B Test)**: Compare performance against previous versions with senior clinicians.
5. **Week 8 (Deploy)**: Push new weights to production.

---

## 12. Critical Success Metrics (KPIs)
| Metric | Target |
| :--- | :--- |
| **Clinical Accuracy** | >92% on JNMA case report benchmarks |
| **Doctor Time Saved** | >3 minutes per OPD case summary |
| **Nepali Language Comprehension** | >95% (BLEU score) |
| **Emergency Sensitivity** | 100% (Zero missed life-threatening cases) |
| **FCHV Uptake** | 1,000+ volunteers in 6 months |

---

## 13. Conclusion: The "Flywheel" of National Health
NepalMed AI is not just a tool; it is a **force multiplier**. By combining Nepal's clinical wisdom with world-class cloud architecture and a self-evolving AI engine, we can ensure that a child in Humla receives the same level of diagnostic excellence as a patient in Kathmandu. 

The foundations are laid, the architecture is ready, and the mission is clear: **One National Brain for the Health of All Nepali People.**
