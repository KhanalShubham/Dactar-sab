# 🚀 SpineAI Hosting & Technology Guide

This guide explains the core technologies used in SpineAI and how you can host the application on the web.

---

## 🛠 Core Technologies

To host and maintain this project, you should be familiar with these main components:

### 1. **Language: Python**
- **What it is:** The programming language used for the entire backend and logic.
- **Where to learn:** [Python.org](https://www.python.org/about/gettingstarted/), [Real Python](https://realpython.com/).

### 2. **UI Framework: [Streamlit](https://streamlit.io/)**
- **What it is:** A fast way to build web apps for data science and AI. It handles the buttons, file uploads, and reports you see in the browser.
- **Files:** `app.py`, `.streamlit/config.toml`.
- **Where to learn:** [Streamlit Documentation](https://docs.streamlit.io/).

### 3. **AI Models & Hosting: [Hugging Face](https://huggingface.co/)**
- **What it is:** A platform for sharing and running AI models. 
- **Models Used:** 
  - **BiomedCLIP:** Used for medical image analysis (vision).
  - **Qwen 2.5:** Used for generating the text-based radiology reports.
- **Inference:** The app uses the `Hugging Face Inference API` to run these models in the cloud so you don't need a powerful GPU locally.
- **Where to learn:** [Hugging Face Course](https://huggingface.co/course/).

### 4. **Medical Imaging: [pydicom](https://pydicom.github.io/) & [MONAI](https://monai.io/)**
- **What it is:** Libraries for reading DICOM files (medical MRI scans) and performing medical AI operations.
- **Where to learn:** [pydicom docs](https://pydicom.github.io/pydicom/stable/), [MONAI tutorials](https://github.com/Project-MONAI/tutorials).

### 5. **Environment Management: `.env`**
- **What it is:** Stores sensitive information like your `HF_TOKEN` (API key) so it doesn't get pushed to GitHub.
- **Where to learn:** [python-dotenv docs](https://pypi.org/project/python-dotenv/).

---

## 🌐 How to Host (Deployment)

Here are the three best ways to host SpineAI on the web:

### Option A: Streamlit Community Cloud (Easiest & Free)
Ideal for prototypes and demos.
1. **Push your code** to a private or public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub and select your repository and `app.py`.
4. **Important:** Add your `HF_TOKEN` in the "Secrets" settings on the Streamlit dashboard.

### Option B: Hugging Face Spaces (Best for AI Apps)
Hugging Face provides a dedicated hosting service for Streamlit apps.
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Create a new Space, choose **Streamlit** as the SDK.
3. Upload your files or sync with GitHub.
4. Add your `HF_TOKEN` in the Space "Settings" under **Variables and Secrets**.

### Option C: Traditional Cloud (AWS / DigitalOcean / Render)
For professional use cases requiring more control.
1. Use **Render** or **Railway** for a simpler setup.
2. Use **Docker** to containerize the app.
3. Deploy to a VPS (Virtual Private Server).
4. Run the command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🏗 Setup Requirements

Before hosting, ensure your `requirements.txt` is up to date. You can install all dependencies locally using:
```bash
pip install -r requirements.txt
```

## 🔑 Required Secrets
You will need a **Hugging Face Token** to run the AI features.
1. Create an account on [Hugging Face](https://huggingface.co/).
2. Go to **Settings > Access Tokens**.
3. Create a **Read** token.
4. Set this as `HF_TOKEN` in your hosting provider's secrets/environment variables.
