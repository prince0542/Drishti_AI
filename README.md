# 👁️ DRISHTI AI — Offline Accessibility Assistant
> **"See the World Through AI"**  
> An offline-first, privacy-preserving AI accessibility assistant designed to empower people with visual impairments to understand their physical surroundings, read signs and documents in English and Hindi, and navigate life with auditory feedback.

---

## 🌟 Table of Contents
- [1. Project Vision & Mission](#1-project-vision--mission)
- [2. System Architecture](#2-system-architecture)
- [3. Key Features](#3-key-features)
- [4. Safety & Navigation Disclaimer](#4-safety--navigation-disclaimer)
- [5. Technology Stack](#5-technology-stack)
- [6. Prerequisites & Dependencies](#6-prerequisites--dependencies)
- [7. Step-by-Step Windows Installation Guide](#7-step-by-step-windows-installation-guide)
- [8. Running the Application](#8-running-the-application)
- [9. Testing with Sample Images](#9-testing-with-sample-images)
- [10. Troubleshooting & Diagnostics](#10-troubleshooting--diagnostics)
- [11. Future Improvement Roadmap](#11-future-improvement-roadmap)

---

## 1. Project Vision & Mission

Visual impairment affects millions worldwide, creating obstacles in daily tasks such as recognizing obstacles, navigating indoor corridors, reading medicine labels, and deciphering street signs.

While several cloud-based visual assistant applications exist, they require high-speed internet connections, introduce network latency, and present major privacy risks by transmitting intimate personal surroundings to remote servers.

**Drishti AI** (दृष्टि meaning *Vision*) operates **100% offline**:
- 🔒 **Zero Cloud / Zero Telemetry:** Images and camera streams never leave your local machine.
- ⚡ **Local Multimodal AI:** Gemma 4 / Ollama runs locally on your CPU or GPU.
- 🇮🇳 **Bilingual Support:** Understands and speaks in both **English** and **Hindi (हिन्दी)**.
- 🔊 **Offline Text-to-Speech:** Real-time auditory synthesis via Windows SAPI5 (`pyttsx3`).
- ♿ **Accessible UI/UX:** High-contrast color palette, large typography, touch targets, and keyboard friendliness.

---

## 2. System Architecture

```text
Drishti-AI/
│
├── app.py                     # Main Streamlit accessibility UI & section routing
├── live_assistant.py          # Real-time background live scanner, deduplication & voice commands
├── vision.py                  # Local Ollama /api/chat vision pipeline & prompts
├── ocr_reader.py              # Tesseract + RapidOCR engine & image preprocessing
├── speech.py                  # In-browser Web Speech & offline pyttsx3 speech synthesis
├── camera.py                  # Camera capture (Streamlit WebRTC & OpenCV DirectShow)
├── config.py                  # Configuration, safety disclaimers & scan intervals
├── sample_images_generator.py # Auto-generator for accessibility test images
├── requirements.txt           # Python library dependencies
├── README.md                  # Comprehensive documentation & setup manual
├── .gitignore                 # Version control exclusion rules
├── sample_images/             # Pre-built test scenes, signs & medicine labels
└── assets/                    # Application logos and visual assets
```

---

## 3. Key Features

### A. 🟢 Real-Time Continuous Live Camera Scanning
- Automatically observes physical surroundings without manual clicking.
- Periodic frame sampling (adjustable 1.5s–5.0s, default 2.5s) optimized for CPU and 8 GB RAM.
- Non-blocking asynchronous background thread (`LiveScanWorker`) keeps the UI 100% fluid.
- Real-time OCR text fusion (signboards, labels, documents) alongside vision scene descriptions.

### B. 🎙️ Hands-Free Voice Commands (Full-Duplex)
- Control the assistant completely through voice without touching the keyboard:
  - **"Drishti, start"** / **"Start scanning"** / **"Start assistant"** ➔ Activates continuous camera scanning.
  - **"Drishti, stop"** / **"Stop"** / **"Stop speaking"** ➔ Immediately halts scanning and cancels active speech.
  - **"Resume"** / **"Resume scanning"** ➔ Resumes live scanning.
- Full-duplex interaction: can hear "Stop" while audio is actively speaking!

### C. 🧠 Smart Announcement System (Anti-Repetition)
- **Duplicate-Description Filtering:** SequenceMatcher string comparison prevents describing the same object repeatedly.
- **Cooldown Window (18 seconds):** Suppresses repeating static objects while prioritizing newly detected items.
- **Object Tag Extraction:** Identifies and displays active objects (`Bottle`, `Person`, `Chair`, `Desk`, etc.) as accessible pill badges.

### D. 📷 AI Scene Description (Snapshot Mode)
- Upload JPG, PNG, WEBP, or BMP images, or take single photo snapshots.
- Describe objects, spatial layout (left, right, center, foreground, background), colors, and lighting.
- Tailored prompts for obstacle detection, ground elevation changes, and people identification.

### E. 📖 Bilingual OCR Reader (RapidOCR + Tesseract)
- Extracts printed and digital text from medicine boxes, warning signs, menus, and notices.
- Support for **English (`eng`)**, **Hindi (`hin`)**, or combined **(`eng+hin`)**.
- Built-in OpenCV preprocessing pipeline (bilateral filtering, adaptive thresholding, and contrast CLAHE).
- Word count and OCR confidence scoring.

### F. 🔊 Dual-Engine Text-to-Speech
- Browser native Web Speech API (`SpeechSynthesis`) provides zero-latency speaker playback.
- Offline `pyttsx3` SAPI5 voice fallback with WAV caching.
- Playback controls: **Speak Again**, **Stop**, and **Repeat Last Response**.
- Direct guidance for enabling official Hindi voices in Windows.

### D. 🎥 Real-Time Camera Assistant
- WebRTC camera capture through your browser.
- Direct hardware capture via OpenCV (`cv2.VideoCapture`).
- Instant one-click scene description and text reading from camera snapshots.

### E. ♿ Accessibility Controls
- High-contrast mode (yellow-on-black theme for low-vision users).
- Large text toggle (enlarges base UI font).
- Auto-speak mode for immediate auditory feedback upon analysis.
- Clear ARIA labels and keyboard focus outlines.

---

## 4. Safety & Navigation Disclaimer

> ⚠️ **CRITICAL ACCESSIBILITY NOTICE:**  
> **Drishti AI is an assistive visual aid, NOT a certified orientation, mobility, or navigation device.**  
> - **Never** make safety-critical decisions, cross streets, or navigate hazardous terrain based on AI descriptions alone.
> - **Never** ingest medications without human verification of prescription labels.
> - **Always** use traditional mobility tools (such as white canes, guide dogs, or tactile indicators) and consult certified orientation and mobility specialists.

---

## 5. Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application logic |
| **Frontend** | Streamlit | Accessible web dashboard |
| **AI Runtime** | Ollama (Local API `localhost:11434`) | Offline LLM execution |
| **Vision Model** | `gemma4:e2b` / `llava` | Multimodal visual reasoning |
| **OCR Engine** | Tesseract OCR (UB-Mannheim) | Text extraction (`eng` + `hin`) |
| **Image Processing** | Pillow & OpenCV | Preprocessing, filters, transformations |
| **Speech Engine** | `pyttsx3` (Windows SAPI5) | Offline TTS voice synthesis |
| **Platform** | Windows 10 / 11 | Host operating system |

---

## 6. Prerequisites & Dependencies

Before running Drishti AI, ensure you have:
1. **Windows 10 or 11 (64-bit)**
2. **Python 3.10 to 3.14** installed.
3. **Ollama** installed on your system.
4. **Tesseract OCR for Windows** installed.

---

## 7. Step-by-Step Windows Installation Guide

Open **Windows PowerShell** (or VS Code Integrated Terminal) and follow these steps:

### Step 1: Clone or Navigate to Drishti AI
```powershell
cd "c:\Users\PRINCE PANDIT\Documents\Drishti-AI"
```

### Step 2: Create and Activate a Python Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### Step 3: Install Required Python Packages
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install and Start Ollama with Gemma 4 Vision
1. Download Ollama for Windows from [https://ollama.com/download](https://ollama.com/download) if not already installed.
2. In PowerShell, pull the multimodal model:
   ```powershell
   ollama pull gemma4:e2b
   ```
   *(Note: You can also use other vision models such as `ollama pull llava` or `ollama pull llama3.2-vision`)*
3. Ensure Ollama is running in the background:
   ```powershell
   ollama serve
   ```

### Step 5: Install Tesseract OCR (for English and Hindi Text Reading)
1. Download the Windows installer from:  
   **[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)**
2. Run the `.exe` installer.
3. In the installer component selection, expand **"Additional language data (download)"** and check **Hindi (`hin`)**.
4. Complete installation to the default path: `C:\Program Files\Tesseract-OCR`.
5. *(Optional)* If installed to a custom folder, add that directory to your Windows System `PATH` or enter the path in Drishti AI **Settings**.

### Step 6: Install Hindi Voice on Windows (Optional for Hindi TTS)
1. Open **Windows Settings** (`Win + I`).
2. Go to **Time & Language** ➔ **Speech**.
3. Under **Manage voices**, click **Add voices**.
4. Search for **Hindi (India)**, select it, and click **Add**.
5. Windows will download the offline voice (e.g. *Microsoft Kalpana* or *Heera*).

---

## 8. Running the Application

Ensure your virtual environment is active, then launch the Streamlit app:

```powershell
streamlit run app.py
```

The application will open automatically in your browser at:
```text
http://localhost:8501
```

---

## 9. Testing with Sample Images

Drishti AI comes with an automatic sample image generator in `sample_images_generator.py`.

You can test without a camera or external pictures right away:
1. Navigate to **Describe My Surroundings** ➔ Select the **"Use Sample Test Image"** tab.
2. Choose `emergency_exit_sign.png` or `indoor_room_scene.png`.
3. Click **"Analyze Scene with AI"** and listen to the spoken description.
4. Navigate to **Read Text (OCR)** ➔ Choose `medicine_label.png`.
5. Click **"Extract and Read Text"** to verify OCR reading and speech readout.

---

## 10. Troubleshooting & Diagnostics

### 1. "Ollama is offline / Connection refused"
- **Cause:** Ollama service is not running.
- **Solution:** Open a new terminal and run `ollama serve`, or open the Ollama application from the Windows Start menu. Verify with `curl http://localhost:11434/api/tags`.

### 2. "Model 'gemma4:e2b' not found"
- **Cause:** The model has not been downloaded yet.
- **Solution:** Run `ollama pull gemma4:e2b`. If you have a different model (e.g. `llava`), select it from the dropdown in Drishti AI **Settings & Diagnostics**.

### 3. "The model does not support images / vision"
- **Cause:** Some Ollama models are text-only.
- **Solution:** In Drishti AI **Settings**, switch your model to `llava`, `moondream`, or `llama3.2-vision`.

### 4. "Tesseract OCR was not found"
- **Cause:** Tesseract is not installed or not in standard paths.
- **Solution:** Open the **Settings & Diagnostics** page in Drishti AI and paste the exact path to `tesseract.exe` (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`), then click **"Test Tesseract Configuration"**.

### 5. "Hindi language pack ('hin') is missing in Tesseract"
- **Cause:** Hindi traineddata was not selected during Tesseract setup.
- **Solution:** Download `hin.traineddata` from [Tesseract tessdata GitHub](https://github.com/tesseract-ocr/tessdata) and copy it into `C:\Program Files\Tesseract-OCR\tessdata\`.

### 6. "Audio is not playing through speakers"
- **Cause:** Windows SAPI5 audio output conflict.
- **Solution:** Use the built-in browser audio player card below each response, which plays the synthesized `.wav` file directly in your web browser.

---

## 11. Future Improvement Roadmap

- [ ] **Continuous Real-Time Video Streaming:** Integration with lightweight edge vision models for 10 FPS continuous hazard warning beeps.
- [ ] **Haptic Feedback Integration:** Connect with Bluetooth wearable vibration bands to signal directional distance.
- [ ] **Currency Note Recognition:** Specialized offline classification for Indian Rupee (₹) and US Dollar notes.
- [ ] **Custom Offline Whisper ASR:** Voice-command input to ask Drishti questions entirely hands-free.

---

*Developed with ❤️ for accessibility, privacy, and independence.*
