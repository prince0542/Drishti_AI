"""
Drishti AI - Configuration Module
Offline-first Accessibility Assistant for Visually Impaired Users.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
ASSETS_DIR = BASE_DIR / "assets"
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"

# Ensure directories exist safely without throwing errors on restricted environments
try:
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Application Identity & Branding
APP_NAME = "DRISHTI AI"
APP_TAGLINE = "See the World Through AI"
APP_VERSION = "1.0.0"
APP_ICON = "👁️"

# Safety and Liability Disclaimer
SAFETY_DISCLAIMER_EN = (
    "⚠️ CRITICAL SAFETY NOTICE: Drishti AI is an AI-powered visual assistance tool, "
    "NOT a certified navigation system or medical device. Never rely on this application "
    "for street crossing, path clearance, or hazardous obstacle avoidance. Always rely on "
    "a white cane, guide dog, or orientation and mobility specialist."
)

SAFETY_DISCLAIMER_HI = (
    "⚠️ महत्वपूर्ण सुरक्षा सूचना: दृष्टि एआई एक सहायता उपकरण है, यह कोई प्रमाणित "
    "नेविगेशन सिस्टम नहीं है। सड़क पार करने, बाधाओं से बचने या सुरक्षा-संवेदनशील "
    "निर्णयों के लिए केवल इस पर निर्भर न रहें। हमेशा छड़ी या मानवीय सहायता का उपयोग करें।"
)

# Ollama API Configuration
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL_NAME = os.getenv("DRISHTI_MODEL", "gemma4:e2b")

# Popular local vision models supported by Ollama
SUPPORTED_VISION_MODELS = [
    "gemma4:e2b",
    "llava",
    "llava:7b",
    "llava:13b",
    "llama3.2-vision",
    "llama3.2-vision:11b",
    "moondream",
    "bakllava",
    "minicpm-v",
]

OLLAMA_REQUEST_TIMEOUT = 60  # seconds

# Tesseract OCR Paths & Settings
COMMON_TESSERACT_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]

DEFAULT_OCR_LANGUAGES = "eng+hin"
FALLBACK_OCR_LANGUAGES = "eng"

# Offline Text-To-Speech (pyttsx3) Settings
TTS_DEFAULT_RATE = 165  # Words per minute (comfortable listening pace)
TTS_DEFAULT_VOLUME = 1.0  # Range 0.0 to 1.0
TTS_AUDIO_FILENAME = "drishti_spoken_response.wav"

# Supported Image Extensions
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

# Vision Prompts for Accessibility
VISION_PROMPT_EN = (
    "You are Drishti AI, an offline accessibility assistant for visually impaired users. "
    "Analyze the image and provide a concise, high-value spoken description: "
    "1. Immediate surroundings and spatial layout (what is in front, left, right, center). "
    "2. Key objects, people, landmarks, colors, and lighting. "
    "3. Any visible text, signs, or prominent labels. "
    "4. Highlight any potential physical obstacles or ground changes with clear caution. "
    "Use simple, natural, dignified language. Do not invent details not visible. "
    "If the image is blurry, dim, or unclear, explicitly say so."
)

VISION_PROMPT_HI = (
    "आप दृष्टि एआई (Drishti AI) हैं, जो दृष्टिबाधित उपयोगकर्ताओं के लिए एक ऑफ़लाइन सुगम्यता सहायक हैं। "
    "इस चित्र को ध्यान से देखकर सरल, स्वाभाविक और स्पष्ट हिंदी में दृश्य का वर्णन करें: "
    "1. आसपास का वातावरण और स्थिति (सामने, बाईं ओर, दाईं ओर, बीच में क्या है)। "
    "2. मुख्य वस्तुएं, लोग, रंग और प्रकाश की स्थिति। "
    "3. कोई भी दिखाई देने वाला पाठ, साइनबोर्ड या निर्देश। "
    "4. यदि कोई संभावित बाधा या रास्ता हो, तो सावधानी के साथ बताएं। "
    "काल्पनिक विवरण न दें। यदि चित्र धुंधला या अंधेरा है, तो स्पष्ट रूप से बताएं।"
)
