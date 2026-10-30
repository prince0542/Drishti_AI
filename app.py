"""
====================================================================
DRISHTI AI — OFFLINE ACCESSIBILITY ASSISTANT
See the World Through AI
====================================================================
A complete, offline-first assistive application for visually impaired
users, featuring local multimodal AI (Gemma 4 / Ollama), Tesseract OCR,
offline text-to-speech (pyttsx3), camera integration, and accessible UI.
====================================================================
"""

import os
import io
import time
from pathlib import Path
from PIL import Image
import streamlit as st

# Local modules
import config
import vision
import ocr_reader
import speech
import camera
import sample_images_generator

# Page Configuration
st.set_page_config(
    page_title=f"{config.APP_NAME} — {config.APP_TAGLINE}",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "language": "en",  # 'en' or 'hi'
        "high_contrast": False,
        "large_text": True,
        "auto_speak": False,
        "ollama_url": config.DEFAULT_OLLAMA_URL,
        "model_name": config.DEFAULT_MODEL_NAME,
        "custom_tesseract_path": "",
        "speech_rate": config.TTS_DEFAULT_RATE,
        "speech_volume": config.TTS_DEFAULT_VOLUME,
        "selected_voice_id": None,
        "last_spoken_text": "",
        "last_audio_file": None,
        "last_description_result": None,
        "last_ocr_result": None,
        "active_section": "Home Dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# -----------------------------------------------------------------------------
# Accessible CSS Styling
# -----------------------------------------------------------------------------
def apply_custom_styles():
    high_contrast = st.session_state.high_contrast
    large_text = st.session_state.large_text

    font_base = "18px" if large_text else "16px"
    font_h1 = "2.4rem" if large_text else "2.0rem"
    font_h2 = "1.8rem" if large_text else "1.5rem"

    if high_contrast:
        # Ultra High Contrast (Accessibility Mode)
        bg_primary = "#000000"
        bg_card = "#0A0A0A"
        border_color = "#FFFF00"
        text_primary = "#FFFFFF"
        text_secondary = "#FFFF00"
        accent_color = "#00FFFF"
        button_bg = "#FFFF00"
        button_text = "#000000"
        badge_bg = "#111111"
    else:
        # Professional Modern Dark Navy / Purple-Blue Theme
        bg_primary = "#0B0F19"
        bg_card = "#161F30"
        border_color = "#2D3748"
        text_primary = "#F8FAFC"
        text_secondary = "#94A3B8"
        accent_color = "#6366F1"
        button_bg = "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)"
        button_text = "#FFFFFF"
        badge_bg = "#1E293B"

    css = f"""
    <style>
    /* Global Styles */
    .stApp {{
        background-color: {bg_primary};
        color: {text_primary};
        font-size: {font_base};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}
    
    h1, .drishti-h1 {{
        color: {text_primary} !important;
        font-size: {font_h1} !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }}

    h2, .drishti-h2 {{
        color: {accent_color} !important;
        font-size: {font_h2} !important;
        font-weight: 700 !important;
        margin-top: 1rem;
    }}

    h3, .drishti-h3 {{
        color: {text_primary} !important;
        font-size: 1.3rem !important;
        font-weight: 600;
    }}

    /* Drishti Accessible Card */
    .drishti-card {{
        background-color: {bg_card};
        border: 2px solid {border_color};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}

    .drishti-card-highlight {{
        background-color: {bg_card};
        border: 2.5px solid {accent_color};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }}

    /* Accessible Safety Alert Banner */
    .safety-banner {{
        background-color: #2D1515;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #FEE2E2;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }}

    /* Badges */
    .status-badge {{
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: {badge_bg};
        border: 1px solid {border_color};
    }}

    .badge-online {{
        color: #10B981;
        border-color: #10B981;
    }}

    .badge-warning {{
        color: #F59E0B;
        border-color: #F59E0B;
    }}

    .badge-offline {{
        color: #EF4444;
        border-color: #EF4444;
    }}

    /* Buttons */
    .stButton > button {{
        background: {button_bg} !important;
        color: {button_text} !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        border: 1px solid {border_color} !important;
        transition: all 0.2s ease-in-out;
        min-height: 48px; /* Accessible touch target */
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.3);
    }}

    /* Large Readable Text Box */
    .result-text-box {{
        background-color: {bg_card};
        border: 2px solid {accent_color};
        border-radius: 10px;
        padding: 1.25rem;
        font-size: 1.2rem;
        line-height: 1.7;
        color: {text_primary};
        margin: 1rem 0;
        white-space: pre-wrap;
    }}

    /* Keyboard focus ring */
    button:focus, input:focus, select:focus, textarea:focus {{
        outline: 3px solid {accent_color} !important;
        outline-offset: 2px !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_custom_styles()

# -----------------------------------------------------------------------------
# Helper Functions: Audio & Speech Handling
# -----------------------------------------------------------------------------
def speak_text(text: str):
    """Generates audio file and updates session state for replay."""
    if not text or not text.strip():
        st.warning("No text to speak.")
        return

    st.session_state.last_spoken_text = text.strip()
    result = speech.speech_manager.generate_audio_file(
        text=text,
        voice_id=st.session_state.selected_voice_id,
        rate=st.session_state.speech_rate,
        volume=st.session_state.speech_volume,
    )
    if result["success"]:
        st.session_state.last_audio_file = result["audio_path"]
        # Also attempt direct system audio playback
        speech.speech_manager.speak_direct(
            text=text,
            voice_id=st.session_state.selected_voice_id,
            rate=st.session_state.speech_rate,
            volume=st.session_state.speech_volume,
        )
    else:
        st.error(f"Speech synthesis error: {result['error']}")


def render_audio_controls(text_to_speak: str, key_prefix: str = "audio"):
    """Renders accessible audio player, speak, stop, and repeat buttons."""
    c1, c2, c3 = st.columns([1.5, 1, 1])

    with c1:
        if st.button("🔊 Read Aloud (बोलें)", key=f"{key_prefix}_speak", use_container_width=True):
            speak_text(text_to_speak)

    with c2:
        if st.button("⏹️ Stop (रोकें)", key=f"{key_prefix}_stop", use_container_width=True):
            speech.speech_manager.stop()
            st.info("Audio playback stopped.")

    with c3:
        if st.button("🔁 Repeat Last", key=f"{key_prefix}_repeat", use_container_width=True):
            last_txt = st.session_state.last_spoken_text
            if last_txt:
                speak_text(last_txt)
            else:
                st.info("No prior response to repeat.")

    # In-browser audio player if audio file is generated
    if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
        try:
            with open(st.session_state.last_audio_file, "rb") as af:
                audio_bytes = af.read()
            st.audio(audio_bytes, format="audio/wav")
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Sidebar Navigation & Accessibility Quick Controls
# -----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Brand Header
        st.markdown(
            f"""
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <h1 style="font-size: 1.8rem; margin: 0; color: #6366F1;">👁️ {config.APP_NAME}</h1>
                <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">{config.APP_TAGLINE}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Language Selection
        st.markdown("### 🌐 Language / भाषा")
        lang_choice = st.radio(
            "Select Interface & Voice Language",
            options=["English", "हिन्दी (Hindi)"],
            index=0 if st.session_state.language == "en" else 1,
            label_visibility="collapsed",
        )
        st.session_state.language = "en" if "English" in lang_choice else "hi"

        st.markdown("---")

        # Navigation Menu
        st.markdown("### 🧭 Navigation")
        nav_options = [
            "Home Dashboard",
            "Describe My Surroundings",
            "Read Text (OCR)",
            "Camera Assistant",
            "Voice Assistant",
            "Settings & Diagnostics",
            "About Drishti AI",
        ]
        selected_nav = st.radio(
            "Go to section",
            options=nav_options,
            index=nav_options.index(st.session_state.active_section)
            if st.session_state.active_section in nav_options
            else 0,
            label_visibility="collapsed",
        )
        st.session_state.active_section = selected_nav

        st.markdown("---")

        # Accessibility Toggles
        st.markdown("### ♿ Accessibility Mode")
        st.session_state.high_contrast = st.toggle(
            "⚡ High Contrast Mode",
            value=st.session_state.high_contrast,
            help="High-contrast black & yellow theme for low vision users.",
        )
        st.session_state.large_text = st.toggle(
            "🔍 Large Text (20pt)",
            value=st.session_state.large_text,
            help="Increases base typography sizing.",
        )
        st.session_state.auto_speak = st.toggle(
            "📢 Auto-Speak Results",
            value=st.session_state.auto_speak,
            help="Automatically speaks scene descriptions and OCR readings aloud.",
        )

        st.markdown("---")

        # Quick System Readiness Check
        st.markdown("### 🔌 Offline Status")
        is_ollama_on, models, _ = vision.check_ollama_status(st.session_state.ollama_url)
        tess_status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)

        ollama_badge = '<span class="status-badge badge-online">● Ollama: Ready</span>' if is_ollama_on else '<span class="status-badge badge-offline">○ Ollama: Offline</span>'
        tess_badge = '<span class="status-badge badge-online">● OCR: Ready</span>' if tess_status["installed"] else '<span class="status-badge badge-warning">○ OCR: Not Found</span>'

        st.markdown(f"{ollama_badge} {tess_badge}", unsafe_allow_html=True)
        st.caption(f"Active Model: `{st.session_state.model_name}`")


render_sidebar()

# -----------------------------------------------------------------------------
# Section 1: Home Dashboard
# -----------------------------------------------------------------------------
def render_home_dashboard():
    st.markdown(f'<h1 class="drishti-h1">👁️ {config.APP_NAME}</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size: 1.25rem; color: #94A3B8; margin-top: -10px;">{config.APP_TAGLINE} — Offline Accessibility Assistant</p>',
        unsafe_allow_html=True,
    )

    # Critical Safety Disclaimer Banner
    disclaimer = config.SAFETY_DISCLAIMER_HI if st.session_state.language == "hi" else config.SAFETY_DISCLAIMER_EN
    st.markdown(f'<div class="safety-banner">{disclaimer}</div>', unsafe_allow_html=True)

    # System Status & Diagnostic Summary Card
    st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="drishti-h3">🖥️ System Health & Local AI Readiness</h3>', unsafe_allow_html=True)

    is_ollama_on, models, ollama_msg = vision.check_ollama_status(st.session_state.ollama_url)
    tess_status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)
    voices_info = speech.speech_manager.get_installed_voices()

    col1, col2, col3 = st.columns(3)
    with col1:
        if is_ollama_on:
            st.success(f"**Ollama AI:** Online ({len(models)} models)")
            has_target = any(st.session_state.model_name.lower() in m.lower() for m in models)
            if has_target:
                st.caption(f"✅ Active model `{st.session_state.model_name}` loaded.")
            else:
                st.warning(f"⚠️ `{st.session_state.model_name}` not in list. Available: {', '.join(models[:3]) or 'None'}")
        else:
            st.error(f"**Ollama AI:** Offline\n\nRun `ollama serve` in terminal.")

    with col2:
        if tess_status["installed"]:
            st.success(f"**Tesseract OCR:** Active (v{tess_status['version']})")
            hin_state = "✅ Hindi enabled" if tess_status["has_hindi"] else "⚠️ English only (hin missing)"
            st.caption(hin_state)
        else:
            st.warning("**Tesseract OCR:** Not installed / not in PATH.")
            st.caption("Read Text (OCR) requires Tesseract. Visit Settings for guide.")

    with col3:
        if voices_info["available"]:
            st.success(f"**TTS Speech:** Ready ({len(voices_info['voices'])} voices)")
            hin_voice_note = "✅ Hindi voice detected" if voices_info["has_hindi"] else "ℹ️ Standard SAPI voice"
            st.caption(hin_voice_note)
        else:
            st.error("**TTS Speech:** Offline engine unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Quick Action Cards
    st.markdown('<h2 class="drishti-h2">🚀 Quick Assistive Actions</h2>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            """
            <div class="drishti-card-highlight">
                <h3 style="margin-top:0;">📷 Describe My Surroundings</h3>
                <p>Upload a photo or capture an image of your environment. Gemma 4 will describe objects, colors, obstacles, and spatial positioning in simple language.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Scene Description →", key="btn_go_describe", use_container_width=True):
            st.session_state.active_section = "Describe My Surroundings"
            st.rerun()

    with q2:
        st.markdown(
            """
            <div class="drishti-card-highlight">
                <h3 style="margin-top:0;">📖 Read Text (OCR)</h3>
                <p>Extract and hear text from street signs, medicine packaging, notices, labels, and printed documents with support for both Hindi and English.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Text Reader (OCR) →", key="btn_go_ocr", use_container_width=True):
            st.session_state.active_section = "Read Text (OCR)"
            st.rerun()

    q3, q4 = st.columns(2)
    with q3:
        st.markdown(
            """
            <div class="drishti-card">
                <h3 style="margin-top:0;">🎥 Camera Assistant</h3>
                <p>Use your computer webcam or USB camera to capture snapshots in real time and receive instant auditory feedback.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Camera Assistant →", key="btn_go_camera", use_container_width=True):
            st.session_state.active_section = "Camera Assistant"
            st.rerun()

    with q4:
        st.markdown(
            """
            <div class="drishti-card">
                <h3 style="margin-top:0;">🔊 Voice Assistant & Controls</h3>
                <p>Test your installed Windows voices, adjust speech rate, replay your last visual description, or synthesize any custom message.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Voice Assistant →", key="btn_go_voice", use_container_width=True):
            st.session_state.active_section = "Voice Assistant"
            st.rerun()


# -----------------------------------------------------------------------------
# Section 2: Describe My Surroundings
# -----------------------------------------------------------------------------
def render_describe_surroundings():
    lang = st.session_state.language
    title = "वर्णन: मेरे आसपास क्या है? (Describe My Surroundings)" if lang == "hi" else "Describe My Surroundings"
    st.markdown(f'<h1 class="drishti-h1">🔍 {title}</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.15rem; color: #94A3B8;">Provides clear, spatial descriptions of objects, people, pathways, and visual context using local AI.</p>',
        unsafe_allow_html=True,
    )

    # Input method selection
    tab_upload, tab_sample = st.tabs(["📁 Upload Image (चित्र अपलोड करें)", "🖼️ Use Sample Test Image"])

    active_image = None
    image_source_label = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG, WEBP, BMP)",
            type=config.SUPPORTED_IMAGE_TYPES,
            help="Select an image file from your computer.",
            key="surroundings_upload",
        )
        if uploaded_file:
            try:
                active_image = Image.open(uploaded_file)
                image_source_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Failed to read image file: {str(e)}")

    with tab_sample:
        # Ensure sample images exist
        sample_paths = sample_images_generator.generate_sample_images()
        sample_options = [p.name for p in sample_paths if p.exists()]
        if sample_options:
            selected_sample = st.selectbox(
                "Select a pre-built accessibility test image:",
                options=sample_options,
                key="surroundings_sample_choice",
            )
            sample_file_path = config.SAMPLE_IMAGES_DIR / selected_sample
            if sample_file_path.exists():
                active_image = Image.open(sample_file_path)
                image_source_label = f"Sample: {selected_sample}"
        else:
            st.info("No sample images found.")

    if active_image is not None:
        col_img, col_ctrl = st.columns([1, 1.2])

        with col_img:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.image(active_image, caption=image_source_label, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ctrl:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">⚙️ Analysis Options</h3>', unsafe_allow_html=True)

            # Language selection for this specific run
            analysis_lang = st.radio(
                "Description Language / भाषा:",
                options=["English", "हिन्दी (Hindi)"],
                index=0 if lang == "en" else 1,
                horizontal=True,
                key="surroundings_lang_choice",
            )
            selected_lang_code = "en" if "English" in analysis_lang else "hi"

            # Preset focal point options
            focus_preset = st.selectbox(
                "Focus Area (वैकल्पिक फोकस):",
                options=[
                    "General Scene & Layout (सामान्य दृश्य और लेआउट)",
                    "Obstacle & Pathway Check (मार्ग और बाधा जांच)",
                    "People & Objects Identification (वस्तुएं और लोग)",
                    "Signboards & Visible Text (बोर्ड और लिखावट)",
                ],
                key="surroundings_focus_preset",
            )

            custom_prompt = None
            if "Obstacle" in focus_preset:
                custom_prompt = (
                    "Focus specifically on ground terrain, floor obstacles, elevation changes, "
                    "doors, stairs, and walking hazards. Describe their exact relative positions."
                    if selected_lang_code == "en"
                    else "विशेष रूप से जमीन की सतह, फर्श की बाधाओं, दरवाजों, सीढ़ियों और पैदल चलने के खतरों पर ध्यान दें। उनकी स्थिति बताएं।"
                )
            elif "Signboards" in focus_preset:
                custom_prompt = (
                    "Transcribe and explain any visible notices, signboards, labels, or text in the scene."
                    if selected_lang_code == "en"
                    else "दृश्य में दिखाई देने वाले किसी भी नोटिस, साइनबोर्ड, लेबल या टेक्स्ट को पढ़कर समझाएं।"
                )

            # Large Accessible Action Button
            btn_label = "🔍 दृश्य का विश्लेषण करें (Analyze Scene)" if selected_lang_code == "hi" else "🔍 Analyze Scene with AI"
            if st.button(btn_label, key="btn_run_vision", use_container_width=True):
                with st.spinner("🤖 Drishti AI is analyzing the scene offline... Please wait."):
                    result = vision.describe_scene(
                        image_input=active_image,
                        language=selected_lang_code,
                        model_name=st.session_state.model_name,
                        ollama_url=st.session_state.ollama_url,
                        custom_prompt=custom_prompt,
                    )
                    st.session_state.last_description_result = result

                    # Auto-speak if enabled
                    if result["success"] and st.session_state.auto_speak:
                        speak_text(result["description"])

            st.markdown("</div>", unsafe_allow_html=True)

        # Display Results
        if st.session_state.last_description_result:
            res = st.session_state.last_description_result
            st.markdown("---")
            if res["success"]:
                st.markdown('<div class="drishti-card-highlight">', unsafe_allow_html=True)
                st.markdown(
                    f'<h2 class="drishti-h2" style="margin-top:0;">🗣️ Visual Description ({res["model_used"]})</h2>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="result-text-box">{res["description"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"⚡ Inferred locally in {res['duration_seconds']}s • 100% Offline Privacy Guarantee")

                # Audio Controls
                render_audio_controls(res["description"], key_prefix="vision_res")
                st.markdown("</div>", unsafe_allow_html=True)

                # Safety Reminder
                st.info(res["disclaimer"])
            else:
                st.error(f"❌ Analysis failed: {res['error']}")
                st.markdown(
                    """
                    **Troubleshooting Tips:**
                    1. Ensure Ollama is running (`ollama serve`).
                    2. Check if the model is installed (`ollama pull gemma4:e2b`).
                    3. If `gemma4:e2b` does not support vision weights on your setup, switch to `llava` in Settings.
                    """
                )
    else:
        st.info("👆 Please upload an image or choose a sample image above to start.")


# -----------------------------------------------------------------------------
# Section 3: Read Text (OCR)
# -----------------------------------------------------------------------------
def render_read_text_ocr():
    lang = st.session_state.language
    title = "पाठ पढ़ें (Read Text — OCR)" if lang == "hi" else "Read Text (OCR Reader)"
    st.markdown(f'<h1 class="drishti-h1">📖 {title}</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.15rem; color: #94A3B8;">Extracts and speaks text from medicine labels, street notices, menus, and documents in English and Hindi.</p>',
        unsafe_allow_html=True,
    )

    # Check Tesseract status
    tess_status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)
    if not tess_status["installed"]:
        st.warning(
            f"⚠️ **Tesseract OCR is not detected:** {tess_status['message']}\n\n"
            "You can configure the exact path to `tesseract.exe` in the **Settings & Diagnostics** tab."
        )

    tab_upload, tab_sample = st.tabs(["📁 Upload Image / Document", "🖼️ Use Sample Test Image"])
    active_image = None
    image_source_label = ""

    with tab_upload:
        ocr_file = st.file_uploader(
            "Upload image containing text",
            type=config.SUPPORTED_IMAGE_TYPES,
            key="ocr_upload_file",
        )
        if ocr_file:
            try:
                active_image = Image.open(ocr_file)
                image_source_label = f"Uploaded: {ocr_file.name}"
            except Exception as e:
                st.error(f"Failed to load image: {str(e)}")

    with tab_sample:
        sample_paths = sample_images_generator.generate_sample_images()
        # Prefer signboard or medicine label
        sample_options = [p.name for p in sample_paths if p.exists()]
        if sample_options:
            selected_sample = st.selectbox(
                "Choose sample text document:",
                options=sample_options,
                index=0,
                key="ocr_sample_choice",
            )
            sample_file_path = config.SAMPLE_IMAGES_DIR / selected_sample
            if sample_file_path.exists():
                active_image = Image.open(sample_file_path)
                image_source_label = f"Sample: {selected_sample}"

    if active_image is not None:
        col_img, col_ctrl = st.columns([1, 1.2])

        with col_img:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.image(active_image, caption=image_source_label, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ctrl:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">⚙️ OCR Settings</h3>', unsafe_allow_html=True)

            ocr_lang_option = st.selectbox(
                "OCR Language Target:",
                options=[
                    "English + Hindi (eng+hin)",
                    "English Only (eng)",
                    "Hindi Only (hin)",
                ],
                index=0,
                key="ocr_lang_target",
            )

            code_map = {
                "English + Hindi (eng+hin)": "eng+hin",
                "English Only (eng)": "eng",
                "Hindi Only (hin)": "hin",
            }
            lang_code = code_map[ocr_lang_option]

            apply_prep = st.checkbox("Apply Contrast & Denoising Preprocessing", value=True)
            prep_method = st.selectbox(
                "Preprocessing Technique:",
                options=["adaptive (Best for documents/signs)", "otsu (High contrast binary)", "contrast (CLAHE)"],
                index=0,
            )
            prep_key = prep_method.split()[0]

            if st.button("📖 Extract and Read Text (पाठ निकालें)", key="btn_run_ocr", use_container_width=True):
                with st.spinner("⏳ Extracting text using Tesseract OCR..."):
                    result = ocr_reader.extract_text(
                        image_input=active_image,
                        language_choice=lang_code,
                        apply_preprocessing=apply_prep,
                        preprocess_method=prep_key,
                        custom_tesseract_path=st.session_state.custom_tesseract_path,
                    )
                    st.session_state.last_ocr_result = result

                    if result["success"] and result["text"] and st.session_state.auto_speak:
                        speak_text(result["text"])

            st.markdown("</div>", unsafe_allow_html=True)

        # Display Extracted Text
        if st.session_state.last_ocr_result:
            res = st.session_state.last_ocr_result
            st.markdown("---")
            if res["success"]:
                st.markdown('<div class="drishti-card-highlight">', unsafe_allow_html=True)
                st.markdown('<h2 class="drishti-h2" style="margin-top:0;">📄 Extracted Text Content</h2>', unsafe_allow_html=True)

                if res.get("warning"):
                    st.warning(res["warning"])

                if res["text"]:
                    st.markdown(
                        f'<div class="result-text-box">{res["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"📊 Words: {res['word_count']} | Language Engine: `{res['language_used']}` | "
                        f"OCR Confidence: {res['confidence']}%"
                    )

                    # Spoken Controls
                    render_audio_controls(res["text"], key_prefix="ocr_res")
                else:
                    st.info("No text characters were recognized in this image. Try adjusting the preprocessing method.")

                # If preprocessed image is available, display comparison
                if res.get("preprocessed_image") and apply_prep:
                    with st.expander("🔍 View Preprocessed Image (OpenCV Filter Output)"):
                        st.image(res["preprocessed_image"], caption="Denoised & Thresholded OCR View", width=450)

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(f"❌ OCR Extraction Error: {res['error']}")
    else:
        st.info("👆 Select or upload an image to read its text.")


# -----------------------------------------------------------------------------
# Section 4: Camera Assistant
# -----------------------------------------------------------------------------
def render_camera_assistant():
    st.markdown('<h1 class="drishti-h1">🎥 Camera Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.15rem; color: #94A3B8;">Capture live snapshots from your laptop webcam or USB camera for immediate voice feedback.</p>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Choose Camera Interface:",
        options=["Streamlit WebRTC Camera (Recommended)", "OpenCV Direct Hardware Snapshot"],
        horizontal=True,
        key="camera_mode_radio",
    )

    captured_frame = None

    if "Streamlit" in mode:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        cam_file = st.camera_input("Take a photo using your webcam", key="st_webcam_input")
        if cam_file:
            captured_frame = camera.convert_streamlit_camera_file(cam_file)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # OpenCV Direct Hardware Capture
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        col_dev, col_btn = st.columns([1, 1])
        with col_dev:
            cam_index = st.number_input("Camera Device Index:", min_value=0, max_value=4, value=0, step=1)
        with col_btn:
            st.write("")
            st.write("")
            if st.button("📸 Capture OpenCV Frame", key="btn_cv_capture", use_container_width=True):
                with st.spinner("Capturing frame from camera..."):
                    cap_result = camera.capture_frame_from_camera(camera_index=cam_index)
                    if cap_result["success"]:
                        st.session_state["cv_captured_image"] = cap_result["image"]
                        st.success(f"Captured {cap_result['resolution'][0]}x{cap_result['resolution'][1]} frame!")
                    else:
                        st.error(f"Camera capture failed: {cap_result['error']}")

        if "cv_captured_image" in st.session_state and st.session_state["cv_captured_image"]:
            captured_frame = st.session_state["cv_captured_image"]
            st.image(captured_frame, caption="Captured Frame Preview", width=500)

        st.markdown("</div>", unsafe_allow_html=True)

    # If a frame was captured, offer immediate assistive actions
    if captured_frame:
        st.markdown('<div class="drishti-card-highlight">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">⚡ Instant Assistive Actions</h3>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Describe What Camera Sees", key="btn_cam_describe", use_container_width=True):
                with st.spinner("Analyzing camera image with local AI..."):
                    res = vision.describe_scene(
                        image_input=captured_frame,
                        language=st.session_state.language,
                        model_name=st.session_state.model_name,
                        ollama_url=st.session_state.ollama_url,
                    )
                    st.session_state.last_description_result = res
                    if res["success"]:
                        speak_text(res["description"])
                        st.success("Analysis complete! Audio is playing.")
                    else:
                        st.error(f"Analysis error: {res['error']}")

        with c2:
            if st.button("📖 Read Text in Front of Camera", key="btn_cam_ocr", use_container_width=True):
                with st.spinner("Reading visible text with OCR..."):
                    res = ocr_reader.extract_text(
                        image_input=captured_frame,
                        language_choice="eng+hin",
                        custom_tesseract_path=st.session_state.custom_tesseract_path,
                    )
                    st.session_state.last_ocr_result = res
                    if res["success"] and res["text"]:
                        speak_text(res["text"])
                        st.success("Extracted text recognized and spoken.")
                    else:
                        st.warning("No readable text detected in current frame.")

        # Display result if available
        if st.session_state.last_description_result and st.session_state.last_description_result.get("success"):
            desc = st.session_state.last_description_result["description"]
            st.markdown(f'<div class="result-text-box">{desc}</div>', unsafe_allow_html=True)
            render_audio_controls(desc, key_prefix="cam_desc_res")

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 5: Voice Assistant & Speech Center
# -----------------------------------------------------------------------------
def render_voice_assistant():
    st.markdown('<h1 class="drishti-h1">🔊 Voice Assistant & Audio Center</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.15rem; color: #94A3B8;">100% offline text-to-speech powered by pyttsx3 and Windows SAPI5 audio engines.</p>',
        unsafe_allow_html=True,
    )

    voices_info = speech.speech_manager.get_installed_voices()

    col_ctrl, col_test = st.columns([1, 1.2])

    with col_ctrl:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">🎛️ Speech Parameters</h3>', unsafe_allow_html=True)

        st.session_state.speech_rate = st.slider(
            "Speech Rate (Words Per Minute):",
            min_value=100,
            max_value=260,
            value=st.session_state.speech_rate,
            step=5,
            help="160 WPM is standard natural speaking speed.",
        )

        st.session_state.speech_volume = st.slider(
            "Volume Level:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.speech_volume,
            step=0.05,
        )

        # Voice Selector
        if voices_info["available"] and voices_info["voices"]:
            voice_options = {v["name"]: v["id"] for v in voices_info["voices"]}
            selected_voice_name = st.selectbox(
                "Active Windows Voice:",
                options=list(voice_options.keys()),
                index=0,
            )
            st.session_state.selected_voice_id = voice_options[selected_voice_name]
        else:
            st.error("No Windows SAPI voices detected.")

        st.markdown("---")
        st.markdown('<h4 class="drishti-h4">🔁 Quick Response Recall</h4>', unsafe_allow_html=True)
        last_spoken = st.session_state.last_spoken_text
        if last_spoken:
            st.caption(f"**Last Spoken:** _{last_spoken[:120]}..._")
            c_rep, c_stp = st.columns(2)
            with c_rep:
                if st.button("🔁 Replay Last Response", key="btn_replay_voice_page", use_container_width=True):
                    speak_text(last_spoken)
            with c_stp:
                if st.button("⏹️ Stop Playback", key="btn_stop_voice_page", use_container_width=True):
                    speech.speech_manager.stop()
        else:
            st.info("No prior response saved in session memory yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_test:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">🗣️ Test Custom Speech Output</h3>', unsafe_allow_html=True)

        sample_prompts = [
            "Hello, I am Drishti AI. Your offline accessibility assistant.",
            "नमस्ते, मैं दृष्टि एआई हूँ। आपकी ऑफ़लाइन सुगम्यता सहायक।",
            "Ahead of you is a clear hallway. A doorway is on your left at two meters.",
            "Emergency exit door located directly in front. Keep the pathway clear.",
        ]
        chosen_sample = st.selectbox("Or choose a sample phrase:", options=["(Custom input)"] + sample_prompts)

        default_txt = "" if chosen_sample == "(Custom input)" else chosen_sample
        user_speech_input = st.text_area(
            "Enter text to speak aloud:",
            value=default_txt,
            height=120,
            key="voice_test_input",
        )

        if st.button("🔊 Speak Text (बोलें)", key="btn_test_speak", use_container_width=True):
            if user_speech_input.strip():
                speak_text(user_speech_input)
            else:
                st.warning("Please type some text to speak.")

        if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
            st.markdown("---")
            st.caption("🎧 Generated Audio Player:")
            with open(st.session_state.last_audio_file, "rb") as f:
                st.audio(f.read(), format="audio/wav")

        st.markdown("</div>", unsafe_allow_html=True)

    # Hindi Voice Installation Guide Card
    st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="drishti-h3">🇮🇳 Hindi Voice Guide for Windows</h3>', unsafe_allow_html=True)
    if voices_info["has_hindi"]:
        st.success(f"✅ Dedicated Hindi voice detected: {', '.join([v['name'] for v in voices_info['hindi_voices']])}")
    else:
        st.info(
            """
            **How to install the official Hindi Speech Voice on Windows 10/11:**
            1. Open **Windows Settings** (`Win + I`).
            2. Navigate to **Time & Language** ➔ **Speech**.
            3. Under **Manage voices**, click **Add voices**.
            4. Search for **Hindi (India)**, select it, and click **Add**.
            5. Once installed, restart Drishti AI and the Hindi voice (e.g. *Microsoft Kalpana* or *Heera*) will automatically appear here!
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 6: Settings & Diagnostics
# -----------------------------------------------------------------------------
def render_settings():
    st.markdown('<h1 class="drishti-h1">⚙️ Settings & System Diagnostics</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.15rem; color: #94A3B8;">Configure local AI endpoints, verify Tesseract installation, and audit system resources.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">🤖 Ollama Model Configuration</h3>', unsafe_allow_html=True)

        new_ollama_url = st.text_input(
            "Ollama Base URL:",
            value=st.session_state.ollama_url,
            help="Default is http://localhost:11434",
        )
        if new_ollama_url != st.session_state.ollama_url:
            st.session_state.ollama_url = new_ollama_url

        # Check installed models
        is_on, installed_models, msg = vision.check_ollama_status(st.session_state.ollama_url)
        st.caption(msg)

        all_model_options = list(set([st.session_state.model_name] + installed_models + config.SUPPORTED_VISION_MODELS))
        selected_model = st.selectbox(
            "Selected Vision Model:",
            options=all_model_options,
            index=all_model_options.index(st.session_state.model_name)
            if st.session_state.model_name in all_model_options
            else 0,
        )
        st.session_state.model_name = selected_model

        if st.button("🔄 Test Ollama Connection", key="btn_test_ollama", use_container_width=True):
            test_on, models_found, test_msg = vision.check_ollama_status(st.session_state.ollama_url)
            if test_on:
                st.success(f"✅ Connection successful! Found {len(models_found)} models: {', '.join(models_found)}")
            else:
                st.error(f"❌ Connection failed: {test_msg}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">🔍 Tesseract OCR Configuration</h3>', unsafe_allow_html=True)

        found_path, auto_path = ocr_reader.find_tesseract_binary()
        st.caption(f"Auto-detected path: `{auto_path or 'Not found in standard paths'}`")

        custom_path_input = st.text_input(
            "Custom Tesseract.exe Path:",
            value=st.session_state.custom_tesseract_path,
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            help="Leave empty to use auto-detected path.",
        )
        st.session_state.custom_tesseract_path = custom_path_input

        if st.button("🧪 Test Tesseract Configuration", key="btn_test_tess", use_container_width=True):
            status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)
            if status["installed"]:
                st.success(f"✅ {status['message']}")
            else:
                st.error(f"❌ {status['message']}")

        st.markdown("---")
        st.markdown(
            """
            **How to install Tesseract OCR on Windows:**
            1. Download Windows installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
            2. Run the installer (check **Additional language data (Hindi)** during setup).
            3. Default install path: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Full Diagnostic Audit Card
    st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="drishti-h3">📋 System Readiness Audit Checklist</h3>', unsafe_allow_html=True)

    c_audit1, c_audit2 = st.columns(2)
    with c_audit1:
        st.write(f"• **Python Platform:** `{os.name}` / Windows")
        st.write(f"• **App Directory:** `{config.BASE_DIR}`")
        st.write(f"• **Temporary Audio Cache:** `{config.TEMP_AUDIO_DIR}`")
    with c_audit2:
        st.write(f"• **OpenCV (cv2) Available:** `{'Yes' if camera.cv2 is not None else 'No'}`")
        st.write(f"• **pytesseract Available:** `{'Yes' if ocr_reader.pytesseract is not None else 'No'}`")
        st.write(f"• **pyttsx3 Available:** `{'Yes' if speech.pyttsx3 is not None else 'No'}`")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 7: About Drishti AI & Accessibility Safety
# -----------------------------------------------------------------------------
def render_about():
    st.markdown('<h1 class="drishti-h1">ℹ️ About DRISHTI AI</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size: 1.25rem; color: #94A3B8;">{config.APP_TAGLINE} — Built for Accessibility, Privacy & Independence</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
    st.markdown(
        """
        ### 🌟 Mission & Core Vision
        **Drishti AI** (दृष्टि meaning *Vision* in Sanskrit and Hindi) was designed from the ground up to restore visual independence for people with blindness or low vision.
        
        Unlike mainstream vision assistants that transmit sensitive camera feeds to remote cloud servers, **Drishti AI operates completely offline on your local machine**:
        - 🔒 **Zero Telemetry / Zero Cloud:** Your images, documents, and surroundings never leave your computer.
        - ⚡ **Local Multimodal AI:** Powered by Gemma 4 / Ollama local neural models.
        - 📖 **Bilingual OCR:** Read signs, prescription labels, notices, and bills in English and Hindi.
        - 🗣️ **Offline Speech Synthesis:** Built-in Windows TTS voice readouts without internet dependency.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Safety Notice
    st.markdown('<div class="safety-banner">', unsafe_allow_html=True)
    st.markdown(
        """
        ### ⚠️ Critical Safety and Orientation Guidelines
        Drishti AI is an accessibility assistant, not a certified navigation system.
        - **Never cross streets or navigate stairs solely relying on AI scene descriptions.**
        - **Do not make medical decisions without human verification of medicine labels.**
        - **Always utilize traditional mobility aids (such as a white cane or guide dog) and consult orientation and mobility specialists.**
        - Drishti AI communicates uncertainty when images are dark, blurry, or occluded to prevent misinformation.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Architecture Overview
    st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
    st.markdown(
        """
        ### 🛠️ Architecture & Technologies
        - **Frontend UI:** Streamlit with accessible high-contrast themes and keyboard focus rings.
        - **Multimodal AI:** Gemma 4 / Ollama via `/api/chat` with base64 encoded image pipelines.
        - **OCR Engine:** Tesseract OCR with OpenCV adaptive noise reduction and thresholding.
        - **Speech Synthesis:** pyttsx3 offline text-to-speech engine with Windows SAPI5 voice routing.
        - **Computer Vision:** Pillow and OpenCV.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Section Router
# -----------------------------------------------------------------------------
def main():
    sec = st.session_state.active_section
    if sec == "Home Dashboard":
        render_home_dashboard()
    elif sec == "Describe My Surroundings":
        render_describe_surroundings()
    elif sec == "Read Text (OCR)":
        render_read_text_ocr()
    elif sec == "Camera Assistant":
        render_camera_assistant()
    elif sec == "Voice Assistant":
        render_voice_assistant()
    elif sec == "Settings & Diagnostics":
        render_settings()
    elif sec == "About Drishti AI":
        render_about()
    else:
        render_home_dashboard()


if __name__ == "__main__":
    main()
