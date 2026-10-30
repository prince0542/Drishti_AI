"""
====================================================================
DRISHTI AI — OFFLINE ACCESSIBILITY ASSISTANT
See the World Through AI
====================================================================
A complete, offline-first assistive application for visually impaired
users, featuring Real-Time Live Camera Scanning, Voice Command Control,
Smart Duplicate-Filtering Announcements, Bilingual OCR, and Speech.
====================================================================
"""

import os
import io
import time
import json
import hashlib
from pathlib import Path
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Local modules
import config
import vision
import ocr_reader
import speech
import camera
import sample_images_generator
import live_assistant
import obstacle_engine

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
        "auto_speak": True,
        "auto_describe_camera": True,
        "auto_describe_upload": False,
        "live_scan_interval": config.DEFAULT_SCAN_INTERVAL,
        "last_processed_cam_hash": None,
        "last_processed_upload_hash": None,
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
        "active_section": "Camera Assistant",
        "voice_command_action": None,
        "last_auto_spoken_live": None,
        "radar_active": False,
        "radar_cam_idx": 1,
        "radar_auto_alert": True,
        "last_spoken_obstacle_alert": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# -----------------------------------------------------------------------------
# Professional AI Product Design & Theme Engine
# -----------------------------------------------------------------------------
def apply_custom_styles():
    high_contrast = st.session_state.high_contrast
    large_text = st.session_state.large_text

    font_base = "18px" if large_text else "16px"
    font_h1 = "2.3rem" if large_text else "1.9rem"
    font_h2 = "1.65rem" if large_text else "1.4rem"

    if high_contrast:
        bg_app = "#000000"
        bg_sidebar = "#0A0A0A"
        bg_card = "#000000"
        border_card = "#FFFF00"
        text_primary = "#FFFFFF"
        text_secondary = "#FFFF00"
        accent_color = "#00FFFF"
        button_bg = "#FFFF00"
        button_text = "#000000"
        button_border = "#FFFF00"
    else:
        bg_app = "#090D16"
        bg_sidebar = "#0D1322"
        bg_card = "#131C2E"
        border_card = "#22314E"
        text_primary = "#F8FAFC"
        text_secondary = "#94A3B8"
        accent_color = "#6366F1"
        button_bg = "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)"
        button_text = "#FFFFFF"
        button_border = "#4F46E5"

    css = f"""
    <style>
    .stApp {{
        background: {bg_app} !important;
        color: {text_primary} !important;
        font-size: {font_base};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid #1E293B !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: {text_primary} !important;
    }}

    section[data-testid="stSidebar"] .stRadio label {{
        color: #E2E8F0 !important;
        font-size: 1.05rem !important;
        padding: 4px 0;
    }}

    h1, .drishti-h1 {{
        color: {text_primary} !important;
        font-size: {font_h1} !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }}

    h2, .drishti-h2 {{
        color: #818CF8 !important;
        font-size: {font_h2} !important;
        font-weight: 700 !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }}

    h3, .drishti-h3 {{
        color: {text_primary} !important;
        font-size: 1.25rem !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}

    .drishti-card {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }}

    .drishti-card-highlight {{
        background: {bg_card};
        border: 2px solid {accent_color};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 30px -5px rgba(99, 102, 241, 0.25);
    }}

    .safety-banner {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.2) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #FEE2E2;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }}

    .status-badge {{
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        background: #111827;
        border: 1px solid #374151;
    }}

    .badge-online {{
        color: #10B981 !important;
        border-color: #10B981 !important;
    }}

    .badge-warning {{
        color: #F59E0B !important;
        border-color: #F59E0B !important;
    }}

    .badge-offline {{
        color: #EF4444 !important;
        border-color: #EF4444 !important;
    }}

    .object-pill {{
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 700;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        background: rgba(99, 102, 241, 0.2);
        border: 1.5px solid #818CF8;
        color: #EEF2FF;
    }}

    .stButton > button {{
        background: {button_bg} !important;
        color: {button_text} !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        border: 1px solid {button_border} !important;
        min-height: 50px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }}

    .btn-stop > button {{
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%) !important;
        border-color: #EF4444 !important;
    }}

    .btn-resume > button {{
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%) !important;
        border-color: #F59E0B !important;
    }}

    .result-text-box {{
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.16) 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.4rem;
        font-size: 1.35rem;
        line-height: 1.75;
        color: #FFFFFF;
        font-weight: 600;
        margin: 1rem 0;
        white-space: pre-wrap;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_custom_styles()

# -----------------------------------------------------------------------------
# Dual-Engine Voice & Audio Output (Browser Web Speech + Audio Element)
# -----------------------------------------------------------------------------
def trigger_browser_speech(text: str, lang: str = "en"):
    """
    Invokes high-clarity speech synthesis on the parent browser window with
    natural voice matching, responsive 1.2x pacing, and immediate playback.
    """
    if not text or not text.strip():
        return

    safe_text = json.dumps(text.strip())
    voice_lang = "hi-IN" if lang == "hi" else "en-US"

    js_code = f"""
    <script>
    (function() {{
        try {{
            var synth = (window.parent && window.parent.speechSynthesis) ? window.parent.speechSynthesis : window.speechSynthesis;
            var Utterance = (window.parent && window.parent.SpeechSynthesisUtterance) ? window.parent.SpeechSynthesisUtterance : SpeechSynthesisUtterance;
            if (synth && Utterance) {{
                synth.cancel();
                
                var utterance = new Utterance({safe_text});
                utterance.rate = 1.15;  // Crisp, intelligible pace
                utterance.pitch = 1.0;  // Natural pitch
                utterance.lang = '{voice_lang}';

                function selectBestVoice() {{
                    var voices = synth.getVoices();
                    if (!voices || voices.length === 0) return null;
                    var target = '{voice_lang}'.toLowerCase().split('-')[0];
                    var matches = voices.filter(function(v) {{
                        return v.lang && v.lang.toLowerCase().replace('_', '-').startsWith(target);
                    }});
                    if (matches.length > 0) {{
                        var natural = matches.find(function(v) {{
                            var n = v.name.toLowerCase();
                            return n.includes('natural') || n.includes('neural') || n.includes('google') || n.includes('online');
                        }});
                        return natural || matches[0];
                    }}
                    return voices[0];
                }}

                var best = selectBestVoice();
                if (best) {{
                    utterance.voice = best;
                }} else if (synth.onvoiceschanged !== undefined) {{
                    synth.onvoiceschanged = function() {{
                        var v = selectBestVoice();
                        if (v) utterance.voice = v;
                    }};
                }}

                synth.speak(utterance);
            }}
        }} catch(e) {{
            console.error('Speech synthesis error:', e);
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)


def speak_text(text: str):
    """Speaks text using Web Speech API, generates local audio file, and plays system audio."""
    if not text or not text.strip():
        return

    st.session_state.last_spoken_text = text.strip()
    trigger_browser_speech(text, lang=st.session_state.language)

    # Hardware speech synthesis directly on laptop speakers (100% offline, zero latency)
    try:
        speech.speech_manager.speak_direct(
            text=text,
            voice_id=st.session_state.selected_voice_id,
            rate=st.session_state.speech_rate,
            volume=st.session_state.speech_volume,
        )
    except Exception:
        pass

    try:
        res = speech.speech_manager.generate_audio_file(
            text=text,
            voice_id=st.session_state.selected_voice_id,
            rate=st.session_state.speech_rate,
            volume=st.session_state.speech_volume,
        )
        if res.get("success"):
            st.session_state.last_audio_file = res["audio_path"]
    except Exception:
        pass


def stop_all_speech():
    """Stops all active speech synthesis immediately."""
    speech.speech_manager.stop()
    js_stop = """
    <script>
    try {
        var synth = (window.parent && window.parent.speechSynthesis) ? window.parent.speechSynthesis : window.speechSynthesis;
        if (synth) synth.cancel();
    } catch(e) {}
    </script>
    """
    components.html(js_stop, height=0, width=0)


# Connect Live Assistant speech callback so background scanning speaks aloud automatically
live_assistant.live_assistant.set_speak_callback(speak_text)


def render_audio_controls(text_to_speak: str, key_prefix: str = "audio"):
    """Renders accessible audio player, speak, stop, and repeat buttons."""
    c1, c2, c3 = st.columns([1.5, 1, 1])

    with c1:
        if st.button("🔊 Speak Again (दोबारा बोलें)", key=f"{key_prefix}_speak", use_container_width=True):
            speak_text(text_to_speak)

    with c2:
        if st.button("⏹️ Stop (रोकें)", key=f"{key_prefix}_stop", use_container_width=True):
            stop_all_speech()
            st.info("Audio stopped.")

    with c3:
        if st.button("🔁 Repeat Last", key=f"{key_prefix}_repeat", use_container_width=True):
            last_txt = st.session_state.last_spoken_text
            if last_txt:
                speak_text(last_txt)
            else:
                st.info("No prior response saved.")

    if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
        try:
            with open(st.session_state.last_audio_file, "rb") as af:
                st.audio(af.read(), format="audio/wav")
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Embedded Voice Command & Accessibility Hotkey Listener
# -----------------------------------------------------------------------------
def render_voice_command_listener():
    """
    Renders an HTML5 SpeechRecognition component and global keyboard listeners.
    """
    js_listener = """
    <div style="background: rgba(19, 28, 46, 0.95); border: 1.5px solid #6366F1; border-radius: 12px; padding: 12px 18px; margin: 8px 0; color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div id="micPulse" style="width: 14px; height: 14px; border-radius: 50%; background: #10B981; box-shadow: 0 0 10px #10B981; transition: all 0.3s ease;"></div>
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <strong style="font-size: 1.05rem; color: #F8FAFC; letter-spacing: 0.3px;">🎙️ Voice & Accessibility Control:</strong>
                    <span id="voiceBadge" style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 8px; border-radius: 9999px; font-weight: 700; border: 1px solid #10B981;">ACTIVE</span>
                </div>
                <div id="voiceStatus" style="font-size: 0.88rem; color: #94A3B8; margin-top: 3px;">
                    Say <b>"Start"</b>, <b>"Stop"</b>, <b>"Resume"</b> or press <b>[Space]</b> to Start/Pause, <b>[Esc]</b> to Stop
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 0.75rem; color: #64748B;">Last Action:</span>
            <div id="lastCmd" style="font-size: 0.88rem; font-weight: 700; color: #A5B4FC; background: #0F172A; padding: 5px 12px; border-radius: 8px; border: 1px solid #334155; min-width: 80px; text-align: center;">Ready</div>
        </div>
    </div>

    <style>
    @keyframes activePulse {
        0% { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    #micPulse.pulsing {
        animation: activePulse 1.4s infinite;
    }
    </style>

    <script>
    (function() {
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            document.getElementById('voiceStatus').innerText = "Web SpeechRecognition not supported in this browser.";
            document.getElementById('micPulse').style.background = "#EF4444";
            return;
        }

        var recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        function clickAction(type) {
            try {
                var doc = window.parent.document;
                var buttons = Array.from(doc.querySelectorAll('button'));
                for (var i = 0; i < buttons.length; i++) {
                    var txt = (buttons[i].innerText || buttons[i].textContent || '').toLowerCase().trim();
                    if (type === 'start' && (txt.includes('start assistant') || txt.includes('शुरू करें') || txt.includes('start'))) {
                        buttons[i].click();
                        return true;
                    }
                    if (type === 'stop' && (txt.includes('stop assistant') || txt.includes('रोकें') || txt.includes('stop'))) {
                        buttons[i].click();
                        return true;
                    }
                    if (type === 'resume' && (txt.includes('resume') || txt.includes('फिर से शुरू'))) {
                        buttons[i].click();
                        return true;
                    }
                }
            } catch(e) {}
            return false;
        }

        // Global Keyboard Accessibility (Space: Start/Pause, Esc: Stop, R: Resume)
        try {
            var targetDoc = window.parent.document || document;
            targetDoc.addEventListener('keydown', function(e) {
                var tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
                if (tag === 'input' || tag === 'textarea') return;

                if (e.code === 'Space' || e.key === ' ') {
                    e.preventDefault();
                    var el = document.getElementById('lastCmd');
                    if (el) { el.innerText = '[Space] Start/Pause'; el.style.color = '#38BDF8'; }
                    clickAction('start');
                } else if (e.code === 'Escape' || e.key === 'Escape') {
                    e.preventDefault();
                    var el = document.getElementById('lastCmd');
                    if (el) { el.innerText = '[Esc] Stop'; el.style.color = '#38BDF8'; }
                    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                    clickAction('stop');
                } else if (e.key === 'r' || e.key === 'R') {
                    var el = document.getElementById('lastCmd');
                    if (el) { el.innerText = '[R] Resume'; el.style.color = '#38BDF8'; }
                    clickAction('resume');
                }
            });
        } catch(e) {}

        recognition.onresult = function(event) {
            var last = event.results.length - 1;
            var text = event.results[last][0].transcript.trim().toLowerCase();
            var cmdEl = document.getElementById('lastCmd');
            if (cmdEl) cmdEl.innerText = '"' + text + '"';

            // Check for STOP (English & Hindi)
            if (
                text.includes("stop") || text.includes("pause") || text.includes("halt") ||
                text.includes("roko") || text.includes("band") || text.includes("chup") ||
                text.includes("quiet") || text.includes("silent")
            ) {
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                clickAction('stop');
            }
            // Check for START (English & Hindi)
            else if (
                text.includes("start") || text.includes("begin") || text.includes("scan") ||
                text.includes("shuru") || text.includes("chalu") || text.includes("drishti")
            ) {
                clickAction('start');
            }
            // Check for RESUME (English & Hindi)
            else if (text.includes("resume") || text.includes("continue") || text.includes("phir se")) {
                clickAction('resume');
            }
        };

        recognition.onerror = function(event) {
            var err = event.error;
            // Suppress network error cleanly - do not alarm user with "Mic error: network"
            if (err === 'network') {
                var s = document.getElementById('voiceStatus');
                if (s) s.innerHTML = '🎙️ Offline Mode: Keyboard Controls <b>[Space] Start/Pause</b>, <b>[Esc] Stop</b> active.';
                var b = document.getElementById('voiceBadge');
                if (b) {
                    b.innerText = "HOTKEYS READY";
                    b.style.borderColor = "#6366F1";
                    b.style.color = "#818CF8";
                }
                var p = document.getElementById('micPulse');
                if (p) p.style.background = "#6366F1";
            } else if (err === 'not-allowed') {
                var s = document.getElementById('voiceStatus');
                if (s) s.innerHTML = 'Mic blocked. Use keyboard controls: <b>[Space]</b> Start/Pause, <b>[Esc]</b> Stop.';
            }
        };

        recognition.onend = function() {
            // Auto restart to keep listening continuously
            try { recognition.start(); } catch(e) {}
        };

        try { recognition.start(); } catch(e) {}
    })();
    </script>
    """
    components.html(js_listener, height=65)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Accessibility Quick Controls
# -----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Professional Brand Header
        st.markdown(
            f"""
            <div style="padding: 0.75rem 0.5rem 1.25rem 0.5rem;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 2rem;">👁️</span>
                    <div>
                        <h2 style="font-size: 1.6rem; margin: 0; font-weight: 900; letter-spacing: 1px; color: #FFFFFF !important;">
                            DRISHTI <span style="color: #818CF8;">AI</span>
                        </h2>
                        <p style="font-size: 0.85rem; color: #94A3B8 !important; margin: 0;">{config.APP_TAGLINE}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Language Selection
        st.markdown("### 🌐 Language / भाषा")
        lang_choice = st.radio(
            "Select Language",
            options=["English", "हिन्दी (Hindi)"],
            index=0 if st.session_state.language == "en" else 1,
            label_visibility="collapsed",
        )
        st.session_state.language = "en" if "English" in lang_choice else "hi"

        st.markdown("---")

        # Navigation Menu
        st.markdown("### 🧭 Navigation")
        nav_options = [
            "Camera Assistant",
            "Describe My Surroundings",
            "Read Text (OCR)",
            "Home Dashboard",
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

        # System Readiness Badges
        st.markdown("### 🔌 Offline Status")
        is_ollama_on, models, _ = vision.check_ollama_status(st.session_state.ollama_url)
        tess_status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)

        ollama_badge = '<span class="status-badge badge-online">● Ollama: Ready</span>' if is_ollama_on else '<span class="status-badge badge-offline">○ Ollama: Offline</span>'
        ocr_badge = f'<span class="status-badge badge-online">● OCR: Ready ({tess_status.get("engine", "Active")})</span>' if tess_status["installed"] else '<span class="status-badge badge-warning">○ OCR: Not Found</span>'

        st.markdown(f"{ollama_badge} {ocr_badge}", unsafe_allow_html=True)
        st.caption(f"Active Model: `{st.session_state.model_name}`")


render_sidebar()

# -----------------------------------------------------------------------------
# Section 1: Camera Assistant (Real-Time Live Scanning & Snapshot Modes)
# -----------------------------------------------------------------------------
def render_camera_assistant():
    st.markdown('<h1 class="drishti-h1">🎥 Camera Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.15rem; color: #94A3B8;">Automatic real-time visual assistance with continuous camera scanning, OCR, and voice commands.</p>',
        unsafe_allow_html=True,
    )

    tab_live, tab_radar, tab_snapshot = st.tabs([
        "🟢 Real-Time Live Assistant (Hands-Free)",
        "🚶 Real-Time Obstacle Radar (Assistive Walking)",
        "📸 Photo Snapshot Mode",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Real-Time Live Assistant (Continuous Background Scanning)
    # -------------------------------------------------------------------------
    with tab_live:
        # Full-Duplex Voice Command Listener
        render_voice_command_listener()

        # Operational Status Indicator Card
        state = live_assistant.live_assistant.get_state()
        is_active = state["is_active"]
        is_paused = state["is_paused"]

        # Status Badges
        cam_badge = '<span class="status-badge badge-online">📷 Camera: Active</span>' if is_active else '<span class="status-badge">📷 Camera: Idle</span>'
        scan_badge = '<span class="status-badge badge-online">🔄 Scanning: Active (every 2.5s)</span>' if is_active and not is_paused else ('<span class="status-badge badge-warning">🔄 Scanning: Paused</span>' if is_paused else '<span class="status-badge">🔄 Scanning: Idle</span>')
        mic_badge = '<span class="status-badge badge-online">🎙️ Mic: Listening for Voice Commands</span>'

        # Camera Selector & Status Badges
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            st.markdown(f'<div style="margin-bottom: 0.5rem;">{cam_badge} {scan_badge} {mic_badge}</div>', unsafe_allow_html=True)
        with col_b2:
            st.session_state.live_cam_idx = st.selectbox(
                "Hardware Camera Device:",
                options=[1, 0],
                format_func=lambda x: f"📷 Camera Index {x} ({'Default' if x == 1 else 'Alternate'})",
                key="live_camera_selector",
                label_visibility="collapsed",
            )

        # Main Control Buttons
        c_btn1, c_btn2, c_btn3 = st.columns(3)

        with c_btn1:
            if st.button("▶️ Start Assistant (शुरू करें)", key="btn_live_start", use_container_width=True, type="primary"):
                msg = live_assistant.live_assistant.start(
                    camera_index=st.session_state.get("live_cam_idx", 1),
                    lang=st.session_state.language,
                    scan_interval=st.session_state.live_scan_interval,
                )
                speak_text("Drishti live assistant started. Observing your surroundings.")
                # Give camera worker 1 second to capture the initial frame
                time.sleep(1.2)
                st.rerun()

        with c_btn2:
            st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
            if st.button("⏹️ Stop Assistant (रोकें)", key="btn_live_stop", use_container_width=True):
                msg = live_assistant.live_assistant.stop()
                stop_all_speech()
                st.info("Assistant stopped. Camera scanning and speech halted.")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c_btn3:
            st.markdown('<div class="btn-resume">', unsafe_allow_html=True)
            if st.button("⏯️ Resume Scanning (फिर से शुरू)", key="btn_live_resume", use_container_width=True):
                msg = live_assistant.live_assistant.resume()
                speak_text("Scanning resumed.")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Real-time Visual and Auditory Dashboard
        col_feed, col_info = st.columns([1, 1.2])

        with col_feed:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">📹 Live Camera View</h3>', unsafe_allow_html=True)
            if is_active:
                if state["latest_frame"] is not None:
                    st.image(state["latest_frame"], caption=f"Live Feed ({state['scanning_status']})", use_container_width=True)
                else:
                    st.info("🔄 Initializing camera hardware and connecting live stream...")
            else:
                st.markdown(
                    """
                    <div style="background: #0B0F19; border: 2px dashed #334155; border-radius: 12px; padding: 3rem 1rem; text-align: center; color: #94A3B8;">
                        <span style="font-size: 3rem;">📷</span>
                        <p style="margin-top: 10px; font-weight: 600;">Camera is currently stopped.</p>
                        <p style="font-size: 0.9rem;">Click <b>Start Assistant</b> or press <b>[Spacebar]</b> to begin live scanning.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_info:
            # Currently Detected Objects Pills
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">🎯 Currently Detected Objects</h3>', unsafe_allow_html=True)
            detected = state["detected_objects"]
            if detected:
                pills_html = "".join([f'<span class="object-pill">📦 {obj}</span>' for obj in detected])
                st.markdown(f'<div style="margin: 0.5rem 0;">{pills_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("Waiting for active objects in frame...")

            # Latest Spoken Description Card
            st.markdown('<h3 class="drishti-h3" style="margin-top: 1.25rem;">🗣️ Latest Spoken Description</h3>', unsafe_allow_html=True)
            announcement = state["last_announced"] or state["latest_description"]
            if announcement:
                st.markdown(f'<div class="result-text-box">{announcement}</div>', unsafe_allow_html=True)
                render_audio_controls(announcement, key_prefix="live_announcement_ctrl")

                # Auto-speak in browser immediately when new announcement arrives during live scanning
                if is_active and announcement != st.session_state.get("last_auto_spoken_live"):
                    st.session_state["last_auto_spoken_live"] = announcement
                    trigger_browser_speech(announcement, lang=st.session_state.language)
            else:
                st.info("No descriptions generated yet. Start the assistant to receive live spoken updates.")

            # Visible Text / Signboard OCR
            if state["latest_ocr"]:
                st.markdown('<h3 class="drishti-h3" style="margin-top: 1rem;">📄 Detected Text / Signboard</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="drishti-card-highlight" style="padding: 1rem;">{state["latest_ocr"]}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Scanning Frequency Tuning
        with st.expander("⚙️ Live Scanning Settings & Interval"):
            st.session_state.live_scan_interval = st.slider(
                "Scan Interval (Seconds between AI analysis passes):",
                min_value=float(config.MIN_SCAN_INTERVAL),
                max_value=float(config.MAX_SCAN_INTERVAL),
                value=float(st.session_state.live_scan_interval),
                step=0.5,
                help="2.5 seconds is optimal for 8 GB RAM and CPU inference.",
            )

        # Auto-refresh loop when active to update live frames and announcements
        if is_active:
            time.sleep(1.0)
            st.rerun()

    # -------------------------------------------------------------------------
    # TAB 2: Real-Time Obstacle Radar & Assistive Walking Awareness
    # -------------------------------------------------------------------------
    with tab_radar:
        # Full-Duplex Voice Command Listener
        render_voice_command_listener()

        # Header & Safety Notice
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.4); border-left: 4px solid #6366F1; padding: 10px 14px; border-radius: 6px; margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 0.95rem; color: #E2E8F0;">
                    🚶 <b>Assistive Walking Radar:</b> Evaluates 3 spatial zones (Left, Center, Right), tracks obstacles (chairs, tables, people, doors, stairs), analyzes ground free-space, and announces immediate collision warnings.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Operational Status Badges
        r_model = '<span class="status-badge badge-online">⚡ YOLOv8n + ByteTrack (60ms)</span>'
        r_state = '<span class="status-badge badge-online">🔄 Radar: Scanning</span>' if st.session_state.radar_active else '<span class="status-badge">🔄 Radar: Idle</span>'
        r_alert = '<span class="status-badge badge-online">🔊 Voice Alerts: Active</span>' if st.session_state.radar_auto_alert else '<span class="status-badge badge-warning">🔊 Voice Alerts: Muted</span>'

        c_badges, c_cam = st.columns([2, 1])
        with c_badges:
            st.markdown(f'<div style="margin-bottom: 0.5rem;">{r_model} {r_state} {r_alert}</div>', unsafe_allow_html=True)
        with c_cam:
            st.session_state.radar_cam_idx = st.selectbox(
                "Radar Camera Index:",
                options=[1, 0],
                format_func=lambda x: f"📷 Camera Index {x} ({'Default' if x == 1 else 'Alternate'})",
                key="radar_camera_selector",
                label_visibility="collapsed",
            )

        # Radar Controls
        col_r1, col_r2, col_r3 = st.columns([1, 1, 1.2])
        with col_r1:
            if not st.session_state.radar_active:
                if st.button("▶️ Start Walking Radar (रडार शुरू)", key="btn_radar_start", use_container_width=True, type="primary"):
                    st.session_state.radar_active = True
                    speak_text("Walking obstacle radar activated. Center path monitoring active.")
                    st.rerun()
            else:
                if st.button("⏹️ Stop Radar (रोकें)", key="btn_radar_stop", use_container_width=True):
                    st.session_state.radar_active = False
                    stop_all_speech()
                    st.info("Obstacle radar halted.")
                    st.rerun()

        with col_r2:
            st.session_state.radar_auto_alert = st.toggle(
                "🔊 Auto-Voice Alerts",
                value=st.session_state.radar_auto_alert,
                help="Automatically speaks critical warnings when an obstacle is detected in walking path.",
            )

        with col_r3:
            st.caption("ℹ️ Press **[Spacebar]** to Start/Pause, **[Esc]** to Stop immediately.")

        # Radar Video Feed & Real-Time Metrics
        col_feed, col_data = st.columns([1.2, 1])

        current_frame = None
        radar_result = None

        if st.session_state.radar_active:
            # Capture frame from active camera
            cam_idx = st.session_state.radar_cam_idx
            cap = None
            try:
                # DirectShow first on Windows
                if hasattr(camera.cv2, "CAP_DSHOW"):
                    cap = camera.cv2.VideoCapture(cam_idx, camera.cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    cap = camera.cv2.VideoCapture(cam_idx)
                
                if cap.isOpened():
                    ret, raw_frame = cap.read()
                    if ret and raw_frame is not None:
                        current_frame = raw_frame
                cap.release()
            except Exception:
                if cap:
                    cap.release()

            if current_frame is not None:
                # Process frame through ObstacleAwarenessSystem
                radar_result = obstacle_engine.obstacle_system.process_frame(
                    frame=current_frame,
                    lang=st.session_state.language,
                    enable_tracking=True,
                )

                # Check for critical audio warnings
                if radar_result.critical_alert and st.session_state.radar_auto_alert:
                    if radar_result.critical_alert != st.session_state.last_spoken_obstacle_alert:
                        st.session_state.last_spoken_obstacle_alert = radar_result.critical_alert
                        speak_text(radar_result.critical_alert)

        with col_feed:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">🚶 Spatial Walking Radar View</h3>', unsafe_allow_html=True)
            if st.session_state.radar_active and radar_result is not None:
                rgb_annotated = camera.cv2.cvtColor(radar_result.annotated_frame, camera.cv2.COLOR_BGR2RGB)
                st.image(
                    rgb_annotated,
                    caption=f"Real-Time Spatial Grid ({radar_result.inference_time_ms} ms | YOLOv8n + ByteTrack)",
                    use_container_width=True,
                )
            elif st.session_state.radar_active:
                st.info("🔄 Connecting camera sensor and initializing spatial grid...")
            else:
                st.markdown(
                    """
                    <div style="background: #0B0F19; border: 2px dashed #334155; border-radius: 12px; padding: 3rem 1rem; text-align: center; color: #94A3B8;">
                        <span style="font-size: 3.5rem;">🚶</span>
                        <p style="margin-top: 10px; font-weight: 700; font-size: 1.15rem; color: #F8FAFC;">Walking Radar is Idle</p>
                        <p style="font-size: 0.95rem; color: #94A3B8;">Click <b>Start Walking Radar</b> or press <b>[Spacebar]</b> to begin spatial obstacle detection.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_data:
            # 1. Walkable Corridor Free Space Meters
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">🧭 Walkable Corridor Clearance</h3>', unsafe_allow_html=True)

            c_left = radar_result.corridor.left_free_score if radar_result else 1.0
            c_center = radar_result.corridor.center_free_score if radar_result else 1.0
            c_right = radar_result.corridor.right_free_score if radar_result else 1.0

            st.write(f"**Left Zone (बाएं):** {int(c_left * 100)}% Free Space")
            st.progress(float(c_left))

            st.write(f"**Center Walking Path (सामने का रास्ता):** {int(c_center * 100)}% Free Space")
            st.progress(float(c_center))

            st.write(f"**Right Zone (दाएं):** {int(c_right * 100)}% Free Space")
            st.progress(float(c_right))

            # 2. Active Voice Alert Display Card
            st.markdown('<h3 class="drishti-h3" style="margin-top: 1.25rem;">📢 Active Voice Warning</h3>', unsafe_allow_html=True)
            active_msg = radar_result.critical_alert if (radar_result and radar_result.critical_alert) else st.session_state.last_spoken_obstacle_alert
            if active_msg:
                st.markdown(f'<div class="result-text-box" style="border-color: #EF4444; background: rgba(239, 68, 68, 0.15);">{active_msg}</div>', unsafe_allow_html=True)
                render_audio_controls(active_msg, key_prefix="radar_audio_ctrl")
            else:
                st.success("✅ Path ahead is clear. No collision hazards detected.")

            # 3. Tracked Obstacle Table
            st.markdown('<h3 class="drishti-h3" style="margin-top: 1.25rem;">🎯 Detected Obstacles (Radar)</h3>', unsafe_allow_html=True)
            if radar_result and radar_result.obstacles:
                for obs in radar_result.obstacles:
                    tier_color = "#EF4444" if obs.proximity_tier.value == "critical" else ("#F59E0B" if obs.proximity_tier.value == "warning" else "#10B981")
                    trend_icon = "⚠️ Approaching" if obs.motion_trend.value == "approaching" else ("⏹️ Static" if obs.motion_trend.value == "stationary" else "➡️ Receding")
                    st.markdown(
                        f"""
                        <div style="background: #0F172A; border-left: 4px solid {tier_color}; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <b style="color: #F8FAFC; font-size: 1rem;">📦 {obs.class_name.upper()}</b>
                                <span style="font-size: 0.8rem; color: #94A3B8; margin-left: 8px;">Zone: <b>{obs.zone.value.upper()}</b></span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.85rem; font-weight: 700; color: {tier_color};">{obs.proximity_tier.value.upper()} ({int(obs.relative_depth*100)}%)</span>
                                <span style="font-size: 0.75rem; color: #64748B; margin-left: 6px;">{trend_icon}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No obstacles in frame.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Assistive Safety Disclaimer
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px dashed #475569; border-radius: 8px; padding: 10px 14px; margin-top: 1rem; font-size: 0.85rem; color: #94A3B8;">
                ⚠️ <b>Experimental Prototype Notice:</b> Drishti AI Obstacle Radar is an assistive sensory aid for indoor awareness, NOT a certified orientation and mobility device. Always use a white cane or sighted guide when walking. Never rely solely on AI for collision avoidance in traffic or outdoor environments.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Auto-refresh loop when radar is active for real-time video streaming
        if st.session_state.radar_active:
            time.sleep(0.12)
            st.rerun()

    # -------------------------------------------------------------------------
    # TAB 3: Photo Snapshot Mode (Existing Single Photo Capture Preserved)
    # -------------------------------------------------------------------------
    with tab_snapshot:
        col_t1, col_t2 = st.columns([1.6, 1])
        with col_t1:
            st.session_state.auto_describe_camera = st.toggle(
                "⚡ **Auto-Describe & Speak on Capture (फोटो खींचते ही अपने आप बोलें)**",
                value=st.session_state.auto_describe_camera,
                help="When enabled, taking a photo automatically analyzes the scene and speaks the description aloud immediately!",
            )
        with col_t2:
            cam_lang = st.radio(
                "Snapshot Audio Language:",
                options=["English", "हिन्दी (Hindi)"],
                index=0 if st.session_state.language == "en" else 1,
                horizontal=True,
                key="snapshot_lang_selector",
            )
            selected_lang_code = "en" if "English" in cam_lang else "hi"
            st.session_state.language = selected_lang_code

        captured_frame = None

        cam_file = st.camera_input("Take a photo using your webcam (फोटो खींचें)", key="st_webcam_input_snap")
        if cam_file:
            captured_frame = camera.convert_streamlit_camera_file(cam_file)

        if cam_file and captured_frame and st.session_state.auto_describe_camera:
            cam_bytes = cam_file.getvalue()
            current_hash = hashlib.md5(cam_bytes).hexdigest()

            if st.session_state.last_processed_cam_hash != current_hash:
                st.session_state.last_processed_cam_hash = current_hash
                with st.spinner("🤖 Drishti AI is analyzing your surroundings..."):
                    res = vision.describe_scene(
                        image_input=captured_frame,
                        language=st.session_state.language,
                        model_name=st.session_state.model_name,
                        ollama_url=st.session_state.ollama_url,
                    )
                    st.session_state.last_description_result = res
                    if res["success"]:
                        speak_text(res["description"])

        if captured_frame:
            st.markdown('<div class="drishti-card-highlight">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">⚡ Assistive Actions</h3>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔍 Describe What Camera Sees (दृश्य समझें)", key="btn_cam_describe_snap", use_container_width=True):
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
                if st.button("📖 Read Text in Front of Camera (पाठ पढ़ें)", key="btn_cam_ocr_snap", use_container_width=True):
                    with st.spinner("Reading visible text with OCR..."):
                        res = ocr_reader.extract_text(
                            image_input=captured_frame,
                            language_choice="eng+hin",
                            custom_tesseract_path=st.session_state.custom_tesseract_path,
                        )
                        st.session_state.last_ocr_result = res
                        if res["success"] and res["text"]:
                            speak_text(res["text"])
                            st.success("Text recognized and spoken!")
                        else:
                            st.warning("No readable text detected in current frame.")

            if st.session_state.last_description_result and st.session_state.last_description_result.get("success"):
                desc = st.session_state.last_description_result["description"]
                model = st.session_state.last_description_result.get("model_used", config.DEFAULT_MODEL_NAME)
                st.markdown(f'<h2 class="drishti-h2">🗣️ Scene Description ({model})</h2>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-text-box">{desc}</div>', unsafe_allow_html=True)
                render_audio_controls(desc, key_prefix="cam_desc_snap")

            if st.session_state.last_ocr_result and st.session_state.last_ocr_result.get("success") and st.session_state.last_ocr_result.get("text"):
                ocr_txt = st.session_state.last_ocr_result["text"]
                st.markdown('<h2 class="drishti-h2">📄 Visible Text Content</h2>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-text-box">{ocr_txt}</div>', unsafe_allow_html=True)
                render_audio_controls(ocr_txt, key_prefix="cam_ocr_snap")

            st.markdown("</div>", unsafe_allow_html=True)


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

    tab_upload, tab_sample = st.tabs(["📁 Upload Image (चित्र अपलोड करें)", "🖼️ Use Sample Test Image"])
    active_image = None
    image_source_label = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG, WEBP, BMP)",
            type=config.SUPPORTED_IMAGE_TYPES,
            key="surroundings_upload",
        )
        if uploaded_file:
            try:
                active_image = Image.open(uploaded_file)
                image_source_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Failed to read image file: {str(e)}")

    with tab_sample:
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

    if active_image is not None:
        col_img, col_ctrl = st.columns([1, 1.2])

        with col_img:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.image(active_image, caption=image_source_label, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ctrl:
            st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="drishti-h3">⚙️ Options</h3>', unsafe_allow_html=True)

            analysis_lang = st.radio(
                "Description Language / भाषा:",
                options=["English", "हिन्दी (Hindi)"],
                index=0 if lang == "en" else 1,
                horizontal=True,
                key="surroundings_lang_choice",
            )
            selected_lang_code = "en" if "English" in analysis_lang else "hi"

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
                    else "विशेष रूप से जमीन की सतह, फर्श की बाधाओं, दरवाजों, सीढ़ियों और पैदल चलने के खतरों पर ध्यान दें।"
                )
            elif "Signboards" in focus_preset:
                custom_prompt = (
                    "Transcribe and explain any visible notices, signboards, labels, or text in the scene."
                    if selected_lang_code == "en"
                    else "दृश्य में दिखाई देने वाले किसी भी नोटिस, साइनबोर्ड, लेबल या टेक्स्ट को पढ़कर समझाएं।"
                )

            btn_label = "🔍 दृश्य का विश्लेषण करें (Analyze Scene)" if selected_lang_code == "hi" else "🔍 Analyze Scene with AI"
            if st.button(btn_label, key="btn_run_vision", use_container_width=True):
                with st.spinner("🤖 Drishti AI is analyzing the scene offline..."):
                    result = vision.describe_scene(
                        image_input=active_image,
                        language=selected_lang_code,
                        model_name=st.session_state.model_name,
                        ollama_url=st.session_state.ollama_url,
                        custom_prompt=custom_prompt,
                    )
                    st.session_state.last_description_result = result
                    if result["success"]:
                        speak_text(result["description"])

            st.markdown("</div>", unsafe_allow_html=True)

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
                render_audio_controls(res["description"], key_prefix="vision_res")
                st.markdown("</div>", unsafe_allow_html=True)
                st.info(res["disclaimer"])
            else:
                st.error(f"❌ Analysis failed: {res['error']}")
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

    tess_status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)
    if tess_status["installed"]:
        st.success(f"✅ **OCR Active:** {tess_status['message']}")
    else:
        st.warning(f"⚠️ {tess_status['message']}")

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
                options=["English + Hindi (eng+hin)", "English Only (eng)", "Hindi Only (hin)"],
                index=0,
                key="ocr_lang_target",
            )
            lang_code = "eng+hin" if "Hindi" in ocr_lang_option and "English" in ocr_lang_option else ("hin" if "Hindi" in ocr_lang_option else "eng")

            apply_prep = st.checkbox("Apply Contrast & Denoising Preprocessing", value=True)
            prep_method = st.selectbox(
                "Preprocessing Technique:",
                options=["adaptive (Best for documents/signs)", "otsu (High contrast binary)", "contrast (CLAHE)"],
                index=0,
            )
            prep_key = prep_method.split()[0]

            if st.button("📖 Extract and Read Text (पाठ निकालें)", key="btn_run_ocr", use_container_width=True):
                with st.spinner("⏳ Extracting text with OCR..."):
                    result = ocr_reader.extract_text(
                        image_input=active_image,
                        language_choice=lang_code,
                        apply_preprocessing=apply_prep,
                        preprocess_method=prep_key,
                        custom_tesseract_path=st.session_state.custom_tesseract_path,
                    )
                    st.session_state.last_ocr_result = result

                    if result["success"] and result["text"]:
                        speak_text(result["text"])

            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.last_ocr_result:
            res = st.session_state.last_ocr_result
            st.markdown("---")
            if res["success"]:
                st.markdown('<div class="drishti-card-highlight">', unsafe_allow_html=True)
                st.markdown('<h2 class="drishti-h2" style="margin-top:0;">📄 Extracted Text Content</h2>', unsafe_allow_html=True)

                if res.get("warning"):
                    st.warning(res["warning"])

                if res["text"]:
                    st.markdown(f'<div class="result-text-box">{res["text"]}</div>', unsafe_allow_html=True)
                    st.caption(
                        f"📊 Words: {res['word_count']} | Engine: `{res['language_used']}` | "
                        f"Confidence: {res['confidence']}%"
                    )
                    render_audio_controls(res["text"], key_prefix="ocr_res")
                else:
                    st.info("No text characters were recognized in this image.")

                if res.get("preprocessed_image") and apply_prep:
                    with st.expander("🔍 View Preprocessed Image (OpenCV Filter Output)"):
                        st.image(res["preprocessed_image"], caption="Denoised OCR View", width=450)

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(f"❌ OCR Extraction Error: {res['error']}")
    else:
        st.info("👆 Select or upload an image to read its text.")


# -----------------------------------------------------------------------------
# Section 4: Home Dashboard
# -----------------------------------------------------------------------------
def render_home_dashboard():
    st.markdown(f'<h1 class="drishti-h1">👁️ {config.APP_NAME}</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size: 1.25rem; color: #94A3B8; margin-top: -10px;">{config.APP_TAGLINE} — Offline Accessibility Assistant</p>',
        unsafe_allow_html=True,
    )

    disclaimer = config.SAFETY_DISCLAIMER_HI if st.session_state.language == "hi" else config.SAFETY_DISCLAIMER_EN
    st.markdown(f'<div class="safety-banner">{disclaimer}</div>', unsafe_allow_html=True)

    st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="drishti-h3">🖥️ System Health & Local AI Readiness</h3>', unsafe_allow_html=True)

    is_ollama_on, models, ollama_msg = vision.check_ollama_status(st.session_state.ollama_url)
    tess_status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)

    col1, col2 = st.columns(2)
    with col1:
        if is_ollama_on:
            st.success(f"**Ollama AI:** Online ({len(models)} models: {', '.join(models)})")
            st.caption(f"✅ Active model `{st.session_state.model_name}` ready.")
        else:
            st.error(f"**Ollama AI:** Offline. Run `ollama serve` in terminal.")

    with col2:
        if tess_status["installed"]:
            st.success(f"**OCR Engine:** Ready ({tess_status['engine']})")
            st.caption("✅ English and Hindi text recognition supported.")
        else:
            st.warning("**OCR:** Engine not configured.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<h2 class="drishti-h2">🚀 Quick Assistive Actions</h2>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            """
            <div class="drishti-card-highlight">
                <h3 style="margin-top:0;">🎥 Camera Assistant</h3>
                <p>Live camera scanning with real-time automatic spoken descriptions, obstacle awareness, and voice commands.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Camera Assistant →", key="btn_go_camera_dash", use_container_width=True):
            st.session_state.active_section = "Camera Assistant"
            st.rerun()

    with q2:
        st.markdown(
            """
            <div class="drishti-card-highlight">
                <h3 style="margin-top:0;">📖 Read Text (OCR)</h3>
                <p>Extract and hear text from street signs, medicine packaging, notices, and documents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Text Reader (OCR) →", key="btn_go_ocr_dash", use_container_width=True):
            st.session_state.active_section = "Read Text (OCR)"
            st.rerun()


# -----------------------------------------------------------------------------
# Section 5: Voice Assistant & Speech Center
# -----------------------------------------------------------------------------
def render_voice_assistant():
    st.markdown('<h1 class="drishti-h1">🔊 Voice Assistant & Audio Center</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.15rem; color: #94A3B8;">Auditory synthesis powered by Web Speech API and offline TTS engines.</p>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_test = st.columns([1, 1.2])

    with col_ctrl:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">🎛️ Audio Controls</h3>', unsafe_allow_html=True)

        st.session_state.speech_rate = st.slider(
            "Speech Rate (Words Per Minute):",
            min_value=100,
            max_value=260,
            value=st.session_state.speech_rate,
            step=5,
        )

        st.session_state.speech_volume = st.slider(
            "Volume Level:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.speech_volume,
            step=0.05,
        )

        st.markdown("---")
        last_spoken = st.session_state.last_spoken_text
        if last_spoken:
            st.caption(f"**Last Spoken:** _{last_spoken[:120]}..._")
            c_rep, c_stp = st.columns(2)
            with c_rep:
                if st.button("🔁 Replay Last Response", key="btn_replay_voice_page", use_container_width=True):
                    speak_text(last_spoken)
            with c_stp:
                if st.button("⏹️ Stop Playback", key="btn_stop_voice_page", use_container_width=True):
                    stop_all_speech()
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
        user_speech_input = st.text_area("Enter text to speak aloud:", value=default_txt, height=120, key="voice_test_input")

        if st.button("🔊 Speak Text (बोलें)", key="btn_test_speak", use_container_width=True):
            if user_speech_input.strip():
                speak_text(user_speech_input)
            else:
                st.warning("Please type some text to speak.")

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 6: Settings & Diagnostics
# -----------------------------------------------------------------------------
def render_settings():
    st.markdown('<h1 class="drishti-h1">⚙️ Settings & System Diagnostics</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">🤖 Ollama Model Configuration</h3>', unsafe_allow_html=True)

        st.session_state.ollama_url = st.text_input("Ollama Base URL:", value=st.session_state.ollama_url)
        is_on, installed_models, msg = vision.check_ollama_status(st.session_state.ollama_url)
        st.caption(msg)

        all_model_options = list(set([st.session_state.model_name] + installed_models + config.SUPPORTED_VISION_MODELS))
        st.session_state.model_name = st.selectbox(
            "Selected Vision Model:",
            options=all_model_options,
            index=all_model_options.index(st.session_state.model_name) if st.session_state.model_name in all_model_options else 0,
        )

        if st.button("🔄 Test Ollama Connection", key="btn_test_ollama", use_container_width=True):
            test_on, models_found, test_msg = vision.check_ollama_status(st.session_state.ollama_url)
            if test_on:
                st.success(f"✅ Connection successful! Found {len(models_found)} models: {', '.join(models_found)}")
            else:
                st.error(f"❌ Connection failed: {test_msg}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="drishti-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="drishti-h3">🔍 OCR Configuration</h3>', unsafe_allow_html=True)
        status = ocr_reader.check_tesseract_status(st.session_state.custom_tesseract_path)
        if status["installed"]:
            st.success(f"✅ {status['message']}")
        else:
            st.warning(f"⚠️ {status['message']}")

        st.session_state.custom_tesseract_path = st.text_input(
            "Custom Tesseract.exe Path (Optional):",
            value=st.session_state.custom_tesseract_path,
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 7: About Drishti AI
# -----------------------------------------------------------------------------
def render_about():
    st.markdown('<h1 class="drishti-h1">ℹ️ About DRISHTI AI</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="drishti-card">
            <h3>🌟 Vision & Mission</h3>
            <p>Drishti AI is a real-time, offline-first visual assistant designed to provide independence for visually impaired individuals through local multimodal AI and speech synthesis.</p>
            <ul>
                <li>🔒 <b>100% Offline:</b> No internet required for vision analysis or OCR.</li>
                <li>🗣️ <b>Voice First:</b> Automatic speech readouts in English and Hindi.</li>
                <li>🎙️ <b>Hands-Free Voice Commands:</b> Start, Stop, and Resume via voice.</li>
                <li>📖 <b>Bilingual OCR:</b> Read signage and labels seamlessly.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Section Router
# -----------------------------------------------------------------------------
def main():
    sec = st.session_state.active_section
    if sec == "Camera Assistant":
        render_camera_assistant()
    elif sec == "Describe My Surroundings":
        render_describe_surroundings()
    elif sec == "Read Text (OCR)":
        render_read_text_ocr()
    elif sec == "Home Dashboard":
        render_home_dashboard()
    elif sec == "Voice Assistant":
        render_voice_assistant()
    elif sec == "Settings & Diagnostics":
        render_settings()
    elif sec == "About Drishti AI":
        render_about()
    else:
        render_camera_assistant()


if __name__ == "__main__":
    main()
