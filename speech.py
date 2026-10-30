"""
Drishti AI - Speech Module (Offline Text-to-Speech)
Provides zero-cloud, 100% offline speech synthesis using pyttsx3.
Supports Windows SAPI5 voices, Hindi voice detection, playback controls,
and audio file generation for the Streamlit interface.
"""

import os
import tempfile
import threading
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from config import (
    TEMP_AUDIO_DIR,
    TTS_DEFAULT_RATE,
    TTS_DEFAULT_VOLUME,
    TTS_AUDIO_FILENAME,
)


class SpeechManager:
    """
    Manages offline speech synthesis, voice selection, thread safety,
    and audio caching for Drishti AI.
    """

    def __init__(self):
        self._last_spoken_text: str = ""
        self._last_audio_path: Optional[str] = None
        self._is_speaking: bool = False
        self._stop_requested: bool = False
        self._active_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def get_installed_voices(self) -> Dict[str, Any]:
        """
        Inspects available Windows TTS voices through SAPI5 / pyttsx3.
        Identifies English and Hindi compatible voices.
        """
        if pyttsx3 is None:
            return {
                "available": False,
                "voices": [],
                "has_hindi": False,
                "hindi_voices": [],
                "default_voice": None,
                "message": "pyttsx3 is not installed. Install via: pip install pyttsx3",
            }

        try:
            # Initialize temporary engine to probe voices
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            voice_list = []
            hindi_voices = []

            for idx, v in enumerate(voices):
                name = getattr(v, "name", f"Voice {idx}")
                vid = getattr(v, "id", str(idx))
                langs = getattr(v, "languages", [])

                # Convert language codes or detect Hindi keywords in voice name
                is_hindi = (
                    "hindi" in name.lower()
                    or "kalpana" in name.lower()
                    or "heera" in name.lower()
                    or "hi-in" in str(langs).lower()
                    or "hi_in" in str(langs).lower()
                )

                voice_info = {
                    "index": idx,
                    "id": vid,
                    "name": name,
                    "languages": langs,
                    "is_hindi": is_hindi,
                }
                voice_list.append(voice_info)
                if is_hindi:
                    hindi_voices.append(voice_info)

            # Clean up engine
            del engine

            has_hin = len(hindi_voices) > 0
            default_voice = voice_list[0]["id"] if voice_list else None

            msg = f"Detected {len(voice_list)} installed offline voice(s)."
            if not has_hin:
                msg += (
                    " No dedicated Hindi voice detected. Windows will use default voice phonetics. "
                    "To install Hindi TTS on Windows: Go to Windows Settings -> Time & Language -> "
                    "Speech -> Add voices -> Select 'Hindi'."
                )

            return {
                "available": True,
                "voices": voice_list,
                "has_hindi": has_hin,
                "hindi_voices": hindi_voices,
                "default_voice": default_voice,
                "message": msg,
            }
        except Exception as e:
            return {
                "available": False,
                "voices": [],
                "has_hindi": False,
                "hindi_voices": [],
                "default_voice": None,
                "message": f"Could not initialize speech engine: {str(e)}",
            }

    def generate_audio_file(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: int = TTS_DEFAULT_RATE,
        volume: float = TTS_DEFAULT_VOLUME,
    ) -> Dict[str, Any]:
        """
        Synthesizes speech directly into a local .wav file.
        This provides rock-solid reliability in Streamlit web interface without
        blocking COM apartments or browser audio playback.
        """
        if not text or not text.strip():
            return {"success": False, "audio_path": None, "error": "No text provided to speak."}

        if pyttsx3 is None:
            return {
                "success": False,
                "audio_path": None,
                "error": "pyttsx3 library is not installed. Please run: pip install pyttsx3",
            }

        clean_text = text.strip()
        timestamp = int(time.time() * 1000)
        try:
            TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            output_file = TEMP_AUDIO_DIR / f"speech_{timestamp}.wav"
        except Exception:
            output_file = Path(tempfile.gettempdir()) / f"speech_{timestamp}.wav"

        try:
            # Create a fresh engine instance for the synthesis
            engine = pyttsx3.init()
            engine.setProperty("rate", max(80, min(300, rate)))
            engine.setProperty("volume", max(0.0, min(1.0, volume)))

            if voice_id:
                try:
                    engine.setProperty("voice", voice_id)
                except Exception:
                    pass  # Fall back to default voice if specific voice ID fails

            engine.save_to_file(clean_text, str(output_file))
            engine.runAndWait()
            del engine

            if output_file.exists() and output_file.stat().st_size > 0:
                with self._lock:
                    self._last_spoken_text = clean_text
                    self._last_audio_path = str(output_file)

                return {
                    "success": True,
                    "audio_path": str(output_file),
                    "text": clean_text,
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "audio_path": None,
                    "error": "TTS engine finished, but the audio file was not written properly.",
                }
        except Exception as e:
            return {
                "success": False,
                "audio_path": None,
                "error": f"Audio synthesis failed: {str(e)}",
            }

    def speak_direct(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: int = TTS_DEFAULT_RATE,
        volume: float = TTS_DEFAULT_VOLUME,
    ) -> bool:
        """
        Speaks text asynchronously via system audio using pyttsx3 in a background thread.
        Allows immediate UI responsiveness and support for stop requests.
        """
        if not text or not text.strip() or pyttsx3 is None:
            return False

        self.stop()  # Stop any active playback

        def _worker():
            with self._lock:
                self._is_speaking = True
                self._stop_requested = False
                self._last_spoken_text = text.strip()

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", rate)
                engine.setProperty("volume", volume)
                if voice_id:
                    engine.setProperty("voice", voice_id)

                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception:
                pass
            finally:
                with self._lock:
                    self._is_speaking = False

        self._active_thread = threading.Thread(target=_worker, daemon=True)
        self._active_thread.start()
        return True

    def stop(self):
        """Signals active speech playback to cease immediately."""
        with self._lock:
            self._stop_requested = True
            self._is_speaking = False

    def get_last_spoken_text(self) -> str:
        """Retrieves the most recent text spoken or synthesized."""
        with self._lock:
            return self._last_spoken_text

    def get_last_audio_path(self) -> Optional[str]:
        """Retrieves path of the last synthesized .wav audio file."""
        with self._lock:
            return self._last_audio_path

    def is_speaking(self) -> bool:
        """Returns True if background speech worker is currently outputting audio."""
        with self._lock:
            return self._is_speaking


# Global instance
speech_manager = SpeechManager()
