"""
Drishti AI - Live Assistant & Smart Announcement Module
Provides real-time continuous camera scanning in a non-blocking background thread,
intelligent duplicate announcement filtering, OCR + Vision fusion, and
hands-free voice command control.
Optimized for Windows 11 with 8 GB RAM and CPU inference.
"""

import time
import re
import difflib
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import threading
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import config
import vision
import ocr_reader
import speech
import camera


class SmartAnnouncementManager:
    """
    Filters duplicate descriptions, applies cooldowns to repeated objects,
    prioritizes newly recognized text, and creates concise sentences
    specifically crafted for visually impaired listeners.
    """

    def __init__(self, cooldown_seconds: float = 18.0, similarity_threshold: float = 0.65):
        self.cooldown_seconds = cooldown_seconds
        self.similarity_threshold = similarity_threshold
        self._history = deque(maxlen=15)
        self._last_announced_text: str = ""
        self._lock = threading.Lock()

    def should_announce(self, new_text: str) -> bool:
        """
        Determines whether a description contains sufficiently new, useful
        information compared to recent announcements within the cooldown window.
        """
        if not new_text or not new_text.strip():
            return False

        clean_new = new_text.strip().lower()
        now = time.time()

        with self._lock:
            # Check against recent history within cooldown
            for prior_text, timestamp in self._history:
                if now - timestamp < self.cooldown_seconds:
                    matcher = difflib.SequenceMatcher(None, clean_new, prior_text)
                    if matcher.ratio() >= self.similarity_threshold:
                        return False

            # Passed cooldown and uniqueness checks: record and allow
            self._history.append((clean_new, now))
            self._last_announced_text = new_text.strip()
            return True

    def combine_vision_and_ocr(
        self,
        vision_description: str,
        ocr_text: str = None,
        lang: str = "en",
    ) -> str:
        """
        Intelligently combines OCR text and visual scene description into
        a crisp, prioritized statement.
        """
        has_ocr = bool(ocr_text and ocr_text.strip())
        clean_desc = vision_description.strip() if vision_description else ""

        if lang == "hi":
            if has_ocr:
                ocr_clean = ocr_text.strip().replace("\n", " ")
                if len(ocr_clean) > 80:
                    ocr_clean = ocr_clean[:77] + "..."
                if clean_desc:
                    return f"लिखावट दिखी: {ocr_clean}। आसपास: {clean_desc}"
                return f"लिखावट दिखी: {ocr_clean}"
            return clean_desc or "रास्ता साफ है।"
        else:
            if has_ocr:
                ocr_clean = ocr_text.strip().replace("\n", " ")
                if len(ocr_clean) > 80:
                    ocr_clean = ocr_clean[:77] + "..."
                if clean_desc:
                    return f"Text detected: {ocr_clean}. Layout: {clean_desc}"
                return f"Text detected: {ocr_clean}"
            return clean_desc or "Path ahead is clear."

    def clear(self):
        """Resets announcement history and cooldowns."""
        with self._lock:
            self._history.clear()
            self._last_announced_text = ""

    def get_last_announced(self) -> str:
        with self._lock:
            return self._last_announced_text


class LiveScanWorker(threading.Thread):
    """
    Asynchronous background thread that captures camera frames periodically,
    executes OCR + Vision analysis, and delivers spoken descriptions without
    blocking the user interface.
    """

    def __init__(
        self,
        controller: "LiveAssistantController",
        camera_index: int = 1,
        scan_interval: float = 2.5,
        lang: str = "en",
    ):
        super().__init__(daemon=True)
        self.controller = controller
        self.camera_index = camera_index
        self.scan_interval = scan_interval
        self.lang = lang
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self.last_scan_time = 0.0

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def run(self):
        if cv2 is None:
            self.controller.current_status = "Error: OpenCV unavailable"
            self.controller.is_active = False
            return

        cap = None
        try:
            # Auto-detect and open best working camera backend and index
            preferred_indices = [self.camera_index, 0 if self.camera_index != 0 else 1]
            for idx in preferred_indices:
                # 1. Try DirectShow backend
                if hasattr(cv2, "CAP_DSHOW"):
                    test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if test_cap is not None and test_cap.isOpened():
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None:
                            cap = test_cap
                            self.camera_index = idx
                            break
                        test_cap.release()

                # 2. Try default/MSMF backend
                test_cap = cv2.VideoCapture(idx)
                if test_cap is not None and test_cap.isOpened():
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        cap = test_cap
                        self.camera_index = idx
                        break
                    test_cap.release()

            if cap is None or not cap.isOpened():
                self.controller.current_status = "Error: Camera device could not be opened. Check permissions."
                self.controller.camera_status = "Error"
                self.controller.is_active = False
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.controller.camera_status = f"Active (Camera {self.camera_index})"
            self.controller.current_status = "Scanning Active"

            fail_count = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count > 25:
                        # Attempt to re-initialize camera if stalled
                        try:
                            cap.release()
                            cap = cv2.VideoCapture(self.camera_index)
                            fail_count = 0
                        except Exception:
                            pass
                    time.sleep(0.08)
                    continue

                fail_count = 0
                # Store latest frame for UI display
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                self.controller.latest_frame = pil_img

                now = time.time()
                # Check if scanning interval has elapsed and not paused
                if not self._pause_event.is_set() and (now - self.last_scan_time) >= self.scan_interval:
                    if not self.controller.is_analyzing:
                        self.last_scan_time = now
                        announcement = self.controller.process_frame(pil_img, lang=self.lang)
                        if announcement:
                            # Deliver speech announcement
                            self.controller.speak_announcement(announcement)

                time.sleep(0.04)

        except Exception as e:
            self.controller.current_status = f"Scanner error: {str(e)[:40]}"
        finally:
            if cap is not None and cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
            self.controller.camera_status = "Inactive"
            self.controller.current_status = "Stopped"
            self.controller.is_active = False


class LiveAssistantController:
    """
    Singleton manager controlling live camera scanning, speech announcements,
    and voice command interactions.
    """

    def __init__(self):
        self.is_active: bool = False
        self.is_paused: bool = False
        self.is_analyzing: bool = False
        self.scan_interval: float = 2.5
        self.camera_status: str = "Inactive"
        self.current_status: str = "Ready"
        self.microphone_status: str = "Listening for Voice Commands"
        self.latest_detected_objects: List[str] = []
        self.latest_description: str = ""
        self.latest_ocr: str = ""
        self.latest_frame: Optional[Image.Image] = None
        self.announcement_manager = SmartAnnouncementManager()
        self._worker: Optional[LiveScanWorker] = None
        self._lock = threading.Lock()
        self._speak_callback = None

    def set_speak_callback(self, callback):
        self._speak_callback = callback

    def speak_announcement(self, text: str):
        # 1. Immediate offline hardware speech synthesis (100% reliable across background threads)
        try:
            speech.speech_manager.speak_direct(text, rate=config.TTS_DEFAULT_RATE)
        except Exception:
            pass

        # 2. Forward to Streamlit web session callback if available
        if self._speak_callback:
            try:
                self._speak_callback(text)
            except Exception:
                pass

    def start(self, camera_index: int = None, lang: str = "en", scan_interval: float = 2.5) -> str:
        """Starts continuous live scanning in background."""
        with self._lock:
            if self.is_active and self._worker and self._worker.is_alive():
                return "Live assistant is already running."

            self.stop()  # Ensure prior worker is stopped

            # Auto-detect camera index
            if camera_index is None:
                avail = camera.list_available_cameras(max_tested=3)
                camera_index = avail[0] if avail else 1

            self.scan_interval = scan_interval
            self.is_active = True
            self.is_paused = False
            self.current_status = "Starting Scanner..."
            self.announcement_manager.clear()

            self._worker = LiveScanWorker(
                controller=self,
                camera_index=camera_index,
                scan_interval=scan_interval,
                lang=lang,
            )
            self._worker.start()

        return "Live Assistant started. Continuously scanning surroundings."

    def stop(self) -> str:
        """Immediately halts scanning and cancels all speech."""
        with self._lock:
            self.is_active = False
            self.is_paused = False
            self.is_analyzing = False
            self.current_status = "Stopped"
            if self._worker:
                self._worker.stop()
                self._worker = None

        speech.speech_manager.stop()
        return "Assistant stopped. Speech and scanning halted."

    def pause(self) -> str:
        """Pauses frame analysis."""
        with self._lock:
            self.is_paused = True
            self.current_status = "Paused"
            if self._worker:
                self._worker.pause()
        speech.speech_manager.stop()
        return "Scanning paused."

    def resume(self) -> str:
        """Resumes active scanning."""
        with self._lock:
            self.is_active = True
            self.is_paused = False
            self.current_status = "Scanning Active"
            if self._worker:
                self._worker.resume()
        return "Scanning resumed."

    def process_frame(self, image_input: Image.Image, lang: str = "en") -> Optional[str]:
        """Runs fast OCR and Ollama vision analysis on the live frame."""
        self.is_analyzing = True
        self.current_status = "Analyzing Frame..."
        announcement_to_speak = None

        try:
            # 1. Fast OCR pass (using RapidOCR or Tesseract)
            ocr_text = ""
            try:
                ocr_res = ocr_reader.extract_text(
                    image_input=image_input,
                    language_choice="eng+hin" if lang == "hi" else "eng",
                    apply_preprocessing=False,
                )
                if ocr_res.get("success") and ocr_res.get("text"):
                    raw = ocr_res["text"].strip()
                    if len(raw) >= 4:
                        ocr_text = raw
                        self.latest_ocr = ocr_text
            except Exception:
                pass

            # 2. Vision analysis (Concise prompt tailored for live accessibility)
            live_prompt = (
                "You are Drishti AI live assistant for a visually impaired user. "
                "In 1 or 2 concise, clear sentences: state key objects, people, obstacles, or "
                "doors in front of the camera and their position. Keep it brief and directly useful."
                if lang == "en"
                else (
                    "आप दृष्टि एआई लाइव सहायक हैं। दृष्टिबाधित व्यक्ति के लिए 1 या 2 छोटे वाक्यों में "
                    "सामने की मुख्य वस्तुओं, लोगों या बाधाओं की स्थिति बताएं। वाक्य सीधे उपयोगी रखें।"
                )
            )

            vis_res = vision.describe_scene(
                image_input=image_input,
                language=lang,
                model_name=config.DEFAULT_MODEL_NAME,
                ollama_url=config.DEFAULT_OLLAMA_URL,
                custom_prompt=live_prompt,
                timeout=25,
                max_dimension=config.LIVE_IMAGE_WIDTH,
                max_tokens=250,
            )

            desc = ""
            if vis_res.get("success") and vis_res.get("description"):
                desc = vis_res["description"].strip()
                self.latest_description = desc
                self._extract_detected_objects(desc)

            # 3. Combine Vision + OCR and apply smart deduplication
            combined = self.announcement_manager.combine_vision_and_ocr(
                vision_description=desc,
                ocr_text=ocr_text,
                lang=lang,
            )

            if combined and self.announcement_manager.should_announce(combined):
                announcement_to_speak = combined

        except Exception as e:
            self.current_status = f"Analysis error: {str(e)[:30]}"
        finally:
            self.is_analyzing = False
            self.current_status = "Scanning Active" if self.is_active and not self.is_paused else "Idle"

        return announcement_to_speak

    def _extract_detected_objects(self, text: str):
        """Extracts common object tags for accessibility pill badges."""
        common_objects = [
            "person", "people", "bottle", "chair", "table", "desk", "door", "window",
            "computer", "laptop", "phone", "mobile", "bag", "backpack", "stairs",
            "wall", "hallway", "corridor", "book", "cup", "signboard", "sign",
            "कुर्सी", "मेज़", "बोतल", "व्यक्ति", "लोग", "दरवाज़ा", "खिड़की", "कंप्यूटर", "फोन", "बैग"
        ]
        found = []
        lower_txt = text.lower()
        for obj in common_objects:
            if obj in lower_txt and obj not in found:
                found.append(obj.capitalize())
        self.latest_detected_objects = found[:6]

    def get_state(self) -> Dict[str, Any]:
        """Returns the real-time operational state for Streamlit dashboard."""
        return {
            "is_active": self.is_active,
            "is_paused": self.is_paused,
            "is_analyzing": self.is_analyzing,
            "camera_status": self.camera_status,
            "scanning_status": self.current_status,
            "microphone_status": self.microphone_status,
            "detected_objects": list(self.latest_detected_objects),
            "latest_description": self.latest_description,
            "latest_ocr": self.latest_ocr,
            "latest_frame": self.latest_frame,
            "last_announced": self.announcement_manager.get_last_announced(),
        }


def parse_voice_command(raw_transcript: str) -> Optional[str]:
    """
    Parses spoken voice commands for controlling Drishti AI hands-free.
    """
    if not raw_transcript or not raw_transcript.strip():
        return None

    clean = raw_transcript.strip().lower()

    # STOP Commands
    stop_patterns = [
        r"\bstop\b", r"\bpause\b", r"\bhalt\b", r"\bshut up\b",
        r"\broko\b", r"\bbas\b", r"\bruko\b", r"\bरुक\b", r"\bरोक\b", r"\bबंद करो\b"
    ]
    for pat in stop_patterns:
        if re.search(pat, clean):
            return "STOP"

    # RESUME Commands
    resume_patterns = [
        r"\bresume\b", r"\bcontinue\b", r"\bshuru karo dobara\b", r"\bफिर से शुरू\b"
    ]
    for pat in resume_patterns:
        if re.search(pat, clean):
            return "RESUME"

    # START Commands
    start_patterns = [
        r"\bstart\b", r"\bbegin\b", r"\bscan\b", r"\bshuru\b", r"\bshuru karo\b",
        r"\bशुरू\b", r"\bचालू करो\b"
    ]
    for pat in start_patterns:
        if re.search(pat, clean):
            return "START"

    return None


# Global Live Assistant Controller
live_assistant = LiveAssistantController()
