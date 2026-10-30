"""
Drishti AI - Camera Module
Handles physical camera device discovery, live frame capture using OpenCV,
and seamless image conversion for accessibility analysis.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def list_available_cameras(max_tested: int = 3) -> List[int]:
    """
    Tests and returns a list of available camera device indices.
    Tests indices safely with short timeout to avoid UI lockup.
    """
    if cv2 is None:
        return []

    available = []
    for idx in range(max_tested):
        # On Windows, cv2.CAP_DSHOW provides fast, reliable camera enumeration
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            # Test reading a single test frame
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def capture_frame_from_camera(
    camera_index: int = 0,
    warmup_frames: int = 5,
    resolution: Tuple[int, int] = (1280, 720),
) -> Dict[str, Any]:
    """
    Captures a high-quality single frame from the specified camera index.
    
    Returns:
        dict:
          - success: bool
          - image: PIL.Image or None
          - frame_array: np.ndarray (OpenCV BGR) or None
          - resolution: tuple (width, height)
          - error: str or None
    """
    if cv2 is None:
        return {
            "success": False,
            "image": None,
            "frame_array": None,
            "resolution": (0, 0),
            "error": "OpenCV (cv2) is not installed. Please install opencv-python.",
        }

    cap = None
    try:
        # Use DirectShow on Windows for instant initialization without freeze
        if hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            return {
                "success": False,
                "image": None,
                "frame_array": None,
                "resolution": (0, 0),
                "error": (
                    f"Could not open camera device at index {camera_index}. "
                    "Ensure your camera is connected, not in use by another app, "
                    "and permissions are enabled in Windows Settings."
                ),
            }

        # Set requested resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Allow auto-exposure and white-balance to stabilize with warmup frames
        frame = None
        for _ in range(max(1, warmup_frames)):
            ret, temp_frame = cap.read()
            if ret:
                frame = temp_frame
            time.sleep(0.04)

        if frame is None or not ret:
            return {
                "success": False,
                "image": None,
                "frame_array": None,
                "resolution": (0, 0),
                "error": "Camera was opened, but failed to capture a valid video frame.",
            }

        # Convert OpenCV BGR to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        actual_h, actual_w = frame.shape[:2]

        return {
            "success": True,
            "image": pil_image,
            "frame_array": frame,
            "resolution": (actual_w, actual_h),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "image": None,
            "frame_array": None,
            "resolution": (0, 0),
            "error": f"Camera hardware capture error: {str(e)}",
        }
    finally:
        if cap is not None and cap.isOpened():
            cap.release()


def convert_streamlit_camera_file(camera_file) -> Optional[Image.Image]:
    """
    Helper function to safely convert a Streamlit st.camera_input UploadedFile
    into an RGB PIL Image ready for AI vision and OCR analysis.
    """
    if camera_file is None:
        return None

    try:
        pil_img = Image.open(camera_file)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return pil_img
    except Exception:
        return None
