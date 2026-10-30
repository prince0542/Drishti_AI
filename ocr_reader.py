"""
Drishti AI - OCR (Optical Character Recognition) Module
Handles text extraction from images using Tesseract OCR with OpenCV preprocessing,
with robust support for English and Hindi text.
"""

import os
import shutil
from typing import Dict, Any, List, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

from config import (
    COMMON_TESSERACT_WINDOWS_PATHS,
    DEFAULT_OCR_LANGUAGES,
    FALLBACK_OCR_LANGUAGES,
)


def find_tesseract_binary() -> Tuple[bool, str]:
    """
    Locates the Tesseract executable on Windows systems.
    Checks environment PATH first, then well-known installation folders.
    """
    # 1. Check system PATH
    system_path = shutil.which("tesseract")
    if system_path and os.path.isfile(system_path):
        return True, system_path

    # 2. Check common Windows installation paths
    for path in COMMON_TESSERACT_WINDOWS_PATHS:
        if os.path.isfile(path):
            return True, path

    return False, ""


def configure_tesseract(custom_path: str = None) -> Tuple[bool, str]:
    """
    Configures pytesseract with the binary path.
    """
    if pytesseract is None:
        return False, "pytesseract Python package is not installed."

    if custom_path and os.path.isfile(custom_path):
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return True, f"Tesseract configured at custom path: {custom_path}"

    found, path = find_tesseract_binary()
    if found:
        pytesseract.pytesseract.tesseract_cmd = path
        return True, f"Tesseract found and configured at: {path}"

    return False, (
        "Tesseract OCR executable was not found. Please install Tesseract from: "
        "https://github.com/UB-Mannheim/tesseract/wiki and ensure it is in your PATH "
        "or configured in Drishti Settings."
    )


def check_tesseract_status(custom_path: str = None) -> Dict[str, Any]:
    """
    Audits the current Tesseract installation, version, and available language packs (eng, hin).
    """
    if pytesseract is None:
        return {
            "installed": False,
            "version": None,
            "executable": "",
            "languages": [],
            "has_english": False,
            "has_hindi": False,
            "message": "pytesseract package is missing. Run: pip install pytesseract",
        }

    success, msg = configure_tesseract(custom_path)
    if not success:
        return {
            "installed": False,
            "version": None,
            "executable": "",
            "languages": [],
            "has_english": False,
            "has_hindi": False,
            "message": msg,
        }

    try:
        version = str(pytesseract.get_tesseract_version())
        available_langs = pytesseract.get_languages(config="")
        has_eng = "eng" in available_langs
        has_hin = "hin" in available_langs

        status_msg = f"Tesseract v{version} ready. Languages: {', '.join(available_langs)}."
        if not has_hin:
            status_msg += " (Note: Hindi 'hin' language pack not detected. Download hin.traineddata to tessdata)."

        return {
            "installed": True,
            "version": version,
            "executable": pytesseract.pytesseract.tesseract_cmd,
            "languages": available_langs,
            "has_english": has_eng,
            "has_hindi": has_hin,
            "message": status_msg,
        }
    except Exception as e:
        return {
            "installed": False,
            "version": None,
            "executable": getattr(pytesseract.pytesseract, "tesseract_cmd", ""),
            "languages": [],
            "has_english": False,
            "has_hindi": False,
            "message": f"Error running Tesseract: {str(e)}",
        }


def preprocess_image_for_ocr(pil_img: Image.Image, method: str = "adaptive") -> Tuple[Image.Image, np.ndarray]:
    """
    Enhances contrast and cleans noise using OpenCV to improve OCR accuracy.
    
    Methods:
      - 'adaptive': Grayscale + Bilateral filter + Gaussian Adaptive Thresholding
      - 'otsu': Grayscale + Otsu's binarization
      - 'contrast': Grayscale + Contrast Limited Adaptive Histogram Equalization (CLAHE)
    """
    if cv2 is None:
        # Fallback to PIL grayscale if OpenCV is missing
        gray = pil_img.convert("L")
        return gray, np.array(gray)

    # Convert PIL Image to OpenCV BGR/Grayscale format
    np_img = np.array(pil_img)
    if len(np_img.shape) == 3:
        if np_img.shape[2] == 4:
            # RGBA to BGR
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = np_img.copy()

    # Apply preprocessing according to method
    if method == "adaptive":
        # Bilateral filter preserves sharp edges while smoothing background grain
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        processed_cv = thresh
    elif method == "otsu":
        filtered = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_cv = thresh
    else:  # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        processed_cv = clahe.apply(gray)

    processed_pil = Image.fromarray(processed_cv)
    return processed_pil, processed_cv


def extract_text(
    image_input: Union[str, bytes, Image.Image],
    language_choice: str = "eng+hin",
    apply_preprocessing: bool = True,
    preprocess_method: str = "adaptive",
    custom_tesseract_path: str = None,
) -> Dict[str, Any]:
    """
    Extracts printed or digital text from an image with full support for Hindi and English.
    
    Returns:
        dict:
          - success: bool
          - text: str (extracted text)
          - word_count: int
          - confidence: float (0.0 to 100.0)
          - language_used: str
          - preprocessed_image: PIL.Image or None
          - error: str or None
          - warning: str or None
    """
    # 1. Check Tesseract availability
    status = check_tesseract_status(custom_tesseract_path)
    if not status["installed"]:
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "confidence": 0.0,
            "language_used": language_choice,
            "preprocessed_image": None,
            "error": status["message"],
            "warning": None,
        }

    # 2. Determine OCR language fallback
    warning_msg = None
    lang_to_use = language_choice

    if "hin" in language_choice and not status["has_hindi"]:
        warning_msg = (
            "Hindi language pack ('hin') is not installed in Tesseract. "
            "Falling back to English ('eng'). "
            "To enable Hindi OCR, place 'hin.traineddata' in your Tesseract 'tessdata' folder."
        )
        lang_to_use = FALLBACK_OCR_LANGUAGES

    # 3. Load Image as PIL Image
    try:
        if isinstance(image_input, str):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            import io
            pil_img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            return {
                "success": False,
                "text": "",
                "word_count": 0,
                "confidence": 0.0,
                "language_used": lang_to_use,
                "preprocessed_image": None,
                "error": f"Invalid image format: {type(image_input)}",
                "warning": None,
            }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "confidence": 0.0,
            "language_used": lang_to_use,
            "preprocessed_image": None,
            "error": f"Failed to load image: {str(e)}",
            "warning": None,
        }

    # 4. Optional Preprocessing
    processed_pil = pil_img
    if apply_preprocessing:
        try:
            processed_pil, _ = preprocess_image_for_ocr(pil_img, method=preprocess_method)
        except Exception:
            # If preprocessing fails, proceed with raw image
            processed_pil = pil_img

    # 5. Execute OCR
    try:
        # Standard OEM 3 (Default engine), PSM 3 (Fully automatic page segmentation)
        custom_config = r"--oem 3 --psm 3"
        extracted_text = pytesseract.image_to_string(
            processed_pil,
            lang=lang_to_use,
            config=custom_config,
        ).strip()

        # Compute average confidence from word-level data
        avg_confidence = 0.0
        try:
            data = pytesseract.image_to_data(
                processed_pil,
                lang=lang_to_use,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )
            confs = [float(c) for c in data.get("conf", []) if str(c).strip() not in ("-1", "")]
            if confs:
                avg_confidence = round(sum(confs) / len(confs), 1)
        except Exception:
            avg_confidence = 0.0

        # Word count
        words = extracted_text.split()
        word_count = len(words)

        if not extracted_text:
            return {
                "success": True,
                "text": "",
                "word_count": 0,
                "confidence": 0.0,
                "language_used": lang_to_use,
                "preprocessed_image": processed_pil,
                "error": None,
                "warning": warning_msg or "No clear text was detected in the provided image.",
            }

        return {
            "success": True,
            "text": extracted_text,
            "word_count": word_count,
            "confidence": avg_confidence,
            "language_used": lang_to_use,
            "preprocessed_image": processed_pil,
            "error": None,
            "warning": warning_msg,
        }

    except Exception as tesseract_err:
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "confidence": 0.0,
            "language_used": lang_to_use,
            "preprocessed_image": processed_pil,
            "error": f"OCR processing failed: {str(tesseract_err)}",
            "warning": warning_msg,
        }
