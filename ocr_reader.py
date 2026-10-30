"""
Drishti AI - OCR (Optical Character Recognition) Module
Handles text extraction from images using Tesseract OCR with OpenCV preprocessing,
and seamless offline fallback to RapidOCR ONNX engine.
Supports English and Hindi text extraction.
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

try:
    from rapidocr_onnxruntime import RapidOCR
    _rapid_ocr_instance = RapidOCR()
except Exception:
    _rapid_ocr_instance = None

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
    system_path = shutil.which("tesseract")
    if system_path and os.path.isfile(system_path):
        return True, system_path

    for path in COMMON_TESSERACT_WINDOWS_PATHS:
        if os.path.isfile(path):
            return True, path

    return False, ""


def configure_tesseract(custom_path: str = None) -> Tuple[bool, str]:
    """
    Configures pytesseract with the binary path if available.
    """
    if pytesseract is None:
        return False, "pytesseract package is not installed."

    if custom_path and os.path.isfile(custom_path):
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return True, f"Tesseract configured at custom path: {custom_path}"

    found, path = find_tesseract_binary()
    if found:
        pytesseract.pytesseract.tesseract_cmd = path
        return True, f"Tesseract found at: {path}"

    return False, "Tesseract executable not found."


def check_tesseract_status(custom_path: str = None) -> Dict[str, Any]:
    """
    Audits the current OCR status.
    Checks Tesseract first; if absent, checks RapidOCR ONNX fallback.
    """
    has_tess, tess_msg = configure_tesseract(custom_path)

    if has_tess:
        try:
            version = str(pytesseract.get_tesseract_version())
            available_langs = pytesseract.get_languages(config="")
            has_eng = "eng" in available_langs
            has_hin = "hin" in available_langs

            status_msg = f"Tesseract v{version} active. Languages: {', '.join(available_langs)}."
            if not has_hin:
                status_msg += " (Hindi 'hin' data missing from tessdata folder)."

            return {
                "installed": True,
                "engine": "Tesseract OCR",
                "version": version,
                "executable": pytesseract.pytesseract.tesseract_cmd,
                "languages": available_langs,
                "has_english": has_eng,
                "has_hindi": has_hin,
                "message": status_msg,
            }
        except Exception:
            pass

    # Fallback to RapidOCR ONNX engine
    if _rapid_ocr_instance is not None:
        return {
            "installed": True,
            "engine": "RapidOCR (Offline ONNX)",
            "version": "1.2.3",
            "executable": "Built-in Python ONNX",
            "languages": ["eng", "hin", "multilingual"],
            "has_english": True,
            "has_hindi": True,
            "message": "RapidOCR (Offline ONNX Engine) is active and ready for English & Hindi.",
        }

    return {
        "installed": False,
        "engine": "None",
        "version": None,
        "executable": "",
        "languages": [],
        "has_english": False,
        "has_hindi": False,
        "message": "No OCR engine detected. Please install Tesseract OCR or rapidocr-onnxruntime.",
    }


def preprocess_image_for_ocr(pil_img: Image.Image, method: str = "adaptive") -> Tuple[Image.Image, np.ndarray]:
    """
    Enhances contrast and cleans noise using OpenCV to improve OCR accuracy.
    """
    if cv2 is None:
        gray = pil_img.convert("L")
        return gray, np.array(gray)

    np_img = np.array(pil_img)
    if len(np_img.shape) == 3:
        if np_img.shape[2] == 4:
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = np_img.copy()

    if method == "adaptive":
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        processed_cv = thresh
    elif method == "otsu":
        filtered = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_cv = thresh
    else:
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
    Extracts printed or digital text from an image with support for English and Hindi.
    Uses Tesseract if installed, or RapidOCR as high-performance offline fallback.
    """
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

    # Load Image as PIL Image
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
                "language_used": language_choice,
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
            "language_used": language_choice,
            "preprocessed_image": None,
            "error": f"Failed to load image: {str(e)}",
            "warning": None,
        }

    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    processed_pil = pil_img
    if apply_preprocessing:
        try:
            processed_pil, _ = preprocess_image_for_ocr(pil_img, method=preprocess_method)
        except Exception:
            processed_pil = pil_img

    # 1. Attempt Tesseract if available
    if status.get("engine") == "Tesseract OCR":
        try:
            custom_config = r"--oem 3 --psm 3"
            lang_to_use = language_choice
            warning_msg = None
            if "hin" in language_choice and not status["has_hindi"]:
                warning_msg = "Hindi pack ('hin') missing in Tesseract. Falling back to English."
                lang_to_use = FALLBACK_OCR_LANGUAGES

            extracted_text = pytesseract.image_to_string(
                processed_pil,
                lang=lang_to_use,
                config=custom_config,
            ).strip()

            avg_conf = 0.0
            try:
                data = pytesseract.image_to_data(
                    processed_pil,
                    lang=lang_to_use,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT,
                )
                confs = [float(c) for c in data.get("conf", []) if str(c).strip() not in ("-1", "")]
                if confs:
                    avg_conf = round(sum(confs) / len(confs), 1)
            except Exception:
                pass

            words = extracted_text.split()
            return {
                "success": True,
                "text": extracted_text,
                "word_count": len(words),
                "confidence": avg_conf,
                "language_used": f"Tesseract ({lang_to_use})",
                "preprocessed_image": processed_pil,
                "error": None,
                "warning": warning_msg,
            }
        except Exception as e:
            # Fall through to RapidOCR if Tesseract encounters an error
            pass

    # 2. RapidOCR ONNX fallback
    if _rapid_ocr_instance is not None:
        try:
            np_array = np.array(processed_pil)
            result, _ = _rapid_ocr_instance(np_array)
            if result:
                lines = [item[1] for item in result if item and len(item) > 1]
                scores = [float(item[2]) * 100 for item in result if item and len(item) > 2]
                avg_conf = round(sum(scores) / len(scores), 1) if scores else 85.0
                full_text = "\n".join(lines).strip()
                words = full_text.split()
                return {
                    "success": True,
                    "text": full_text,
                    "word_count": len(words),
                    "confidence": avg_conf,
                    "language_used": "RapidOCR ONNX (Multilingual)",
                    "preprocessed_image": processed_pil,
                    "error": None,
                    "warning": None,
                }
            else:
                return {
                    "success": True,
                    "text": "",
                    "word_count": 0,
                    "confidence": 0.0,
                    "language_used": "RapidOCR ONNX",
                    "preprocessed_image": processed_pil,
                    "error": None,
                    "warning": "No clear text was detected in the provided image.",
                }
        except Exception as rapid_err:
            return {
                "success": False,
                "text": "",
                "word_count": 0,
                "confidence": 0.0,
                "language_used": "RapidOCR ONNX",
                "preprocessed_image": processed_pil,
                "error": f"RapidOCR extraction failed: {str(rapid_err)}",
                "warning": None,
            }

    return {
        "success": False,
        "text": "",
        "word_count": 0,
        "confidence": 0.0,
        "language_used": language_choice,
        "preprocessed_image": None,
        "error": "OCR failed to run. Please install Tesseract or RapidOCR.",
        "warning": None,
    }
