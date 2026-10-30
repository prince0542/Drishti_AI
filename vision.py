"""
Drishti AI - Vision & Local AI Integration Module
Handles communication with Ollama, image encoding, and scene understanding
using Gemma 4 / local multimodal models.
"""

import base64
import io
import time
from typing import Dict, Any, List, Tuple, Union
from PIL import Image
import requests

from config import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_MODEL_NAME,
    OLLAMA_REQUEST_TIMEOUT,
    VISION_PROMPT_EN,
    VISION_PROMPT_HI,
    SAFETY_DISCLAIMER_EN,
    SAFETY_DISCLAIMER_HI,
)


def check_ollama_status(ollama_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, List[str], str]:
    """
    Checks if Ollama daemon is reachable and lists all installed local models.
    
    Returns:
        (is_running: bool, installed_models: list of model names, status_message: str)
    """
    clean_url = ollama_url.rstrip("/")
    try:
        response = requests.get(f"{clean_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models_data = data.get("models", [])
            installed_models = [m.get("name", "") for m in models_data if "name" in m]
            return True, installed_models, f"Ollama is online ({len(installed_models)} models detected)."
        else:
            return False, [], f"Ollama responded with HTTP {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return (
            False,
            [],
            "Cannot connect to Ollama. Ensure the Ollama service is running (Run 'ollama serve' or open the Ollama desktop app).",
        )
    except requests.exceptions.Timeout:
        return False, [], "Connection to Ollama timed out after 5 seconds."
    except Exception as e:
        return False, [], f"Error contacting Ollama: {str(e)}"


def check_model_available(model_name: str, ollama_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, str]:
    """
    Checks if a specific model (or model tag) is installed in Ollama.
    """
    is_running, installed_models, msg = check_ollama_status(ollama_url)
    if not is_running:
        return False, msg

    # Compare exact or base name (e.g., 'gemma4:e2b' or 'gemma4:e2b' in 'gemma4:e2b:latest')
    matched = [m for m in installed_models if model_name.lower() in m.lower() or m.lower() in model_name.lower()]
    if matched:
        return True, f"Model '{matched[0]}' is ready."
    
    return False, (
        f"Model '{model_name}' is not installed in your Ollama library. "
        f"Available models: {', '.join(installed_models) if installed_models else 'None'}. "
        f"To install, run in terminal: ollama pull {model_name}"
    )


def encode_image_to_base64(image_input: Union[str, bytes, Image.Image], max_dimension: int = 1280) -> str:
    """
    Converts an image (file path, raw bytes, or PIL Image) into a base64 encoded string.
    Optionally scales high-resolution images down to max_dimension for faster local AI inference.
    """
    if isinstance(image_input, str):
        # File path
        pil_img = Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Convert to RGB (in case of RGBA or palette images)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize if image is exceptionally large to conserve VRAM/RAM
    width, height = pil_img.size
    if max(width, height) > max_dimension:
        scale = max_dimension / float(max(width, height))
        new_size = (int(width * scale), int(height * scale))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def describe_scene(
    image_input: Union[str, bytes, Image.Image],
    language: str = "en",
    model_name: str = DEFAULT_MODEL_NAME,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    custom_prompt: str = None,
    timeout: int = OLLAMA_REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """
    Sends an image to the local Ollama vision endpoint (/api/chat) and returns
    an accessible, natural language scene description.
    
    Returns:
        dict with keys:
          - success: bool
          - description: str
          - model_used: str
          - duration_seconds: float
          - error: str or None
          - disclaimer: str
    """
    start_time = time.time()
    clean_url = ollama_url.rstrip("/")
    disclaimer = SAFETY_DISCLAIMER_HI if language == "hi" else SAFETY_DISCLAIMER_EN

    # 1. Check Ollama server availability
    is_online, installed_models, status_msg = check_ollama_status(clean_url)
    if not is_online:
        return {
            "success": False,
            "description": "",
            "model_used": model_name,
            "duration_seconds": round(time.time() - start_time, 2),
            "error": f"Ollama is offline: {status_msg}",
            "disclaimer": disclaimer,
        }

    # 2. Encode image
    try:
        base64_image = encode_image_to_base64(image_input)
    except Exception as img_err:
        return {
            "success": False,
            "description": "",
            "model_used": model_name,
            "duration_seconds": round(time.time() - start_time, 2),
            "error": f"Failed to process image: {str(img_err)}",
            "disclaimer": disclaimer,
        }

    # 3. Formulate Prompt
    if custom_prompt:
        prompt_text = custom_prompt
    elif language == "hi":
        prompt_text = VISION_PROMPT_HI
    else:
        prompt_text = VISION_PROMPT_EN

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [base64_image],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,  # Low temperature for factual, grounded visual descriptions
        },
    }

    # 4. Invoke local Ollama /api/chat
    try:
        response = requests.post(
            f"{clean_url}/api/chat",
            json=payload,
            timeout=timeout,
        )

        elapsed = round(time.time() - start_time, 2)

        if response.status_code == 200:
            result_json = response.json()
            message_obj = result_json.get("message", {})
            description = message_obj.get("content", "").strip()

            if not description:
                return {
                    "success": False,
                    "description": "",
                    "model_used": model_name,
                    "duration_seconds": elapsed,
                    "error": "The model returned an empty description.",
                    "disclaimer": disclaimer,
                }

            return {
                "success": True,
                "description": description,
                "model_used": model_name,
                "duration_seconds": elapsed,
                "error": None,
                "disclaimer": disclaimer,
            }

        elif response.status_code == 404:
            return {
                "success": False,
                "description": "",
                "model_used": model_name,
                "duration_seconds": elapsed,
                "error": (
                    f"Model '{model_name}' was not found by Ollama. "
                    f"Please run 'ollama pull {model_name}' in your command prompt, "
                    f"or choose an installed model in Drishti Settings."
                ),
                "disclaimer": disclaimer,
            }
        else:
            try:
                err_data = response.json()
                err_msg = err_data.get("error", response.text)
            except Exception:
                err_msg = response.text

            # Special detection for models without vision weights
            if "does not support images" in err_msg.lower() or "vision" in err_msg.lower():
                err_msg = (
                    f"The specified model '{model_name}' does not appear to support visual/image inputs. "
                    f"Error from Ollama: '{err_msg}'. "
                    f"Recommended vision models: 'llava', 'gemma4:e2b' (with vision), 'moondream', or 'llama3.2-vision'."
                )

            return {
                "success": False,
                "description": "",
                "model_used": model_name,
                "duration_seconds": elapsed,
                "error": f"Ollama returned error: {err_msg}",
                "disclaimer": disclaimer,
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "description": "",
            "model_used": model_name,
            "duration_seconds": round(time.time() - start_time, 2),
            "error": (
                f"Image analysis timed out after {timeout} seconds. "
                "Local inference may require more time on CPU/GPU or a smaller model."
            ),
            "disclaimer": disclaimer,
        }
    except Exception as req_err:
        return {
            "success": False,
            "description": "",
            "model_used": model_name,
            "duration_seconds": round(time.time() - start_time, 2),
            "error": f"Unexpected request failure: {str(req_err)}",
            "disclaimer": disclaimer,
        }
