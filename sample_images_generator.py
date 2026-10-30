"""
Drishti AI - Sample Test Image Generator
Generates realistic high-contrast test images for offline vision and OCR verification.
"""

from pathlib import Path
from PIL import Image, ImageDraw

def generate_sample_images(target_dir: Path = None):
    """
    Generates test images for immediate demonstration of OCR and Scene Analysis.
    """
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent / "sample_images"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # 1. Emergency Exit & Safety Signboard (Bilingual)
    sign_path = target_dir / "emergency_exit_sign.png"
    try:
        if not sign_path.exists():
            img1 = Image.new("RGB", (820, 420), color=(248, 250, 252))
            d1 = ImageDraw.Draw(img1)
            # Outer Border
            d1.rectangle([(15, 15), (805, 405)], outline=(30, 41, 59), width=4)
            # Header Box (Emergency Red)
            d1.rectangle([(25, 25), (795, 105)], fill=(220, 38, 38))
            d1.text((50, 48), "SAFETY NOTICE / EMERGENCY EXIT", fill=(255, 255, 255))
            # English Instructions
            d1.text((50, 135), "CAUTION: KEEP PATH CLEAR AT ALL TIMES", fill=(15, 23, 42))
            d1.text((50, 185), "DO NOT BLOCK THIS DOORWAY - WHEELCHAIR ACCESSIBLE", fill=(71, 85, 105))
            d1.text((50, 235), "BUILDING A - GROUND FLOOR MAIN CORRIDOR", fill=(15, 23, 42))
            d1.text((50, 285), "IN CASE OF FIRE: USE STAIRS TO REACH ASSEMBLY POINT", fill=(185, 28, 28))
            d1.text((50, 345), "EMERGENCY HELPLINE: 112 / 100 / 108", fill=(2, 132, 199))
            img1.save(sign_path)
    except Exception:
        pass

    # 2. Indoor Room Scene with Spatial Objects & Pathways
    room_path = target_dir / "indoor_room_scene.png"
    try:
        if not room_path.exists():
            img2 = Image.new("RGB", (800, 600), color=(241, 245, 249))
            d2 = ImageDraw.Draw(img2)
            # Wall background
            d2.rectangle([(0, 0), (800, 380)], fill=(226, 232, 240))
            # Floor
            d2.rectangle([(0, 380), (800, 600)], fill=(148, 163, 184))
            # Exit Door on Left
            d2.rectangle([(50, 120), (220, 450)], fill=(180, 83, 9), outline=(120, 53, 15), width=4)
            d2.rectangle([(190, 280), (205, 295)], fill=(245, 158, 11))  # Door handle
            d2.text((70, 240), "EXIT DOOR", fill=(255, 255, 255))
            # Wooden Table in Center
            d2.rectangle([(300, 320), (560, 430)], fill=(101, 67, 33))
            d2.rectangle([(320, 430), (340, 520)], fill=(78, 52, 26))
            d2.rectangle([(520, 430), (540, 520)], fill=(78, 52, 26))
            d2.text((370, 360), "WOODEN TABLE", fill=(255, 255, 255))
            # Potted Plant on Right
            d2.rectangle([(640, 280), (740, 440)], fill=(22, 101, 52))
            d2.rectangle([(660, 440), (720, 500)], fill=(194, 65, 12))  # Pot
            d2.text((660, 350), "PLANT", fill=(255, 255, 255))
            # Caution Pathway marker
            d2.text((310, 550), "WALKING PATH - OBSTACLES AHEAD", fill=(185, 28, 28))
            img2.save(room_path)
    except Exception:
        pass

    # 3. Medical Prescription Label
    med_path = target_dir / "medicine_label.png"
    try:
        if not med_path.exists():
            img3 = Image.new("RGB", (720, 360), color=(255, 255, 255))
            d3 = ImageDraw.Draw(img3)
            d3.rectangle([(15, 15), (705, 345)], outline=(2, 132, 199), width=4)
            d3.text((40, 35), "HEALTHCARE PHARMACY - RX 98214", fill=(2, 132, 199))
            d3.text((40, 85), "PARACETAMOL TABLETS IP 500 MG", fill=(15, 23, 42))
            d3.text((40, 135), "DIRECTIONS: TAKE 1 TABLET TWICE DAILY AFTER MEALS", fill=(15, 23, 42))
            d3.text((40, 185), "WARNING: DO NOT EXCEED 4 DOSES IN 24 HOURS", fill=(220, 38, 38))
            d3.text((40, 235), "PRECAUTION: KEEP OUT OF REACH OF CHILDREN", fill=(185, 28, 28))
            d3.text((40, 285), "EXPIRY: 12/2028 | BATCH: PR-4402", fill=(71, 85, 105))
            img3.save(med_path)
    except Exception:
        pass

    return [sign_path, room_path, med_path]

if __name__ == "__main__":
    generate_sample_images()
