import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_ela(image: Image.Image, quality: int = 95) -> Image.Image:
    """
    Perform Error Level Analysis on an image.
    
    ELA works by:
    1. Saving the image at a specified JPEG quality
    2. Reloading it
    3. Computing the absolute difference from the original
    
    This highlights regions with different compression characteristics,
    which can indicate manipulation or AI generation.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")
    
    original = np.array(image, dtype=np.float64)
    resaved = np.array(resaved, dtype=np.float64)
    
    diff = np.abs(original - resaved)
    
    diff_scaled = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    
    ela_image = Image.fromarray(diff_scaled)
    
    return ela_image


def compute_ela_scores(image: Image.Image, quality: int = 95) -> dict:
    """
    Compute ELA statistics for an image.
    Returns mean, std, and max error levels which can be used as features.
    """
    ela_image = compute_ela(image, quality)
    ela_array = np.array(ela_image, dtype=np.float64)
    
    return {
        "ela_mean": float(ela_array.mean()),
        "ela_std": float(ela_array.std()),
        "ela_max": float(ela_array.max()),
        "ela_median": float(np.median(ela_array)),
    }


def create_ela_representation(image: Image.Image, quality: int = 95) -> Image.Image:
    """
    Create a multi-channel ELA representation for CNN input.
    Returns a 3-channel image combining original and ELA information.
    """
    ela = compute_ela(image, quality)
    ela_resized = ela.resize(image.size)
    
    original = image.convert("RGB")
    
    combined = Image.new("RGB", image.size)
    combined.paste(original)
    
    return combined
