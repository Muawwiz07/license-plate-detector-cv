"""
ocr.py — OCR Text Extraction from License Plate Crop

Steps:
  1. Convert plate crop to grayscale
  2. Upscale for better OCR accuracy
  3. Threshold (Otsu) to create clean black/white image
  4. Apply Tesseract OCR with license-plate-optimized config
  5. Post-process: clean up the text
"""

import cv2
import numpy as np
import pytesseract
import re

# ── Tesseract config ──────────────────────────────────────────────────────────
# --oem 3  : Use LSTM + Legacy engine (best accuracy)
# --psm 8  : Treat image as a single word (good for plates)
#            Use psm 7 for single text line if psm 8 gives poor results
# tessedit_char_whitelist: only allow alphanumeric chars found on plates
TESS_CONFIG = (
    r"--oem 3 --psm 8 "
    r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def prepare_plate_for_ocr(plate_img: np.ndarray) -> np.ndarray:
    """
    Enhance the plate crop for optimal OCR results.

    Args:
        plate_img: BGR or grayscale plate crop

    Returns:
        Binary image ready for Tesseract
    """
    # Convert to grayscale if needed
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()

    # Upscale — Tesseract works best at 300 DPI equivalent
    # Aim for ~100px height minimum
    scale = max(1, 100 // gray.shape[0] + 1)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Otsu thresholding — automatically finds best threshold
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological opening to remove small noise dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def clean_plate_text(raw_text: str) -> str:
    """
    Post-process raw Tesseract output.
    - Strip whitespace
    - Remove non-alphanumeric characters
    - Uppercase

    Args:
        raw_text: Raw string from pytesseract

    Returns:
        Cleaned plate text
    """
    text = raw_text.strip().upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def extract_text(plate_img: np.ndarray, debug: bool = False) -> str:
    """
    Run OCR on a plate crop and return cleaned plate number.

    Args:
        plate_img : BGR/gray plate region
        debug     : Save prepared image if True

    Returns:
        Plate text string (empty string if nothing found)
    """
    binary = prepare_plate_for_ocr(plate_img)

    if debug:
        import os
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/debug_6_ocr_input.jpg", binary)
        print("      [debug] Saved OCR input to output/")

    # Run Tesseract
    raw = pytesseract.image_to_string(binary, config=TESS_CONFIG)
    text = clean_plate_text(raw)

    if not text:
        # Retry with inverted image (dark text on light bg vs light on dark)
        inverted = cv2.bitwise_not(binary)
        raw2 = pytesseract.image_to_string(inverted, config=TESS_CONFIG)
        text = clean_plate_text(raw2)

    return text if text else "UNREADABLE"
