"""
preprocess.py — Image Preprocessing for License Plate Detection

Steps:
  1. Convert to grayscale
  2. Apply bilateral filter (removes noise, keeps edges)
  3. Apply Canny edge detection
"""

import cv2
import numpy as np


def preprocess_image(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Preprocess image for license plate detection.

    Args:
        image : BGR image loaded with cv2.imread
        debug : If True, saves each step to output/

    Returns:
        edges : Edge-detected grayscale image (uint8)
    """
    # --- Step 1: Grayscale ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Step 2: Bilateral Filter ---
    # Reduces noise while preserving edges better than Gaussian blur
    # d=11: diameter of each pixel neighborhood
    # sigmaColor=17: filter sigma in color space
    # sigmaSpace=17: filter sigma in coordinate space
    filtered = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    # --- Step 3: Canny Edge Detection ---
    # Threshold1=30, Threshold2=200 work well for plate edges
    edges = cv2.Canny(filtered, threshold1=30, threshold2=200)

    if debug:
        import os
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/debug_1_gray.jpg", gray)
        cv2.imwrite("output/debug_2_filtered.jpg", filtered)
        cv2.imwrite("output/debug_3_edges.jpg", edges)
        print("      [debug] Saved preprocessing steps to output/")

    return edges
