"""
detect_plate.py — License Plate Region Detection

Strategy:
  1. Find all contours in the edge image
  2. Filter by rectangular shape (approxPolyDP → 4 corners)
  3. Filter by aspect ratio typical of license plates (2:1 to 5:1)
  4. Filter by area (not too small, not too large)
  5. Return the best candidate ROI
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def detect_license_plate(
    original: np.ndarray,
    edges: np.ndarray,
    debug: bool = False
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Detect license plate region in the image.

    Args:
        original : Original BGR image
        edges    : Canny edge image from preprocess.py
        debug    : Save debug images if True

    Returns:
        (plate_crop, (x, y, w, h))  or  None if not found
    """
    # Find all contours
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area descending — plates tend to be sizable regions
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    plate_contour = None

    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon=0.018 * perimeter, closed=True)

        # License plates are rectangles → 4 corners
        if len(approx) != 4:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h

        # --- Filters ---
        # 1. Aspect ratio: plates are wider than tall (typically 2x to 5x)
        aspect_ratio = w / float(h)
        if not (2.0 <= aspect_ratio <= 6.0):
            continue

        # 2. Area: must be a meaningful portion of the image
        image_area = original.shape[0] * original.shape[1]
        if not (0.005 * image_area <= area <= 0.25 * image_area):
            continue

        # 3. Solidity: area of contour vs bounding box (plates are solid rects)
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / float(area)
        if solidity < 0.4:
            continue

        plate_contour = approx
        plate_bbox = (x, y, w, h)
        break  # Take best (largest valid) candidate

    if plate_contour is None:
        return None

    # Crop plate region from original image
    x, y, w, h = plate_bbox
    # Add small padding
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(original.shape[1], x + w + pad)
    y2 = min(original.shape[0], y + h + pad)
    plate_crop = original[y1:y2, x1:x2]

    if debug:
        import os
        os.makedirs("output", exist_ok=True)
        debug_img = original.copy()
        cv2.drawContours(debug_img, [plate_contour], -1, (255, 0, 0), 3)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("output/debug_4_contours.jpg", debug_img)
        cv2.imwrite("output/debug_5_plate_crop.jpg", plate_crop)
        print("      [debug] Saved detection steps to output/")

    return plate_crop, plate_bbox
