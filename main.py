"""
app.py — Flask Backend for License Plate Detection
Hosted on PythonAnywhere (free tier)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
import re
import base64
import os

app = Flask(__name__)
CORS(app)  # Allow requests from GitHub Pages frontend

# ── Tesseract path (PythonAnywhere) ──────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# ── Helper: decode base64 image ───────────────────────────────────────────────
def decode_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# ── Helper: encode image to base64 ───────────────────────────────────────────
def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)
    edges    = cv2.Canny(filtered, threshold1=30, threshold2=200)
    return gray, filtered, edges


# ── Detect plate ──────────────────────────────────────────────────────────────
def detect_plate(original, edges):
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        approx    = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)

        if not (4 <= len(approx) <= 6):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        aspect     = w / float(h)

        if not (1.5 <= aspect <= 8.0):
            continue

        img_area = original.shape[0] * original.shape[1]
        if not (0.002 * img_area <= w * h <= 0.35 * img_area):
            continue

        pad   = 5
        plate = original[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
        return plate, (x, y, w, h)

    return None, None


# ── OCR ───────────────────────────────────────────────────────────────────────
def run_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) \
           if len(plate_img.shape) == 3 else plate_img.copy()

    scale = max(2, 100 // gray.shape[0] + 1)
    gray  = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = (r'--oem 3 --psm 7 '
              r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    text = re.sub(r'[^A-Z0-9]', '',
                  pytesseract.image_to_string(binary, config=config).strip().upper())

    if not text:
        inv  = cv2.bitwise_not(binary)
        text = re.sub(r'[^A-Z0-9]', '',
                      pytesseract.image_to_string(inv, config=config).strip().upper())

    return text or 'UNREADABLE', binary


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'project': 'License Plate Detector — Classical CV',
        'author': 'HAMMADFOUZAN',
        'endpoints': ['/detect', '/health']
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode image
        image = decode_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400

        h, w = image.shape[:2]

        # Pipeline
        gray, filtered, edges = preprocess(image)
        plate_img, bbox        = detect_plate(image, edges)

        if plate_img is None:
            # Return edges image even if no plate found
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return jsonify({
                'detected': False,
                'plate_text': 'NOT DETECTED',
                'bbox': None,
                'edges_image': encode_image(edges_rgb),
                'image_size': f'{w}x{h}',
                'message': 'No plate region found. Try a clearer image.'
            })

        # OCR
        plate_text, binary = run_ocr(plate_img)

        # Draw result on original
        x, y, bw, bh = bbox
        result_img   = image.copy()
        cv2.rectangle(result_img, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
        cv2.putText(result_img, plate_text, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Encode outputs
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return jsonify({
            'detected': True,
            'plate_text': plate_text,
            'bbox': {'x': x, 'y': y, 'w': bw, 'h': bh},
            'result_image': encode_image(result_img),
            'plate_crop': encode_image(plate_img),
            'edges_image': encode_image(edges_rgb),
            'image_size': f'{w}x{h}',
            'message': 'Plate detected successfully!'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
