# 🚗 License Plate Detection — Classical CV

A license plate detection system built with **OpenCV + Tesseract OCR**.  
No GPU required. Works on clean, high-contrast images.

---

## 📁 Project Structure

```
license_plate_detection/
├── main.py           # Full pipeline entry point
├── preprocess.py     # Grayscale → Bilateral Filter → Canny edges
├── detect_plate.py   # Contour detection → Plate region extraction
├── ocr.py            # Plate crop → Tesseract OCR → Clean text
├── requirements.txt
├── images/           # Put your test images here
├── videos/           # Put your test videos here
└── output/           # Results saved here automatically
```

---

## ⚙️ Installation

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (system-level)

**Windows:**  
Download from: https://github.com/UB-Mannheim/tesseract/wiki  
Add install path to system PATH, or set in `ocr.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr
```

---

## 🚀 Usage

### Process a single image
```bash
python main.py images/car.jpg
```

### Process a video file
```bash
python main.py videos/traffic.mp4 --video
```

### Use webcam (live detection)
```bash
python main.py 0 --video
```

---

## 🔄 Pipeline Explained

```
Input Image
    │
    ▼
[preprocess.py]
  • Grayscale conversion
  • Bilateral filter  (noise reduction, edge preservation)
  • Canny edge detection
    │
    ▼
[detect_plate.py]
  • Find all contours
  • Filter by: 4-sided shape, aspect ratio (2:1–6:1), area
  • Extract ROI (Region of Interest)
    │
    ▼
[ocr.py]
  • Upscale plate crop
  • Otsu thresholding (binary image)
  • Tesseract OCR with alphanumeric whitelist
  • Clean & return plate text
    │
    ▼
Output: Annotated image + plate text
```

---

## 📤 Output Files (saved to `output/`)

| File | Description |
|---|---|
| `result.jpg` | Original image with bounding box + plate text |
| `plate_crop.jpg` | Cropped plate region |
| `debug_1_gray.jpg` | Grayscale image |
| `debug_2_filtered.jpg` | After bilateral filter |
| `debug_3_edges.jpg` | Canny edges |
| `debug_4_contours.jpg` | Detected contours highlighted |
| `debug_5_plate_crop.jpg` | Raw plate crop |
| `debug_6_ocr_input.jpg` | Binary image fed to Tesseract |

*(Debug files are saved only when `debug=True` in `process_image()`)*

---

## 🧪 Tips for Best Results

- Use **clear, well-lit images** with visible plate contrast
- Plates should occupy a **reasonable portion** of the image
- If OCR is inaccurate, try adjusting `--psm` in `ocr.py`:
  - `--psm 8` — single word (default)
  - `--psm 7` — single text line
  - `--psm 6` — uniform block of text

---

## 📊 Evaluation

Test your system with:
- ✅ Different lighting conditions
- ✅ Various distances/zoom levels
- ✅ Different countries' plate formats
- ✅ Angled plates (slight tilt)

---

## 🔮 Possible Extensions

- [ ] Multiple plate detection per image
- [ ] Database logging (SQLite)
- [ ] Web UI with Flask
- [ ] Upgrade to YOLOv8 for harder cases
