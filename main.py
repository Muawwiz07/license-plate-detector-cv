"""
License Plate Detection - Main Pipeline
Classical CV Approach using OpenCV + Tesseract OCR
"""

import cv2
import sys
from preprocess import preprocess_image
from detect_plate import detect_license_plate
from ocr import extract_text


def process_image(image_path: str, debug: bool = False) -> dict:
    """
    Full pipeline: Load → Preprocess → Detect → OCR → Output
    
    Args:
        image_path: Path to input image
        debug: If True, saves intermediate steps
    
    Returns:
        dict with 'plate_image', 'text', 'bbox'
    """
    # Step 1: Load image
    print(f"\n[1/4] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    print(f"      Image shape: {image.shape}")

    # Step 2: Preprocess
    print("[2/4] Preprocessing image...")
    preprocessed = preprocess_image(image, debug=debug)

    # Step 3: Detect plate
    print("[3/4] Detecting license plate...")
    result = detect_license_plate(image, preprocessed, debug=debug)

    if result is None:
        print("      ⚠ No license plate detected.")
        return {"plate_image": None, "text": None, "bbox": None}

    plate_img, bbox = result
    print(f"      ✓ Plate detected at: {bbox}")

    # Step 4: OCR
    print("[4/4] Running OCR...")
    plate_text = extract_text(plate_img, debug=debug)
    print(f"      ✓ Extracted text: '{plate_text}'")

    # Show final result
    output = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(output, plate_text, (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imwrite("output/result.jpg", output)
    cv2.imwrite("output/plate_crop.jpg", plate_img)
    print("\n✅ Results saved to output/")
    print(f"   Detected Plate: {plate_text}")

    return {"plate_image": plate_img, "text": plate_text, "bbox": bbox}


def process_video(video_path: str) -> None:
    """
    Process a video file frame by frame for license plate detection.
    
    Args:
        video_path: Path to video file (or 0 for webcam)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    print(f"\n[VIDEO] Processing: {video_path}")
    print("       Press 'q' to quit\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every 5th frame for performance
        if frame_count % 5 != 0:
            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        preprocessed = preprocess_image(frame)
        result = detect_license_plate(frame, preprocessed)

        if result:
            plate_img, (x, y, w, h) = result
            plate_text = extract_text(plate_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, plate_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video processing complete.")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <image_path>          # Process image")
        print("  python main.py <video_path> --video  # Process video")
        print("  python main.py 0 --video             # Webcam")
        sys.exit(1)

    path = sys.argv[1]
    is_video = "--video" in sys.argv

    if is_video:
        src = int(path) if path == "0" else path
        process_video(src)
    else:
        process_image(path, debug=True)
