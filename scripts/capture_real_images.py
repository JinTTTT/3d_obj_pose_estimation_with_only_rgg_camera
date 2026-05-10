"""
Capture real images from webcam for dataset annotation.

Run with:
    python3 scripts/capture_real_images.py

Controls:
    Space  — save current frame
    q/Esc  — quit
"""

import cv2
from pathlib import Path

CAMERA_INDEX = 2
OUTPUT_DIR   = Path(__file__).resolve().parents[1] / "dataset" / "real" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

webcam = cv2.VideoCapture(CAMERA_INDEX)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

saved_count = len(list(OUTPUT_DIR.glob("*.png")))

print(f"Saving images to: {OUTPUT_DIR}")
print(f"Already saved: {saved_count} images")
print("Press Space to capture, q or Esc to quit.")

while True:
    frame_ok, frame = webcam.read()
    if not frame_ok:
        print("Failed to read frame.")
        break

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = OUTPUT_DIR / f"real_{saved_count:04d}.png"
        cv2.imwrite(str(filename), frame)
        saved_count += 1
        print(f"Saved {filename.name}  (total: {saved_count})")

    elif key in (ord("q"), 27):
        break

webcam.release()
cv2.destroyAllWindows()
print(f"\nDone. Total saved: {saved_count} images.")
