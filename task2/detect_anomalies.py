import cv2
import os
import json
from ultralytics import YOLO
from datetime import timedelta
import pygame

# === SETTINGS ===
FRAME_DIR = "frames2"  # Frames extracted using ffmpeg
ALARM_FRAMES_DIR = "output/alarm_frames2"
OUTPUT_VIDEO = "output/annotated_video.mp4"
ANOMALY_LOG = "output/anomalies.json"
CONFIDENCE_THRESHOLD = 0.4
FPS = 5  # Matches the ffmpeg extraction rate

# === Load YOLOv8 model ===
model = YOLO("yolov8n.pt")  # Use yolov8x.pt for better accuracy

# === Prepare Directories ===
os.makedirs(ALARM_FRAMES_DIR, exist_ok=True)

# === Alarm Setup ===
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# === Read extracted frames ===
frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
if not frame_files:
    raise RuntimeError("No frames found in 'frames2'. Did you run ffmpeg?")

# Get frame dimensions from the first image
first_frame = cv2.imread(os.path.join(FRAME_DIR, frame_files[0]))
frame_height, frame_width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (frame_width, frame_height))

anomalies = []
print("[INFO] Starting detection on frames...")

for idx, fname in enumerate(frame_files):
    frame_path = os.path.join(FRAME_DIR, fname)
    frame = cv2.imread(frame_path)
    results = model(frame)[0]
    detected_person = False

    for result in results.boxes:
        cls = int(result.cls[0])
        conf = float(result.conf[0])
        label = model.names[cls]
        if conf >= CONFIDENCE_THRESHOLD and label == "person":
            detected_person = True
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if detected_person:
        timestamp = str(timedelta(seconds=idx / FPS))
        anomalies.append({
            "frame": idx,
            "timestamp": timestamp,
            "object": "person"
        })

        cv2.imwrite(f"{ALARM_FRAMES_DIR}/frame_{idx:04d}.jpg", frame)

        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    out.write(frame)

out.release()

# === Save anomaly log ===
with open(ANOMALY_LOG, "w") as f:
    json.dump({"anomalies": anomalies}, f, indent=2)

print(f"[DONE] {len(anomalies)} anomalies detected.")
print(f"Annotated video saved to: {OUTPUT_VIDEO}")
