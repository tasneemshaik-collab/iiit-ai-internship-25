import cv2
import os
import json
from ultralytics import YOLO
from datetime import datetime
import pygame

# === SETUP ===
ALARM_FRAMES_DIR = "output/alarm_frames_realtime"
ANOMALY_LOG = "output/anomalies_realtime.json"
CONFIDENCE_THRESHOLD = 0.4

# === Load Model ===
model = YOLO("yolov8n.pt")

# === Alarm Setup ===
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# === Directory Prep ===
os.makedirs(ALARM_FRAMES_DIR, exist_ok=True)

# === Capture from Webcam ===
cap = cv2.VideoCapture(0)  # Use 0 for default webcam; use RTSP/URL string for IP cam
if not cap.isOpened():
    raise IOError("[ERROR] Cannot access camera")

anomalies = []
frame_id = 0

print("[INFO] Real-time anomaly detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        anomaly_info = {
            "frame": frame_id,
            "timestamp": timestamp,
            "object": "person"
        }
        anomalies.append(anomaly_info)
        cv2.imwrite(f"{ALARM_FRAMES_DIR}/frame_{frame_id:04d}.jpg", frame)

        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    # === Show Live Video ===
    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# === Clean Up ===
cap.release()
cv2.destroyAllWindows()

with open(ANOMALY_LOG, "w") as f:
    json.dump({"anomalies": anomalies}, f, indent=2)

print(f"[DONE] {len(anomalies)} anomalies detected in real-time.")
print(f"Anomaly frames saved to: {ALARM_FRAMES_DIR}")
