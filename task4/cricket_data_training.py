import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    print("✅ CUDA is available. GPU detected:", torch.cuda.get_device_name(0))
else:
    print("❌ CUDA not available. Using CPU.")

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml', 
    epochs=20, 
    imgsz=640, 
    batch=4,
    workers=0,
    verbose=True
    )


model = YOLO("runs/detect/train/weights/best.pt")

model.predict(
    source="test_video.mp4",
    conf=0.3,
    save=True,
    save_txt=False
)
