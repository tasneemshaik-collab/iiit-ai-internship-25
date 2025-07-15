!nvidia-smi

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

results = model.train(data="african-wildlife.yaml", epochs=20, imgsz=640, batch=8)

#inferencing the best fine-tuned model
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict("https://ultralytics.com/assets/african-wildlife-sample.jpg")
