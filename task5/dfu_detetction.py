from ultralytics import YOLO
import os
import torch

dataset_path = "dfu_dataset"
yaml_path = os.path.join(dataset_path, "data.yaml")

def detect_gpu():
    if torch.cuda.is_available():
        print(f"Number of GPUs Available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\n--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {round(torch.cuda.memory_allocated(i)/1024**2, 2)} MB")
            print(f"Memory Cached: {round(torch.cuda.memory_reserved(i)/1024**2, 2)} MB")
    else:
        print("No GPU available. Using CPU.")

detect_gpu()

model = YOLO('yolov8n.pt')

results = model.train(
    data = yaml_path,
    epochs = 50,
    imgsz = 512,
    batch = 4,
    name = 'dfu_ulcer_detector'
)
