from ultralytics import YOLO
import os
import cv2

model = YOLO("yolov8x-seg.pt")
input_folder = "local_images"
output_folder = "segmented_images"
os.makedirs(output_folder, exist_ok=True)

count = 1
for img in sorted(os.listdir(input_folder)):
    if img.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(input_folder, img)
        results = model.predict(source=path, save=False, stream=True)
        for r in results:
            seg_img = r.plot()
            cv2.imwrite(os.path.join(output_folder, f"{count:04d}.jpg"), seg_img)
            print(f"Segmented {img}")
            count += 1
