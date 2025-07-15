from ultralytics import YOLO
import os
import cv2

model = YOLO("yolov8x-seg.pt")
input_folder = "frames"
output_folder = "segmented_frames"
os.makedirs(output_folder, exist_ok=True)

count = 1
for img in sorted(os.listdir(input_folder)):
    if img.endswith(".jpg"):
        path = os.path.join(input_folder, img)
        results = model.predict(source=path, save=False, stream=True)
        for r in results:
            seg_img = r.plot()
            out_path = os.path.join(output_folder, f"{count:04d}.jpg")
            cv2.imwrite(out_path, seg_img)
            print(f"Processed {img} → {out_path}")
            count += 1
