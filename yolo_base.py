from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained
results = model("download (27) (2).png")

for r in results:
    boxes = r.boxes.xyxy