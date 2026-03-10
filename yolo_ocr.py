from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # pretrained model

model.train(
    data=r"passport_dataset\dataset.yaml",
    epochs=10,
    imgsz=640,
    batch=1
)           