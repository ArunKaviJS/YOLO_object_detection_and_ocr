from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
print("model loading")
model = YOLO(r"runs/detect/train/weights/best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

image = cv2.imread("download (27) (2).png")

results = model(image)
print("results")
for r in results:   
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        crop = image[y1:y2, x1:x2]

        result = ocr.ocr(crop)

        text = ""
        if result:
            text = result[0][1][0]

        print("Class:", int(cls))
        print("Text:", text)
        print("----------------")