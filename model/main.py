import cv2
import random
from ultralytics import YOLO
import numpy as np
from PIL import Image

def prediction(img):
    model = YOLO("model/yolov8s.pt")
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image)
    legend = []
    class_colors = {}
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            class_id = int(class_ids[i])
            confidence = confidences[i]
            class_name = result.names[class_id]
            if class_name not in class_colors:
                color = [random.randint(0, 255) for _ in range(3)]
                class_colors[class_name] = color
                legend.append((class_name, color))
            else:
                color = class_colors[class_name]

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    processed_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return processed_img
