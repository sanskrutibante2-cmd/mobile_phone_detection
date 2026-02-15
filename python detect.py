from ultralytics import YOLO 
import cv2

model = YOLO("yolov8n.pt")

image = cv2.imread("phone.jpg")

results = model(image)

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label == "cell phone": 
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = float(box.conf[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0,  255, 0), 2)
            cv2.putText(image, f"Mobile Detected {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("Mobile Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()