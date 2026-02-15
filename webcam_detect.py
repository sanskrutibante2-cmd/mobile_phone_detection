from ultralytics import YOLO 
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot acess webcam")
    exit()

while True:
    ret, frame = cap.read()
    print(ret)

    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    mobile_found = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "cell phone": 
                mobile_found = True

                coords = box.xyxy[0]
                x1, y1, x2, y2 = map(int, coords) 
                confidence = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,  255, 0), 2)
                cv2.putText(frame, f"Mobile Detected {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if not mobile_found:
            cv2.putText(frame, f"No Mobile Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    cv2.imshow("Webcam Mobile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()