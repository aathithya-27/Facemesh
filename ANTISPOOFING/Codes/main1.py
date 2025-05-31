from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch

confidence = 0.8

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the model on the GPU if available
model = YOLO("../models/n_version_1_300.pt")
model.to(device)  # Move the model to GPU

classNames = ["Fake", "Real"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break  # Break the loop if frame not captured

    # Inference on the GPU
    results = model(img, stream=True, verbose=False, device=device)
    for r in results:
        boxes = r.boxes  # Bounding boxes
        for box in boxes:
            # Bounding Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence Score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:
                if classNames[cls].lower() == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                # Annotate bounding box and confidence
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Display the output
    cv2.imshow("Anti-Spoofing Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
