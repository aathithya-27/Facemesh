import os
import cv2
import numpy as np
import time
import torch
import requests
import cvzone
from ultralytics import YOLO

# ---------------- CONFIGURATION ---------------- #
ESP32_IP = "192.168.73.131"  # Change this to your ESP32 IP
MODEL_PATH = r"D:\ANTISPOOFING\models\n_version_1_300.pt"
CONFIDENCE_THRESHOLD = 0.6

ORIGINAL_IMG_PATH = r'D:\ANTISPOOFING\original_img'
os.makedirs(ORIGINAL_IMG_PATH, exist_ok=True)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Load YOLO Model
model = YOLO(MODEL_PATH).to(device)

# Open Laptop Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height

# ----------------- FUNCTION TO SEND LOGS TO ESP32-CAM ----------------- #
def send_log_to_esp32(message):
    try:
        url = f"http://{ESP32_IP}/log?msg={message}"
        requests.get(url, timeout=2)
    except:
        print("⚠️ Could not send log to ESP32-CAM")

# ----------------- FACE MATCHING FUNCTION ----------------- #
def compare_faces(face_img):
    """Compares detected faces with stored images and returns name."""
    best_match, min_diff = None, float('inf')

    for file in os.listdir(ORIGINAL_IMG_PATH):
        if file.endswith((".jpg", ".png", ".jpeg")):
            stored_img = cv2.imread(os.path.join(ORIGINAL_IMG_PATH, file))

            # Ensure both images have the same size
            stored_img = cv2.resize(stored_img, (face_img.shape[1], face_img.shape[0]))

            # Compute Pixel Difference
            diff = np.sum(cv2.absdiff(stored_img, face_img))

            if diff < min_diff:
                min_diff, best_match = diff, os.path.splitext(file)[0]  # Get name from file

    return best_match if min_diff < 10000000 else "Unknown"  # Adjust threshold if needed

# ----------------- MAIN PROCESSING LOOP ----------------- #
prev_frame_time = 0
stream_running = True  # Flag to control stream

while stream_running:
    success, img = cap.read()
    if not success:
        print("❌ Error: Unable to capture frame")
        break

    new_frame_time = time.time()
    results = model(img, stream=True, verbose=False, device=device)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])
            label = "Face"

            if confidence > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # Green for face
                w, h = x2 - x1, y2 - y1

                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{label} {int(confidence * 100)}%',
                                   (x1, y1 - 10), scale=2, thickness=4, colorR=color)

                # Extract face image
                face_img = img[y1:y2, x1:x2]
                if face_img.size == 0:
                    print("❌ Error: Extracted face is empty!")
                    continue

                # Compare face
                person_name = compare_faces(face_img)
                if person_name != "Unknown":
                    log_msg = f"✅ Real Face Matched: {person_name}"
                    print(log_msg)
                    send_log_to_esp32(log_msg)  # Unlock door

                    # Stop the stream after 5 seconds
                    print("⏳ Waiting 5 seconds before stopping stream...")
                    time.sleep(5)
                    print("🛑 Stopping stream now.")
                    stream_running = False  # Exit loop

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"⚡ FPS: {fps:.2f}")

    # Show Live Stream
    cv2.imshow("Face Detection & Door Unlock", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
