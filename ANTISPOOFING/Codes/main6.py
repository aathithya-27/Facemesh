import os
import cv2
import numpy as np
import mediapipe as mp
import time
import torch
import math
import cvzone
from ultralytics import YOLO
import requests

# ---------------- CONFIGURATION ---------------- #
ORIGINAL_IMG_PATH = r'D:\ANTISPOOFING\original_img'
REAL_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_real'
FAKE_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_fake'
OUTPUT_PATH = r'D:\ANTISPOOFING\output.jpg'
MODEL_PATH = r"D:\ANTISPOOFING\models\n_version_1_300.pt"
CONFIDENCE_THRESHOLD = 0.6

# ESP32-CAM IP Address
ESP32_CAM_IP = "http://192.168.73.131/"  # Replace with your ESP32-CAM IP

# Ensure directories exist
os.makedirs(ORIGINAL_IMG_PATH, exist_ok=True)
os.makedirs(REAL_FACE_PATH, exist_ok=True)
os.makedirs(FAKE_FACE_PATH, exist_ok=True)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Load YOLO Model
model = YOLO(MODEL_PATH).to(device)
classNames = ["Fake", "Real"]

# Load MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Open ESP32-CAM Stream
cap = cv2.VideoCapture(f"{ESP32_CAM_IP}/stream")

# ----------------- FACE LANDMARK EXTRACTION ----------------- #
def get_landmarks(image):
    """Extracts facial landmarks using MediaPipe FaceMesh."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    return None  # No face detected

# ----------------- FACE MATCHING FUNCTION ----------------- #
def compare_faces(face_img):
    """Compares the detected face landmarks with stored images."""
    new_landmarks = get_landmarks(face_img)
    if new_landmarks is None:
        return None  # No landmarks detected in new face
    best_match, min_diff = None, float('inf')
    for file in os.listdir(ORIGINAL_IMG_PATH):
        if file.endswith((".jpg", ".png", ".jpeg")):
            stored_img = cv2.imread(os.path.join(ORIGINAL_IMG_PATH, file))
            stored_landmarks = get_landmarks(stored_img)
            if stored_landmarks is not None:
                diff = np.mean((stored_landmarks - new_landmarks) ** 2)
                if diff < min_diff:
                    min_diff, best_match = diff, file
    return best_match if min_diff < 0.01 else None  # Adjust threshold if needed

# ----------------- RELAY CONTROL FUNCTIONS ----------------- #
def unlock_door():
    """Send HTTP request to unlock the door."""
    response = requests.get(f"{ESP32_CAM_IP}/unlock")
    print(response.text)

def lock_door():
    """Send HTTP request to lock the door."""
    response = requests.get(f"{ESP32_CAM_IP}/lock")
    print(response.text)

# ----------------- MAIN PROCESSING LOOP ----------------- #
prev_frame_time = 0
while True:
    success, img = cap.read()
    if not success:
        print("❌ Error: Unable to capture frame from ESP32-CAM")
        break

    new_frame_time = time.time()
    results = model(img, stream=True, verbose=False, device=device)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = classNames[cls]
            if confidence > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)
                w, h = x2 - x1, y2 - y1
                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{label.upper()} {int(confidence * 100)}%',
                                   (x1, y1 - 10), scale=2, thickness=4, colorR=color)
                # Extract face image
                face_img = img[y1:y2, x1:x2]
                if face_img.size == 0:
                    print("❌ Error: Extracted face is empty!")
                    continue
                timestamp = int(time.time())
                # If a **Real Face** is detected
                if label == "Real":
                    saved_path = os.path.join(REAL_FACE_PATH, f"real_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"✅ Saved Real Face: {saved_path}")
                    # Compare with stored images
                    match = compare_faces(face_img)
                    if match:
                        print(f"✅ Real Face Matched with: {match}")
                        unlock_door()  # Unlock the door
                    else:
                        print("❌ No Match Found - Face Not Recognized!")
                # If a **Fake Face** is detected
                else:
                    saved_path = os.path.join(FAKE_FACE_PATH, f"fake_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"⚠️ Fake Face Detected & Saved: {saved_path}")
                    # Compare Fake Face with stored images
                    fake_match = compare_faces(face_img)
                    if fake_match:
                        print(f"🚨 Fake Face Matched with: {fake_match} - Someone is using a Fake Image!")
                    else:
                        print("⚠️ Fake Face Detected but Not Recognized!")

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"⚡ FPS: {fps:.2f}")

    # Show Live Stream
    cv2.imshow("Face Detection & Anti-Spoofing", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()