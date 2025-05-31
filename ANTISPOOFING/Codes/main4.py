import os
import cv2
import numpy as np
import mediapipe as mp
import time
import torch
import cvzone
import requests
from ultralytics import YOLO

# ---------------- CONFIGURATION ---------------- #
ESP32_UNLOCK_URL = "http://192.168.168.131/unlock"  # Replace with ESP32 IP
ORIGINAL_IMG_PATH = r'D:\ANTISPOOFING\original_img'
REAL_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_real'
FAKE_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_fake'
OUTPUT_PATH = r'D:\ANTISPOOFING\output.jpg'
MODEL_PATH = r"D:\ANTISPOOFING\models\n_version_1_300.pt"
CONFIDENCE_THRESHOLD = 0.75  # Adjust as needed

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
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height


# ----------------- FACE MATCHING FUNCTION ----------------- #
def get_landmarks(image):
    """Extracts facial landmarks using MediaPipe FaceMesh."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


def compare_faces(face_img):
    """Compares the detected face landmarks with stored images."""
    new_landmarks = get_landmarks(face_img)
    if new_landmarks is None:
        return None

    best_match, min_diff = None, float('inf')
    for file in os.listdir(ORIGINAL_IMG_PATH):
        stored_img = cv2.imread(os.path.join(ORIGINAL_IMG_PATH, file))
        stored_landmarks = get_landmarks(stored_img)
        if stored_landmarks is not None:
            diff = np.mean([(s.x - n.x) ** 2 + (s.y - n.y) ** 2 for s, n in zip(stored_landmarks, new_landmarks)])
            if diff < min_diff:
                min_diff, best_match = diff, file

    return best_match if min_diff < 0.01 else None


# ----------------- SEND UNLOCK REQUEST TO ESP32 ----------------- #
def send_unlock_request():
    try:
        response = requests.get(ESP32_UNLOCK_URL, timeout=3)
        if response.status_code == 200:
            print("🔓 Door Unlocked! ✅")
        else:
            print(f"⚠️ Failed to unlock door. Status Code: {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Error: Could not connect to ESP32.")


# ----------------- MAIN PROCESSING LOOP ----------------- #
prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("❌ Error: Unable to capture frame")
        break

    new_frame_time = time.time()
    results = model(img, stream=True, verbose=False, device=device)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = classNames[cls]

            if confidence > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{label.upper()} {int(confidence * 100)}%',
                                   (x1, y1 - 10), scale=2, thickness=4, colorR=color)

                face_img = img[y1:y2, x1:x2]
                if face_img.size == 0:
                    print("❌ Error: Extracted face is empty!")
                    continue

                timestamp = int(time.time())

                if label == "Real":
                    saved_path = os.path.join(REAL_FACE_PATH, f"real_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"✅ Saved Real Face: {saved_path}")

                    match = compare_faces(face_img)
                    if match:
                        print(f"✅ Real Face Matched with: {match}")
                        send_unlock_request()
                else:
                    saved_path = os.path.join(FAKE_FACE_PATH, f"fake_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"⚠️ Fake Face Detected & Saved: {saved_path}")

                    match = compare_faces(face_img)
                    if match:
                        print(f"❌ Fake Face Matched with: {match}")

                    landmarks = get_landmarks(face_img)
                    if landmarks is not None:
                        for lm in landmarks:
                            h, w, _ = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(img, (cx, cy), 1, (0, 0, 255), -1)  # Draw red landmarks for fake faces

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"⚡ FPS: {fps:.2f}")

    cv2.imshow("Face Detection & Anti-Spoofing", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
