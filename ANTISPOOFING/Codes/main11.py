import os
import cv2
import numpy as np
import mediapipe as mp
import time
import torch
import cvzone
import requests
import logging
import face_recognition
from ultralytics import YOLO

# ---------------- CONFIGURATION ---------------- #
ORIGINAL_IMG_PATH = r'D:\ANTISPOOFING\original_img'
REAL_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_real'
FAKE_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_fake'
OUTPUT_PATH = r'D:\ANTISPOOFING\output.jpg'
MODEL_PATH = r"D:\ANTISPOOFING\models\n_version_1_300.pt"
KNOWN_FACES_PATH = r'D:\ANTISPOOFING\known_faces.npy'
CONFIDENCE_THRESHOLD = 0.6
FACE_MATCH_THRESHOLD = 0.4  # Adjusted for better accuracy
EYE_BLINK_THRESHOLD = 2  # Defined constant for eye blink detection
ESP32_UNLOCK_URL = "http://192.168.64.131/unlock"
ESP32_STREAM_URL = "http://192.168.64.131:81/stream"
BRIGHTNESS_THRESHOLD = 200  # Define a threshold for brightness

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure directories exist
os.makedirs(ORIGINAL_IMG_PATH, exist_ok=True)
os.makedirs(REAL_FACE_PATH, exist_ok=True)
os.makedirs(FAKE_FACE_PATH, exist_ok=True)

# Load known faces
if os.path.exists(KNOWN_FACES_PATH):
    known_faces = np.load(KNOWN_FACES_PATH, allow_pickle=True).item()
else:
    known_faces = {}

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load YOLO Model
try:
    model = YOLO(MODEL_PATH).to(device)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

classNames = ["Fake", "Real"]

# Load MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3)

# Load OpenCV's Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("❌ Error: Unable to open webcam")
    raise Exception("Webcam not accessible")

cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height

# Variables for eye blink detection
eye_blink_count = 0


def is_bright_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > BRIGHTNESS_THRESHOLD


def get_landmarks(face_img):
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None


def detect_eye_blink(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(eyes) > 0


def compare_faces(face_img):
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_face)
    if face_encodings:
        face_encoding = face_encodings[0]
        for name, known_encoding in known_faces.items():
            match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=FACE_MATCH_THRESHOLD)
            if match[0]:
                return name
    return None


def save_new_face(name, face_img):
    face_encoding = face_recognition.face_encodings(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))[0]
    known_faces[name] = face_encoding
    np.save(KNOWN_FACES_PATH, known_faces)
    logging.info(f"✅ New face saved: {name}")


def send_unlock_request():
    try:
        response = requests.get(ESP32_UNLOCK_URL)
        if response.status_code == 200:
            logging.info("✅ Unlock signal sent successfully")
        else:
            logging.error("❌ Failed to send unlock signal")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error sending unlock request: {e}")

# ----------------- MAIN PROCESSING LOOP ----------------- #
while True:
    success, img = cap.read()
    if not success:
        logging.error("❌ Error: Unable to capture frame")
        break

    if is_bright_image(img):
        label = "Real"
    else:
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
                    cvzone.putTextRect(img, f'{label.upper()} {int(confidence * 100)}%', (x1, y1 - 10), scale=2, thickness=4, colorR=color)
                    face_img = img[y1:y2, x1:x2]
                    if face_img.size == 0:
                        logging.error("❌ Error: Extracted face is empty!")
                        continue
                    if label == "Real":
                        match = compare_faces(face_img)
                        if match:
                            logging.info(f"✅ Real Face Matched with: {match}")
                            send_unlock_request()
                        else:
                            new_name = f"User_{int(time.time())}"
                            save_new_face(new_name, face_img)
                    else:
                        logging.warning("⚠ Fake Face Detected!")
cap.release()
cv2.destroyAllWindows()
