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
ORIGINAL_IMG_PATH = r'D:\ANTISPOOFING\original_img'
REAL_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_real'
FAKE_FACE_PATH = r'D:\ANTISPOOFING\detected_faces_fake'
OUTPUT_PATH = r'D:\ANTISPOOFING\output.jpg'
MODEL_PATH = r"D:\ANTISPOOFING\models\n_version_1_300.pt"
CONFIDENCE_THRESHOLD = 0.6
ESP32_UNLOCK_URL = "http://192.168.124.131/unlock" # <<-- IP Address here
FACE_MATCH_THRESHOLD = 0.01 # Adjust threshold for face matching - landmark distance

# Ensure directories exist - CREATE DIRECTORIES AUTOMATICALLY
os.makedirs(ORIGINAL_IMG_PATH, exist_ok=True)
os.makedirs(REAL_FACE_PATH, exist_ok=True)
os.makedirs(FAKE_FACE_PATH, exist_ok=True)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO Model (Anti-spoofing)
model = YOLO(MODEL_PATH).to(device)
classNames = ["Fake", "Real"]

# Load MediaPipe FaceMesh (Landmark detection)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height

# ----------------- FACE LANDMARK EXTRACTION (MediaPipe) ----------------- #
def get_landmarks(image):
    """Extracts facial landmarks using MediaPipe FaceMesh."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        return [[(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]]
    return None  # No face detected by MediaPipe


# ----------------- DRAW DOTTED LANDMARKS ----------------- #
def draw_landmark_dots(image, landmarks, color, bbox):
    """Draws facial landmarks as dots within the bounding box."""
    if landmarks is None:
        return  # Prevents error
    x1, y1, x2, y2 = bbox
    h, w, _ = image.shape
    for face_landmarks in landmarks:
        for lm in face_landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            if x1 <= x <= x2 and y1 <= y2:
                cv2.circle(image, (x, y), 1, color, -1)


# ----------------- FACE MATCHING FUNCTION (using landmark distance) ----------------- #
def compare_faces(face_img):
    """Compares the detected face landmarks with stored landmarks."""
    new_landmarks = get_landmarks(face_img)
    if new_landmarks is None:
        return None  # No landmarks detected in new face

    best_match_name, min_diff = None, float('inf')

    for filename in os.listdir(ORIGINAL_IMG_PATH):
        if filename.endswith((".npy")): # Assuming you save landmarks as .npy files
            known_landmarks = np.load(os.path.join(ORIGINAL_IMG_PATH, filename))
            if known_landmarks is not None and new_landmarks is not None: # Ensure both are not None
                diff = np.mean((np.array(known_landmarks) - np.array(new_landmarks)) ** 2) # Mean squared error of landmarks
                if diff < min_diff:
                    min_diff = diff
                    best_match_name = filename[:-4] # Remove ".npy" extension to get name

    return best_match_name if min_diff < FACE_MATCH_THRESHOLD else None # Adjust threshold


# ----------------- UNLOCK REQUEST FUNCTION ----------------- #
def send_unlock_request():
    """Sends an unlock request to ESP32."""
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
                landmarks = get_landmarks(face_img) # Using MediaPipe landmarks
                draw_landmark_dots(img, landmarks, (255, 0, 0), (x1, y1, x2, y2))

                if label == "Real":
                    saved_path = os.path.join(REAL_FACE_PATH, f"real_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"✅ Saved Real Face: {saved_path}")

                    if landmarks is not None:
                        print(f"✅ Landmarks detected for real face at {saved_path}")

                    matched_name = compare_faces(face_img) # Now using landmark distance comparison
                    if matched_name:
                        print(f"✅ Real Face Matched with: {matched_name}")
                        send_unlock_request()
                    else:
                        print("Real Face Detected, but Not Recognized.") # If landmark is matched, but name is None

                else:
                    saved_path = os.path.join(FAKE_FACE_PATH, f"fake_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"⚠️ Fake Face Detected & Saved: {saved_path}")

                    fake_match = compare_faces(face_img) # Still compare fake faces as well (optional)
                    if fake_match:
                        print(f"🚨 Fake Face Matched with: {fake_match} - Someone is using a Fake Image!")
                    else:
                        print("⚠️ Fake Face Detected but Not Recognized!")
                saved_path = os.path.join(REAL_FACE_PATH if label == "Real" else FAKE_FACE_PATH,
                                          f"{label.lower()}_face_{timestamp}.jpg")

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"⚡ FPS: {fps:.2f}")

    # Show Live Stream
    cv2.imshow("Face Detection & Anti-Spoofing", img)

    # Save output image (optional for debugging)
    cv2.imwrite(OUTPUT_PATH, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()