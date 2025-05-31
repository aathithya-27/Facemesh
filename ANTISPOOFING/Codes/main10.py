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
ESP32_UNLOCK_URL = "http://192.168.64.131/unlock"
ESP32_STREAM_URL = "http://192.168.64.131:81/stream"

# Ensure directories exist
os.makedirs(ORIGINAL_IMG_PATH, exist_ok=True)
os.makedirs(REAL_FACE_PATH, exist_ok=True)
os.makedirs(FAKE_FACE_PATH, exist_ok=True)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO Model
model = YOLO(MODEL_PATH).to(device)
classNames = ["Fake", "Real"]

# Load MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3)

# Load OpenCV's Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height
#cap = cv2.VideoCapture(f"{ESP32_STREAM_URL}")
# Variables for eye blink detection
eye_blink_count = 0
last_eye_state = None


# ----------------- FACE LANDMARK EXTRACTION ----------------- #
def get_landmarks(image):
    """Extracts facial landmarks using MediaPipe FaceMesh."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        return [[(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]]
    return None  # No face detected


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
            if x1 <= x <= x2 and y1 <= y <= y2:
                cv2.circle(image, (x, y), 1, color, -1)


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
                diff = np.mean((np.array(stored_landmarks) - np.array(new_landmarks)) ** 2)
                if diff < min_diff:
                    min_diff, best_match = diff, file

    return best_match if min_diff < 0.01 else None  # Adjust threshold if needed


# ----------------- EYE BLINK DETECTION ----------------- #
def detect_eye_blink(face_img):
    """Detects eye blink by checking for eyes."""
    global eye_blink_count, last_eye_state
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)

    if len(eyes) >= 2:
        if last_eye_state is None or last_eye_state == False:
            last_eye_state = True
        return True
    else:
        if last_eye_state is None or last_eye_state == True:
            eye_blink_count += 1
            last_eye_state = False
        return False


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
                landmarks = get_landmarks(face_img)
                draw_landmark_dots(img, landmarks, (255, 0, 0), (x1, y1, x2, y2))

                if label == "Real":
                    saved_path = os.path.join(REAL_FACE_PATH, f"real_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"✅ Saved Real Face: {saved_path}")

                    if landmarks is not None:
                        print(f"✅ Landmarks detected for real face at {saved_path}")

                    # Eye blink detection
                    if detect_eye_blink(face_img):
                        print("✅ Eye blink detected!")
                        eye_blink_count = 0  # Reset blink count
                    else:
                        if eye_blink_count >= 2:  # Adjust threshold as needed
                            print("🚨 No eye blink detected! Possible spoofing attempt.")
                            continue

                    match = compare_faces(face_img)
                    if match:
                        print(f"✅ Real Face Matched with: {match}")
                        send_unlock_request()
                    #if send_unlock_request() == True:
                     #   print("🛑 Stopping the program in 5 seconds...")
                      #  time.sleep(5)
                       # break  # Exit the while loop

                else:
                    saved_path = os.path.join(FAKE_FACE_PATH, f"fake_face_{timestamp}.jpg")
                    cv2.imwrite(saved_path, face_img)
                    print(f"⚠️ Fake Face Detected & Saved: {saved_path}")

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

    # Save output image (optional for debugging)
    cv2.imwrite(OUTPUT_PATH, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
