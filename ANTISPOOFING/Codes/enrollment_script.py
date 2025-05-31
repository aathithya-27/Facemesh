import os
import cv2
import mediapipe as mp
import numpy as np

ENROLLMENT_IMAGES_DIR = r'D:\ANTISPOOFING\enrollment_images' # Directory with images of known people
ORIGINAL_IMG_PATH = r'D:\ANTISPOOFING\original_img' # Directory to save landmarks

# Ensure directories exist - CREATE DIRECTORIES AUTOMATICALLY
os.makedirs(ORIGINAL_IMG_PATH, exist_ok=True)
os.makedirs(ENROLLMENT_IMAGES_DIR, exist_ok=True) # Create enrollment directory if it doesn't exist

# Load MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) # Static image mode for enrollment

def get_face_landmarks_from_file(image_path):
    """Extracts facial landmarks from an image file using MediaPipe FaceMesh."""
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        return np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark]) # Return landmark coordinates as numpy array
    return None

for filename in os.listdir(ENROLLMENT_IMAGES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(ENROLLMENT_IMAGES_DIR, filename)
        landmarks = get_face_landmarks_from_file(image_path)
        if landmarks is not None:
            name = os.path.splitext(filename)[0] # Use filename (without extension) as name
            landmark_filename = os.path.join(ORIGINAL_IMG_PATH, name + ".npy")
            np.save(landmark_filename, landmarks)
            print(f"Saved landmarks for {filename} to {landmark_filename}")
        else:
            print(f"Could not extract landmarks for {filename}. No face detected or issue with MediaPipe.")

print("Enrollment process complete.")