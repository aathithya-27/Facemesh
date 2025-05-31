import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import face_recognition

# Directories
AUTHORIZED_FOLDER = "authorized_faces/"
CAPTURED_FOLDER = "captured_faces/"
MODEL_FOLDER = "models/"
ANTI_SPOOF_MODEL_PATH = "antispoofing_model.tflite"
EMBEDDING_MODEL_PATH = os.path.join(MODEL_FOLDER, "openface_nn4.small2.v1.t7")

# Ensure directories exist
os.makedirs(AUTHORIZED_FOLDER, exist_ok=True)
os.makedirs(CAPTURED_FOLDER, exist_ok=True)

# Load TensorFlow Lite anti-spoofing model
interpreter = tf.lite.Interpreter(model_path=ANTI_SPOOF_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mediapipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load OpenFace model for face embeddings
face_recognizer = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)

# Function: Anti-spoofing check
def check_anti_spoofing(face):
    resized_face = cv2.resize(face, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(resized_face, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction > 0.5  # True if real, False if fake

# Function: Extract embedding from face
def get_face_embedding(face):
    blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    face_recognizer.setInput(blob)
    return face_recognizer.forward()

# Function: Compare embeddings with authorized faces
def match_face(captured_embedding, threshold=0.6):
    for file_name in os.listdir(AUTHORIZED_FOLDER):
        authorized_image_path = os.path.join(AUTHORIZED_FOLDER, file_name)
        authorized_image = cv2.imread(authorized_image_path)
        authorized_embedding = get_face_embedding(authorized_image)

        similarity = cosine_similarity([captured_embedding], [authorized_embedding])[0][0]
        if similarity >= threshold:
            return True, file_name
    return False, None

# Function: Register new user
def register_user(face, user_name):
    save_path = os.path.join(AUTHORIZED_FOLDER, f"{user_name}.jpg")
    cv2.imwrite(save_path, face)
    return save_path

# Streamlit UI
st.title("Advanced Face Lock System")
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Choose an option", ["Home", "Register New User", "Test Face Lock"])

# Webcam or Image Upload
if menu == "Home":
    st.write("Real-time Face Lock System")
    cap = cv2.VideoCapture(0)

    lock_status = st.empty()
    lock_status.warning("Locked: Waiting for face verification...")

    if st.button("Start Webcam"):
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture webcam feed.")
                break

            # Face detection
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, width, height = (int(bbox.xmin * w), int(bbox.ymin * h),
                                           int(bbox.width * w), int(bbox.height * h))

                    face = frame[y:y+height, x:x+width]
                    if face.size == 0:
                        lock_status.warning("Locked: No face detected.")
                        continue

                    # Anti-spoofing
                    if not check_anti_spoofing(face):
                        lock_status.error("Locked: Fake face detected.")
                        continue

                    # Face matching
                    captured_embedding = get_face_embedding(face)
                    match, matched_user = match_face(captured_embedding)
                    if match:
                        lock_status.success(f"Unlocked: Welcome, {matched_user.split('.')[0]}!")
                    else:
                        lock_status.error("Locked: Face not recognized.")

            # Display frame
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        cap.release()

# User Registration
elif menu == "Register New User":
    st.write("Register a New Authorized User")

    user_name = st.text_input("Enter a name for the new user:")
    if not user_name:
        st.warning("Please enter a valid name.")
    else:
        cap = cv2.VideoCapture(0)
        st.write("Capture a clear face for registration:")
        if st.button("Capture Face"):
            ret, frame = cap.read()
            if ret:
                cap.release()
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image")
                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, width, height = (int(bbox.xmin * w), int(bbox.ymin * h),
                                               int(bbox.width * w), int(bbox.height * h))

                        face = frame[y:y+height, x:x+width]
                        if face.size == 0:
                            st.error("Face detection failed. Try again.")
                            break

                        # Register user
                        save_path = register_user(face, user_name)
                        st.success(f"User '{user_name}' registered successfully. Face saved at: {save_path}")
                else:
                    st.error("No face detected in the image. Try again.")
            else:
                st.error("Failed to capture face.")

# Test Face Lock
elif menu == "Test Face Lock":
    st.write("Test the Face Lock System")
    image_file = st.file_uploader("Upload an image:", type=["jpg", "png"])

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x, y, width, height = (int(bbox.xmin * w), int(bbox.ymin * h),
                                       int(bbox.width * w), int(bbox.height * h))

                face = image[y:y+height, x:x+width]
                if face.size == 0:
                    st.error("Face cropping failed. Try again.")
                    break

                # Anti-spoofing
                if not check_anti_spoofing(face):
                    st.error("Fake face detected.")
                else:
                    # Face matching
                    captured_embedding = get_face_embedding(face)
                    match, matched_user = match_face(captured_embedding)
                    if match:
                        st.success(f"Unlocked: Welcome, {matched_user.split('.')[0]}!")
                    else:
                        st.error("Access Denied: Face not recognized.")
        else:
            st.error("No face detected in the image.")
