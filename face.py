import cv2
import face_recognition
import serial
import time

# Setup Serial Communication (Change "COM3" to your correct port)
ser = serial.Serial("COM3", 9600)  # Change COM port as needed
time.sleep(2)  # Give time for the serial connection to establish

# Load known image and encode face
known_image = face_recognition.load_image_file("known_face.jpg")  # Replace with your image
known_encodings = face_recognition.face_encodings(known_image)

if len(known_encodings) > 0:
    known_encoding = known_encodings[0]
else:
    raise ValueError("No face detected in known image!")

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # Convert frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  # Skip this frame if no faces are detected

    # Encode faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known face
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Unknown"

        if True in matches:
            name = "Recognized"
            ser.write(b'1')  # Send "1" to Arduino when face is recognized
            print("Face Recognized! Sending '1' to Arduino...")

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
ser.close()  # Close Serial Connection