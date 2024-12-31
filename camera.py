import cv2
import pickle
import numpy as np
import os
from pathlib import Path

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def capture_faces(output_dir='data/', frame_size=(50,50), total_frames=100, capture_interval=2):
    ensure_dir(output_dir)
    
    # Initialize video capture
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Could not open video capture device")
        
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("Error loading face cascade classifier")

    faces_data = []
    frame_count = 0
    name = input("Enter your Aadhar no: ")

    try:
        while len(faces_data) < total_frames:
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                if frame_count % capture_interval == 0:
                    face_img = frame[y:y+h, x:x+w]
                    resized_face = cv2.resize(face_img, frame_size)
                    faces_data.append(resized_face)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            # Display progress
            cv2.putText(frame, f"Captured: {len(faces_data)}/{total_frames}", 
                       (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Capture', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        video.release()
        cv2.destroyAllWindows()

    if faces_data:
        # Process captured faces
        faces_array = np.asarray(faces_data)
        faces_array = faces_array.reshape(len(faces_array), -1)
        
        # Save data
        data_file = os.path.join(output_dir, 'faces_data.pkl')
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                existing_data = pickle.load(f)
            faces_array = np.vstack((existing_data, faces_array))
            
        with open(data_file, 'wb') as f:
            pickle.dump(faces_array, f)
            
        print(f"Successfully captured and saved {len(faces_data)} face images.")
    else:
        print("No faces were captured.")

if __name__ == "__main__":
    capture_faces()