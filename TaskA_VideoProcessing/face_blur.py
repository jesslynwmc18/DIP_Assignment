"""
face_blur.py

This file contains functions for:
1. Loading the supplied Haar Cascade face detector.
2. Detecting faces in each video frame.
3. Blurring detected faces.

The functions are designed to be imported and used in the main video pipeline.
"""

import cv2

# Load the supplied Haar Cascade face detector XML file.
def load_face_detector(cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Stop the program if the cascade file cannot be loaded.
    if face_cascade.empty():
        raise ValueError(f"Cannot load cascade file: {cascade_path}")

    return face_cascade


# Detect and blur all faces found in a single video frame.
def blur_faces(frame, face_cascade):
    
    # Copy the original frame so it is not directly modified.
    processed_frame = frame.copy()

    # Convert the colour frame to grayscale for face detection.
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces at different sizes in the grayscale frame.
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    # Process every detected face location.
    for (x, y, w, h) in faces:
        # Extract the face region from the colour frame.
        face_region = processed_frame[y:y + h, x:x + w]

        # Set an odd blur size based on the detected face width.
        blur_size = max(15, (w // 3) | 1)

        # Apply Gaussian Blur to hide the face.
        blurred_face = cv2.GaussianBlur(
            face_region,
            (blur_size, blur_size),
            0
        )

        # Replace the original face region with the blurred version.
        processed_frame[y:y + h, x:x + w] = blurred_face

    # Return the final colour frame for the next processing step.
    return processed_frame
