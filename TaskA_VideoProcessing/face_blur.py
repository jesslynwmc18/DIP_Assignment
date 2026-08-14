import cv2

def load_face_detector(cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise ValueError(f"Cannot load cascade file: {cascade_path}")

    return face_cascade


def blur_faces(frame, face_cascade):
    processed_frame = frame.copy()

    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    for (x, y, w, h) in faces:
        face_region = processed_frame[y:y + h, x:x + w]

        blur_size = max(15, (w // 3) | 1)

        blurred_face = cv2.GaussianBlur(
            face_region,
            (blur_size, blur_size),
            0
        )

        processed_frame[y:y + h, x:x + w] = blurred_face

    return processed_frame