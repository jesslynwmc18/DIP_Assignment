# -*- coding: utf-8 -*-
import cv2

from brightness import detect_day_night, adjust_brightness
from face_blur import load_face_detector, blur_faces


INPUT_VIDEO = "input/videos/street.mp4"
OUTPUT_VIDEO = "member1_output.mp4"
CASCADE_FILE = "models/face_detector.xml"


def main():
    
    #detect day/night once 
    is_night = detect_day_night(INPUT_VIDEO)

    #load the Haar Cascade 
    face_cascade = load_face_detector(CASCADE_FILE)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {INPUT_VIDEO}")

    #use the input video's original settings
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #create an output video with same FPS and resolution.
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise ValueError(f"Cannot create output video: {OUTPUT_VIDEO}")

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = adjust_brightness(frame, is_night)
        frame = blur_faces(frame, face_cascade)

        #save the processed colour frame
        writer.write(frame)

        frame_number += 1

        #print progress once every 30 frames
        if frame_number % 30 == 0:
            print(f"Processed {frame_number} frames")

    cap.release()
    writer.release()

    print(f"Done. Saved: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
