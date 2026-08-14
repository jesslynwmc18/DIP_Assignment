# -*- coding: utf-8 -*-
import cv2
import numpy as np


def detect_day_night(video_path):
    """Return True for night videos, False for day videos."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(30, total_frames)
    brightness_values = []

    for frame_index in np.linspace(0, total_frames - 1, sample_count, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))

    cap.release()

    average_brightness = np.mean(brightness_values)
    night_threshold = 80
    is_night = average_brightness < night_threshold

    print(f"Average brightness: {average_brightness:.2f}")
    print("Video type:", "NIGHT" if is_night else "DAY")

    return is_night


def adjust_brightness(frame, is_night):
    """Brighten the frame only when the video is a night video."""
    if not is_night:
        return frame

    return cv2.convertScaleAbs(frame, alpha=1.3, beta=30)