"""
brightness.py

This file contains functions for:
1. Determining whether an input video was recorded during day or night.
2. Increasing brightness only for videos classified as nighttime videos.

The functions are designed to be imported and used in the main video pipeline.
"""

import cv2
import numpy as np


def detect_day_night(video_path):
    """
    Detect whether a video is day or night using average frame brightness.
    Returns True for night and False for day.
    """

    # Open the video file.
    cap = cv2.VideoCapture(video_path)

    # Stop the program with a clear message if the video cannot be opened.
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get the total number of frames in the video.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample up to 30 frames to estimate overall video brightness.
    sample_count = min(30, total_frames)
    brightness_values = []

    # Select evenly spaced frames from the start to the end of the video.
    for frame_index in np.linspace(0, total_frames - 1, sample_count, dtype=int):
        # Move to the selected frame and read it.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if success:
            # Convert the colour frame to grayscale for brightness calculation.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Store the average pixel brightness of this frame.
            brightness_values.append(np.mean(gray))

    # Close the video after frame sampling is complete.
    cap.release()

    # Calculate the average brightness across all sampled frames.
    average_brightness = np.mean(brightness_values)

    # Videos below this threshold are classified as nighttime videos.
    night_threshold = 80
    is_night = average_brightness < night_threshold

    # Print the classification result for testing.
    print(f"Average brightness: {average_brightness:.2f}")
    print("Video type:", "NIGHT" if is_night else "DAY")

    return is_night


def adjust_brightness(frame, is_night):
    """Brighten the frame only when the video is a night video."""

    # Keep the original frame unchanged for daytime videos.
    if not is_night:
        return frame

    # Increase contrast using alpha and brightness using beta.
    return cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
