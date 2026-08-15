from pathlib import Path

import cv2

# Overlay talking video frame onto the main video frame
def overlay_talking(frame, talking_frame):

    # Get the dimensions of the main video frame
    frame_height, frame_width = frame.shape[:2]

    # Set talking video overlay size to 25% of the main video
    overlay_width = int(frame_width * 0.25)
    overlay_height = int(frame_height * 0.25)

    # Resize talking video frame
    talking_frame = cv2.resize(
        talking_frame,
        (overlay_width, overlay_height)
    )

    # Set overlay position
    overlay_x = 100
    overlay_y = 50

    # Overlay talking video onto the main video
    frame[
        overlay_y:overlay_y + overlay_height,
        overlay_x:overlay_x + overlay_width
    ] = talking_frame

    return frame


# Add watermark1 to video
def add_watermark1(frame, watermark1):

    # Convert watermark to grayscale
    watermark_gray = cv2.cvtColor(
        watermark1,
        cv2.COLOR_BGR2GRAY
    )

    # Create binary mask
    _, watermark_mask = cv2.threshold(
        watermark_gray,
        130,
        255,
        cv2.THRESH_BINARY
    )

    # Keep only the watermark pixels
    watermark_area = cv2.bitwise_and(
        watermark1,
        watermark1,
        mask=watermark_mask
    )
    

    # Keep only the original video pixels
    background_mask = cv2.bitwise_not(watermark_mask)

    background_area = cv2.bitwise_and(
        frame,
        frame,
        mask=background_mask
    )

    # Combine the two areas
    frame = cv2.add(
        background_area,
        watermark_area
    )

    return frame

# Add watermark2 to video
def add_watermark2(frame, watermark2):

    # Convert watermark to grayscale
    watermark_gray = cv2.cvtColor(
        watermark2,
        cv2.COLOR_BGR2GRAY
    )

    # Create binary mask
    _, watermark_mask = cv2.threshold(
        watermark_gray,
        100,
        255,
        cv2.THRESH_BINARY
    )

    # Keep only the watermark pixels
    watermark_area = cv2.bitwise_and(
        watermark2,
        watermark2,
        mask=watermark_mask
    )

    # Keep only the original video pixels
    background_mask = cv2.bitwise_not(watermark_mask)

    background_area = cv2.bitwise_and(
        frame,
        frame,
        mask=background_mask
    )

    # Combine the two areas
    frame = cv2.add(
        background_area,
        watermark_area
    )

    return frame


def add_watermarks(frame, watermark1, watermark2):
    """Apply both watermark algorithms in the agreed order."""

    frame = add_watermark1(frame, watermark1)
    return add_watermark2(frame, watermark2)


def main():
    """Run standalone street-video test."""

    task_a_dir = Path(__file__).resolve().parent
    repository_root = task_a_dir.parent
    video_dir = repository_root / "input" / "videos"
    image_dir = repository_root / "input" / "images"
    output_dir = repository_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_video_path = video_dir / "street.mp4"
    talking_video_path = video_dir / "talking.mp4"
    output_path = output_dir / "overlay_test.avi"

    # Open videos using repository-relative paths.
    main_video = cv2.VideoCapture(str(main_video_path))
    talking_video = cv2.VideoCapture(str(talking_video_path))
    output = None

    try:
        if not main_video.isOpened():
            raise ValueError(f"Cannot open video: {main_video_path}")
        if not talking_video.isOpened():
            raise ValueError(f"Cannot open video: {talking_video_path}")

        # Get the main video settings for the standalone test output.
        main_width = int(main_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        main_height = int(main_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = main_video.get(cv2.CAP_PROP_FPS)

        print("Main video:", main_width, "x", main_height)
        print("FPS:", fps)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (main_width, main_height)
        )
        if not output.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        # Load both supplied watermark images.
        watermark1 = cv2.imread(str(image_dir / "watermark1.png"), 1)
        watermark2 = cv2.imread(str(image_dir / "watermark2.png"), 1)
        if watermark1 is None or watermark2 is None:
            raise ValueError(f"Cannot load watermark images from: {image_dir}")

        while True:
            success_main, main_frame = main_video.read()
            if not success_main:
                break

            # Restart talking.mp4 when it reaches the end.
            success_talking, talking_frame = talking_video.read()
            if not success_talking:
                talking_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success_talking, talking_frame = talking_video.read()

            if success_talking:
                main_frame = overlay_talking(main_frame, talking_frame)

            main_frame = add_watermarks(
                main_frame,
                watermark1,
                watermark2
            )
            output.write(main_frame)
    finally:
        main_video.release()
        talking_video.release()
        if output is not None:
            output.release()

    print(f"Processing complete! Saved: {output_path}")


if __name__ == "__main__":
    main()
