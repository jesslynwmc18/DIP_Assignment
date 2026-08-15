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


# Open videos
main_video = cv2.VideoCapture("../input/videos/street.mp4")
talking_video = cv2.VideoCapture("../input/videos/talking.mp4")


# Get main video dimensions
main_width = int(
    main_video.get(cv2.CAP_PROP_FRAME_WIDTH)
)

main_height = int(
    main_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# Get main video FPS
fps = main_video.get(cv2.CAP_PROP_FPS)

print("Main video:", main_width, "x", main_height)
print("FPS:", fps)


# Create output video
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

output = cv2.VideoWriter(
    "overlay_test.avi",
    fourcc,
    fps,
    (main_width, main_height)
)


# Load watermark
watermark1 = cv2.imread(
    "../input/images/watermark1.png",
    1
)

watermark2 = cv2.imread(
    "../input/images/watermark2.png",
    1
)

# Check watermark grayscale values
watermark_gray = cv2.cvtColor(
    watermark1,
    cv2.COLOR_BGR2GRAY
)


# Process video frame-by-frame
while True:

    # Read one frame from the main video
    success_main, main_frame = main_video.read()

    # Stop when the main video ends
    if not success_main:
        break

    # Read one frame from the talking video
    success_talking, talking_frame = talking_video.read()

    # If talking video reaches the end, restart it
    if not success_talking:
        talking_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success_talking, talking_frame = talking_video.read()

    # Overlay talking video
    if success_talking:
        main_frame = overlay_talking(
            main_frame,
            talking_frame
        )

    # Add watermark 1
    main_frame = add_watermark1(
        main_frame,
        watermark1
    )
    
    # Add watermark 2
    main_frame = add_watermark2(
        main_frame,
        watermark2
    )
    
    # Save processed frame
    output.write(main_frame)


# Release everything
main_video.release()
talking_video.release()
output.release()

print("Processing complete!")