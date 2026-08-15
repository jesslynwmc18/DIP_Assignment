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
    overlay_y = 100

    # Overlay talking video onto the main video
    frame[
        overlay_y:overlay_y + overlay_height,
        overlay_x:overlay_x + overlay_width
    ] = talking_frame

    return frame


# Open videos
main_video = cv2.VideoCapture("street.mp4")
talking_video = cv2.VideoCapture("talking.mp4")


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


# Process video frame-by-frame
while True:

    # Read one frame from each video
    success_main, main_frame = main_video.read()
    success_talking, talking_frame = talking_video.read()

    # Stop when the main video ends
    if not success_main:
        break

    # Overlay talking video if a frame was successfully read
    if success_talking:
        main_frame = overlay_talking(
            main_frame,
            talking_frame
        )

    # Save processed frame
    output.write(main_frame)


# Release everything
main_video.release()
talking_video.release()
output.release()

print("Processing complete!")
