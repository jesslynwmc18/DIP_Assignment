
"""

"""

import cv2

# Open the videos
main_video = cv2.VideoCapture("street.mp4")
talking_video = cv2.VideoCapture("talking.mp4")

# Get main video size
main_width = int(main_video.get(cv2.CAP_PROP_FRAME_WIDTH))
main_height = int(main_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get talking video size
talking_width = int(talking_video.get(cv2.CAP_PROP_FRAME_WIDTH))
talking_height = int(talking_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Print the sizes
print("Main video:", main_width, "x", main_height)
print("Talking video:", talking_width, "x", talking_height)

# Release
main_video.release()
talking_video.release()