"""Image loading and Otsu preprocessing for Task B."""

import cv2

def load_image(image_path):
    """Load both the colour image (for saving) and a grayscale copy."""

    # Read the original image as a colour image using OpenCV's BGR format
    colour = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Check whether the image was successfully loaded
    if colour is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert the colour image to grayscale
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)

    # Keep the colour image for final paragraph extraction
    # Return the grayscale image for preprocessing and detection
    return colour, gray


def otsu_threshold(gray_image):
    """Convert a page to black text (0) on a white background (255)."""

    # Apply Otsu's automatic thresholding to separate text from the background
    # A threshold value of 0 allows Otsu's method to determine the threshold
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Return the binary image for subsequent projection analysis
    return binary
