"""Image loading and Otsu preprocessing for Task B."""

# from pathlib import Path

import cv2


# def load_image(image_path):
#     """Load both the colour image (for saving) and a grayscale copy."""
#     image_path = Path(image_path)
#     colour = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
#     if colour is None:
#         raise FileNotFoundError(f"Could not read image: {image_path}")
#     gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
#     return colour, gray

def load_image(image_path):
    """Load both the colour image (for saving) and a grayscale copy."""
    colour = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if colour is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    return colour, gray


def otsu_threshold(gray_image):
    """Convert a page to black text (0) on a white background (255)."""
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary
