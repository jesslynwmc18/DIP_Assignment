"""Shared output and diagnostic helpers for Task B."""

from pathlib import Path

import cv2
from matplotlib import pyplot as pt


def prepare_output_directory(output_root, paper_stem):
    """Create a separate output folder for each paper."""

    # Create a folder with "extracted_00n" for each current paper
    output_dir = Path(output_root) / f"extracted_{paper_stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove previously saved paragraphs to prevent old results
    # from remaining when the same paper is processed again
    for old_file in output_dir.glob("paragraph_*.png"):
        old_file.unlink()
    return output_dir


def save_paragraphs(colour_image, boxes, output_dir):
    """Save each detected paragraph as a separate image."""
    saved = []
    for number, (x1, y1, x2, y2) in enumerate(boxes, start=1):

        # Create a filename using its paragraph's detection order
        path = Path(output_dir) / f"paragraph_{number}.png"

        # Crop the paragraph using its bounding box coordinates and save it
        if not cv2.imwrite(str(path), colour_image[y1:y2, x1:x2]):
            raise OSError(f"Could not save paragraph image: {path}")
        saved.append(path)
    return saved


def plot_projections(horizontal, vertical):
    """Optional visualisation of the two projections used by the algorithm."""
    # Display both projections side by side for easier comparison
    figure, axes = pt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].plot(horizontal)
    axes[0].set_title("Horizontal black-pixel projection")
    
    axes[1].plot(vertical)
    axes[1].set_title("Vertical black-pixel projection")
    
    figure.tight_layout()
    pt.show()

def boxed_paragraph(colour_image, boxes):
    """Display detected paragraph regions using bounding boxes."""

    # Convert the image from OpenCV's BGR -> RGB format for matplotlib
    colour_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB)
    
    iteration = 1
    for number, (x1, y1, x2, y2) in enumerate(boxes, start=1):

        # Draw a rectangle around the detected paragraph
        cv2.rectangle(
            colour_image,       # image to draw on
            (x1, y1),           # top-left corner
            (x2, y2),           # bottom-right corner
            (0, 0, 255),        # colour (RGB) -> blue
            2                   # thickness
        )
        
        # Display the image only after all paragraph boxes have been drawn
        if iteration == len(boxes): 
            pt.figure()
            pt.imshow(colour_image)
            pt.show()
        iteration += 1
    
