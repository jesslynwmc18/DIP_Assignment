"""Shared output and diagnostic helpers for Task B."""

from pathlib import Path

import cv2
from matplotlib import pyplot as pt


def prepare_output_directory(output_root, paper_stem):
    output_dir = Path(output_root) / f"extracted_{paper_stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prevent stale paragraph files when a page is processed again.
    for old_file in output_dir.glob("paragraph_*.png"):
        old_file.unlink()
    return output_dir


def save_paragraphs(colour_image, boxes, output_dir):
    saved = []
    for number, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        path = Path(output_dir) / f"paragraph_{number}.png"
        if not cv2.imwrite(str(path), colour_image[y1:y2, x1:x2]):
            raise OSError(f"Could not save paragraph image: {path}")
        saved.append(path)
    return saved


def plot_projections(horizontal, vertical):
    """Optional visualisation of the two projections used by the algorithm."""
    figure, axes = pt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(horizontal)
    axes[0].set_title("Horizontal black-pixel projection")
    axes[1].plot(vertical)
    axes[1].set_title("Vertical black-pixel projection")
    figure.tight_layout()
    pt.show()

def boxed_paragraph(colour_image, boxes):
    colour_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB)
    
    iteration = 1
    for number, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(
            colour_image,       # image to draw on
            (x1, y1),           # top-left corner
            (x2, y2),           # bottom-right corner
            (0, 0, 255),        # colour (RGB) -> blue
            2                   # thickness
        )
        # only displays fully boxed page, 
        # not process of each paragraph boxed up
        if iteration == len(boxes): 
            pt.figure()
            pt.imshow(colour_image)
            pt.show()
        iteration += 1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    