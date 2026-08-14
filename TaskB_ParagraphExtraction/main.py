"""Run the complete reusable Task B paragraph-extraction pipeline."""

from pathlib import Path

from column_sorting import detect_columns, sort_reading_order, vertical_projection
from paragraph_detection import detect_paragraphs_in_column, horizontal_projection
from preprocessing import load_image, otsu_threshold
from utils import prepare_output_directory, save_paragraphs, boxed_paragraph

# Define the main project folder and the I/O locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input" / "images" / "papers"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "extracted_paragraphs"


def process_paper(image_path, output_root=DEFAULT_OUTPUT_DIR):
    """Process one paper image and return information about saved paragraphs."""

    # Load the original colour image and create a grayscale copy for preocessing
    image_path = Path(image_path)
    colour, gray = load_image(image_path)

    # Convert the grayscale image into a black and white binary image
    binary = otsu_threshold(gray)
    
    # ==============================================================================
    # VISUALISATION (Projections - Full Page without column/paragraph separation)
    # ==============================================================================
    # plot_projections(horizontal_projection(binary), vertical_projection(binary))

    # Detect the text columns in the page
    columns = detect_columns(binary)
    
    boxes = []

    # Detect the paragraphs regions separately within each detected column
    for column in columns:        
        boxes.extend(detect_paragraphs_in_column(binary, column))

    # Sort the detected paragraphs intothe correct reading order
    boxes = sort_reading_order(boxes)
    
    # ========================================
    # VISUALISATION (Paragraph Bounding Boxes)
    # ========================================
    # Display the detected paragraph boxes on the original image
    # boxed_paragraph(colour, boxes)

    # Create the output folder and save each detected paragraph as an image
    output_dir = prepare_output_directory(output_root, image_path.stem)
    saved_files = save_paragraphs(colour, boxes, output_dir)
    
    # Return the results so they can be used to check the processing outcome    
    return {
        "image": image_path,
        "columns": columns,
        "paragraph_boxes": boxes,
        "saved_files": saved_files,
    }


def main():
    """Process supplied images 001.png through 008.png without manual edits."""

    # Automatically process all 8 paper images
    for number in range(1, 9):

        # Generate the input filename: input/images/papers/00n.png
        image_path = DEFAULT_INPUT_DIR / f"{number:03d}.png"

        # Run the complete processing pipeline forthe current paper
        result = process_paper(image_path)

        # Display the number of detected columns and saved paragraphs (console)
        print(
            f"{image_path.name}: {len(result['columns'])} column(s), "
            f"{len(result['saved_files'])} paragraph(s) saved to "
            f"{result['saved_files'][0].parent if result['saved_files'] else DEFAULT_OUTPUT_DIR}\n"
        )

# Run Task B pipeline when this file is executed directly
if __name__ == "__main__":
    main()
