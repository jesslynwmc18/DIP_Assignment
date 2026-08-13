"""Run the complete reusable Task B paragraph-extraction pipeline."""

from pathlib import Path

from column_sorting import detect_columns, sort_reading_order
from paragraph_detection import detect_paragraphs_in_column
from preprocessing import load_image, otsu_threshold
from utils import prepare_output_directory, save_paragraphs


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input" / "images" / "papers"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "extracted_paragraphs"


def process_paper(image_path, output_root=DEFAULT_OUTPUT_DIR):
    """Process one paper image and return information about saved paragraphs."""
    image_path = Path(image_path)
    colour, gray = load_image(image_path)
    binary = otsu_threshold(gray)

    columns = detect_columns(binary)
    print(f"len: {len(columns)}")
    boxes = []
    for column in columns:        
        boxes.extend(detect_paragraphs_in_column(binary, column))
        
    boxes = sort_reading_order(boxes)

    output_dir = prepare_output_directory(output_root, image_path.stem)
    saved_files = save_paragraphs(colour, boxes, output_dir)
    
    # visualisation 
    # plot_projections(horizontal_projection(binary), vertical_projection(binary))
    
    return {
        "image": image_path,
        "columns": columns,
        "paragraph_boxes": boxes,
        "saved_files": saved_files,
    }


def main():
    """Process supplied images 001.png through 008.png without manual edits."""
    for number in range(1, 9):
        image_path = DEFAULT_INPUT_DIR / f"{number:03d}.png"
        result = process_paper(image_path)
        print(
            f"{image_path.name}: {len(result['columns'])} column(s), "
            f"{len(result['saved_files'])} paragraph(s) saved to "
            f"{result['saved_files'][0].parent if result['saved_files'] else DEFAULT_OUTPUT_DIR}\n"
        )


if __name__ == "__main__":
    main()
