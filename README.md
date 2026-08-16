# CSC2014 Digital Image Processing Assignment

## Group Members

| Name | Student ID |
|------|------------|
| Lee Jean Suen (Group Leader) | 24020644 |
| Janet Beh Jing Le | 24121097 |
| Cindy Wong | 22112908 |
| Jesslyn Wong Mei Chui | 23029804 |
| Lim Wen Le | 23028160 |

This repository contains the implementation for the CSC2014 Digital Image Processing Group Assignment, consisting of two tasks:

- **Task A:** YouTube Video Processing
- **Task B:** Paragraph Extraction

## Project Structure

```text
DIP_Assignment/
│
├── README.md
├── .gitignore
│
├── Task_A_Video_Processing/
│   ├── main.py
│   ├── brightness.py
│   ├── face_blur.py
│   ├── overlay.py
│   ├── watermark.py
│   └── utils.py
│
├── Task_B_Paragraph_Extraction/
│   ├── main.py
│   ├── preprocessing.py
│   ├── paragraph_detection.py
│   ├── column_sorting.py
│   └── utils.py
│
├── models/
│   └── face_detector.xml
│
├── input/
│   ├── videos/
│   │   ├── street.mp4
│   │   ├── talking.mp4
│   │   ├── endscreen.mp4
│   │   └── ...
│   │
│   └── images/
│       ├── watermark1.png
│       ├── watermark2.png
│       └── papers/
│           ├── 001.png
│           └── ...
│
├── output/
│   ├── .gitkeep
│   ├── processed_video.avi
│   └── extracted_paragraphs/
│       └── extracted_001/
│           └── paragraph_1.png
│
└── samples/
    ├── Sample_outputs_from_008/
    │   ├── paragraph 1.png
    │   └── ...
    └── Sample output-part A.avi

```

## Task A - YouTube Video Processing

Task A processes the supplied videos with the following operations:
- Detects whether footage is daytime or nighttime and adjusts nighttime brightness.
- Detects and blurs camera-facing faces.
- Resizes and overlays `talking.mp4`.
- Adds `watermark1.png` and `watermark2.png`.
- Appends `endscreen.mp4` to the processed video.

The Haar Cascade face detector used for face detection is located in:

`models/face_detector.xml`

## Running Task A

From the project root:

`python Task_A_Video_Processing/main.py`

Input videos and supporting media are loaded from the `input/` directory. Processed videos are saved to the `output/` directory.

## Task B - Paragraph Extraction

Task B extracts paragraphs from scientific-paper images with different layouts.

The processing pipeline consists of:

Image loading and grayscale conversion.
Otsu thresholding.
Column detection using vertical black-pixel projection.
Paragraph detection using horizontal black-pixel projection.
Paragraph validation and bounding-box extraction.
Reading-order sorting.
Saving extracted paragraphs as individual images.
Running Task B

From the project root:

`python Task_B_Paragraph_Extraction/main.py`

The program automatically processes the paper images stored in:

`input/images/papers/`

and saves the extracted paragraphs under:

`output/extracted_paragraphs/`

For example, paragraph 4 extracted from 003.png is saved as:

`output/extracted_paragraphs/extracted_003/paragraph_4.png`

The program also prints the number of detected columns and paragraphs to the console.

## Requirements

The implementation uses only the libraries permitted by the assignment:

- Python Standard Library
- OpenCV
- NumPy
- Matplotlib

## Samples

The `samples`/ directory contains sample outputs used for reference and comparison:

- Sample paragraph outputs from `008.png`
- Sample output video for Task A

## Output

Generated files are stored in the `output/` directory. Existing paragraph outputs are cleared and regenerated when Task B processes a paper again, preventing outdated paragraph files from remaining in the output folder.

