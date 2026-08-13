"""Paragraph detection using horizontal black-pixel projection."""

import numpy as np
# from matplotlib import pyplot as pt # VISUALISATION

def consecutive_false_runs(mask):
    """Return (start, end) pairs for consecutive False values.

    This is the reusable equivalent of Jesslyn's getConsecutiveZeros().
    End positions use normal Python exclusive indexing.
    """
    runs = []
    start = None
    for index, value in enumerate(mask):
        # Start recording when a blank regions begins
        if not value and start is None: 
            start = index

        # Save the blank region when conent is encountered
        elif value and start is not None: 
            runs.append((start, index))
            start = None

    # Include a blank region that continues until the end of the image
    if start is not None: 
        runs.append((start, len(mask)))
    return runs


def _true_runs(mask):
    """Return consecutive True ranges in a one-dimensional Boolean array."""
    # Invert the mask so that actually True regions can be found using
    # the consecutive_false_runs function
    return consecutive_false_runs(np.logical_not(mask))


def horizontal_projection(binary_region):
    """Count the number of black pixels in every row"""
    return np.sum(binary_region == 0, axis=1)


def _looks_like_paragraph(binary_region, minimum_text_lines=2):
    """Reject large pictures and ruled tables without using OCR.

    Paragraphs consist of several separated horizontal text-line bands. A table
    normally has a near-full-width rule, while a photograph has ink in almost
    every row. These tests remain projection-based and deliberately simple.
    """

    # Reject empty regions
    height, width = binary_region.shape
    if height == 0 or width == 0:
        return False

    # Identify black pixels and count them in each row and column
    ink = binary_region == 0
    row_counts = np.sum(ink, axis=1)
    column_counts = np.sum(ink, axis=0)

    # Count separate groups of active rows, representing text-line bands
    line_count = len(_true_runs(row_counts > 0))

    # Measure how much of the region contains black pixels
    active_row_fraction = np.count_nonzero(row_counts) / height

    # Detect strong horizontal / vertical lines that may indicate tables
    has_horizontal_rule = np.max(row_counts) >= 0.75 * width
    has_vertical_rule = np.max(column_counts) >= 0.75 * height

    # Accept the region only if it has multiple text lines and
    # does not resemble an image / table
    return (
        line_count >= minimum_text_lines
        and active_row_fraction < 0.90
        and not has_horizontal_rule
        and not has_vertical_rule
    )


def detect_paragraphs_in_column(
    binary_image,
    column,
    minimum_paragraph_gap=24,
    padding=30,
):
    """Find paragraph boxes inside one column using horizontal whitespace.

    A paragraph gap must be wider than ordinary gaps between text lines. The
    value is modestly scaled by the page height so the method is reusable for
    pages rendered at a different resolution.
    """

    # Extract the current column from the binary image with the column coordinates
    x1, x2 = column
    region = binary_image[:, x1:x2]

    # Calculate the number of black pixels in each row
    projection = horizontal_projection(region)
    
    # =====================================
    # VISUALISATION (Horizontal Projection)
    # =====================================
    # pt.figure()
    # pt.plot(projection)
    # pt.title("Horizontal Projection")
    # pt.xlabel("Row")
    # pt.ylabel("Number of Black Pixels")
    # pt.show()

    # Mark rows containing at least one black pixel as active
    active_rows = projection > 0

    # Scale the paragraph-gap threshold according to the page height
    page_scale = binary_image.shape[0] / 2339.0
    gap = max(8, round(minimum_paragraph_gap * page_scale))

    # Find blank row ranges that are large enough to separate paragraphs
    blank_runs = [
        (start, end)
        for start, end in consecutive_false_runs(active_rows)
        if end - start >= gap
    ]
    
   # Use large blank regions to divide the column into candidate blocks
    boundaries = [0]
    boundaries.extend((start + end) // 2 for start, end in blank_runs)
    boundaries.append(binary_image.shape[0])
    
    boxes = []
    for top_limit, bottom_limit in zip(boundaries, boundaries[1:]):

        # Find index of rows containing content within the current candidate block
        rows = np.flatnonzero(active_rows[top_limit:bottom_limit])
        if rows.size == 0:
            continue

        # Determine the top and bottom boundaries of the actual content
        content_top = top_limit + int(rows[0])
        content_bottom = top_limit + int(rows[-1]) + 1

        # Extract the candidate region for paragraph validation
        candidate = region[content_top:content_bottom]

        # Reject regions that do not resemble normal paragraph text
        if not _looks_like_paragraph(candidate):
            continue

        # Add padding around the detected paragraph boundaries
        y1 = max(0, content_top - padding)
        y2 = min(binary_image.shape[0], content_bottom + padding)

        # Store the paragraph bounding box
        boxes.append((x1, y1, x2, y2))
    return boxes
