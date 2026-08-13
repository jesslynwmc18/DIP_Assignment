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
        if not value and start is None: # if its BLANK and BEGINNING (= beginning of zero runs)
            start = index
        elif value and start is not None: # if its FILLED and has START (= end of zero runs)
            runs.append((start, index))
            start = None
    if start is not None: # to append the final blank of the end page
        runs.append((start, len(mask)))
    return runs


def _true_runs(mask):
    """Return consecutive True ranges in a one-dimensional Boolean array."""
    return consecutive_false_runs(np.logical_not(mask))


def horizontal_projection(binary_region):
    """Count black pixels in every row, as in the original prototype."""
    return np.sum(binary_region == 0, axis=1)


def _looks_like_paragraph(binary_region, minimum_text_lines=2):
    """Reject large pictures and ruled tables without using OCR.

    Paragraphs consist of several separated horizontal text-line bands. A table
    normally has a near-full-width rule, while a photograph has ink in almost
    every row. These tests remain projection-based and deliberately simple.
    """
    height, width = binary_region.shape
    if height == 0 or width == 0:
        return False

    ink = binary_region == 0
    row_counts = np.sum(ink, axis=1)
    column_counts = np.sum(ink, axis=0)
    line_count = len(_true_runs(row_counts > 0))
    active_row_fraction = np.count_nonzero(row_counts) / height

    has_horizontal_rule = np.max(row_counts) >= 0.75 * width
    has_vertical_rule = np.max(column_counts) >= 0.75 * height
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
    x1, x2 = column
    region = binary_image[:, x1:x2]
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
    
    active_rows = projection > 0

    page_scale = binary_image.shape[0] / 2339.0
    gap = max(8, round(minimum_paragraph_gap * page_scale))
    blank_runs = [
        (start, end)
        for start, end in consecutive_false_runs(active_rows)
        if end - start >= gap
    ]
    
    # Long blank bands divide the column into candidate content blocks.
    boundaries = [0]
    boundaries.extend((start + end) // 2 for start, end in blank_runs)
    boundaries.append(binary_image.shape[0])
    
    boxes = []
    for top_limit, bottom_limit in zip(boundaries, boundaries[1:]):
        rows = np.flatnonzero(active_rows[top_limit:bottom_limit]) # index of area with content in the selected topand bottom limit
        if rows.size == 0:
            continue
        content_top = top_limit + int(rows[0])
        content_bottom = top_limit + int(rows[-1]) + 1
        candidate = region[content_top:content_bottom]
        if not _looks_like_paragraph(candidate):
            continue

        y1 = max(0, content_top - padding)
        y2 = min(binary_image.shape[0], content_bottom + padding)
        boxes.append((x1, y1, x2, y2))
    return boxes
