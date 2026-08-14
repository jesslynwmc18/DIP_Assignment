"""Column detection and column-major reading-order sorting."""

import numpy as np

from paragraph_detection import consecutive_false_runs
# from matplotlib import pyplot as pt # VISUALISATION

def vertical_projection(binary_image):
    """Count black pixels in each column, matching Jesslyn's prototype."""
    return np.sum(binary_image == 0, axis=0)


def _merge_nearby_ranges(ranges, maximum_gap):
    """Combine blank ranges that are too close together"""
    merged = []
    for start, end in ranges:

        # If the current blank range is close to the previous one,
        # combine them into one large blank range
        if merged and start - merged[-1][1] <= maximum_gap:
            merged[-1] = (merged[-1][0], end)

        # Otherwise, keepthe current range as a separate blankrange (no change)
        else:
            merged.append((start, end))
    return merged


def detect_columns(binary_image, maximum_columns=3, padding=30):
    """Detect one, two or three text columns from wide vertical gutters."""

    # Get the number of pixels in row and columns of the page
    height, width = binary_image.shape

    # Calculate the number of black pixels in each column
    projection = vertical_projection(binary_image)
    
    # ===================================
    # VISUALISATION (Vertical Projection)
    # ===================================
    # pt.figure()
    # pt.plot(projection)
    # pt.title("Vertical Projection")
    # pt.xlabel("Column")
    # pt.ylabel("Number of Black Pixels")
    # pt.show()

    # A gutter can contain a few table-rule pixels, so it need not be exactly 0.
    nearly_blank = projection <= max(2, round(0.005 * height)) # classify minor noise as blank

    # Find continuous ranges of nearly blank columns
    blank_ranges = consecutive_false_runs(np.logical_not(nearly_blank))

    # Combine blank ranges that are separated by only a small gap
    blank_ranges = _merge_nearby_ranges(blank_ranges, round(0.01 * width))

    minimum_gutter = round(0.025 * width) # a blank range must be wide enough to be considered a column gutter
    ink_columns = np.flatnonzero(projection > 0) # index of area with content (vertically)

    # If there is no content on the page, return no columns
    if ink_columns.size == 0:
        return []
        
    content_left, content_right = int(ink_columns[0]), int(ink_columns[-1]) + 1

    gutters = [
        (start, end) # Keep only blank ranges that:
        for start, end in blank_ranges
        if end - start >= minimum_gutter # are wide enough to be gutters, and
        and start > content_left # are located inside the actual page content
        and end < content_right
    ]

    # The assignment contains at most three columns
    # Wider gutters are more likely to separate actual text columns
    gutters = sorted(gutters, key=lambda item: item[1] - item[0], reverse=True) # get width of gutters, sort by largest
    gutters = sorted(gutters[: maximum_columns - 1]) # take 2 gutters

    # Use the gutters to divide the pagge into separate columns
    raw_columns = []
    start = content_left
    
    for gutter_start, gutter_end in gutters:

        # The are before the current gutter forms one column (start of content -> start of gutter)
        raw_columns.append((start, gutter_start))

        # Start the next column after the gutter
        start = gutter_end

    # Add the final columnafter the last gutter
    raw_columns.append((start, content_right))
    
    # Add padding around each detected column
    return [
        (max(0, x1 - padding), min(width, x2 + padding))
        for x1, x2 in raw_columns
    ]


def sort_reading_order(boxes):
    """Sort down the left column first, followed by each column to its right."""

    # Sort by x-coordinate first, then by y-coordinate
    return sorted(boxes, key=lambda box: (box[0], box[1]))
