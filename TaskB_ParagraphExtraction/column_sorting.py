"""Column detection and column-major reading-order sorting."""

import numpy as np

from paragraph_detection import consecutive_false_runs


def vertical_projection(binary_image):
    """Count black pixels in each column, matching Jesslyn's prototype."""
    return np.sum(binary_image == 0, axis=0)


def _merge_nearby_ranges(ranges, maximum_gap):
    merged = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= maximum_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def detect_columns(binary_image, maximum_columns=3, padding=30):
    """Detect one, two or three text columns from wide vertical gutters."""
    height, width = binary_image.shape
    projection = vertical_projection(binary_image)

    # A gutter can contain a few table-rule pixels, so it need not be exactly 0.
    nearly_blank = projection <= max(2, round(0.005 * height))
    blank_ranges = consecutive_false_runs(np.logical_not(nearly_blank))
    blank_ranges = _merge_nearby_ranges(blank_ranges, round(0.01 * width))

    minimum_gutter = round(0.025 * width)
    ink_columns = np.flatnonzero(projection > 0)
    if ink_columns.size == 0:
        return []
    content_left, content_right = int(ink_columns[0]), int(ink_columns[-1]) + 1

    gutters = [
        (start, end)
        for start, end in blank_ranges
        if end - start >= minimum_gutter
        and start > content_left
        and end < content_right
    ]

    # The assignment contains at most three columns. Prefer the widest gutters.
    gutters = sorted(gutters, key=lambda item: item[1] - item[0], reverse=True)
    gutters = sorted(gutters[: maximum_columns - 1])

    raw_columns = []
    start = content_left
    for gutter_start, gutter_end in gutters:
        raw_columns.append((start, gutter_start))
        start = gutter_end
    raw_columns.append((start, content_right))

    return [
        (max(0, x1 - padding), min(width, x2 + padding))
        for x1, x2 in raw_columns
    ]


def sort_reading_order(boxes):
    """Sort down the left column first, followed by each column to its right."""
    return sorted(boxes, key=lambda box: (box[0], box[1]))
