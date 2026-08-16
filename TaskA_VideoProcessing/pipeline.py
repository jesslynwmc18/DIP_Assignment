"""Master integration pipeline for Task A video processing.

Processing order:
    day/night detection -> brightness adjustment -> face blur
    -> Member 2 talking overlay -> Member 2 watermarks -> endscreen

Member 2's algorithms remain in overlay_watermark.py and are imported here through
their agreed integration functions. This file supplies the video timing, assets, and
master processing order without duplicating those algorithms.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional, Sequence

import cv2

from brightness import adjust_brightness, detect_day_night
from face_blur import blur_faces, load_face_detector


TASK_A_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_A_DIR.parent
VIDEO_DIR = REPOSITORY_ROOT / "input" / "videos"
IMAGE_DIR = REPOSITORY_ROOT / "input" / "images"
MODEL_DIR = REPOSITORY_ROOT / "models"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "output"

FACE_CASCADE_PATH = MODEL_DIR / "face_detector.xml"
TALKING_VIDEO_PATH = VIDEO_DIR / "talking.mp4"
ENDSCREEN_VIDEO_PATH = VIDEO_DIR / "endscreen.mp4"
WATERMARK_PATHS = (
    IMAGE_DIR / "watermark1.png",
    IMAGE_DIR / "watermark2.png",
)

REQUIRED_VIDEO_NAMES = (
    "alley.mp4",
    "office.mp4",
    "singapore.mp4",
    "traffic.mp4",
)


class Member2Functions:
    """Validated references to Member 2's two future processing functions."""

    def __init__(
        self,
        overlay_talking: Callable,
        add_watermarks: Callable,
        module_name: str,
    ) -> None:
        self.overlay_talking = overlay_talking
        self.add_watermarks = add_watermarks
        self.module_name = module_name


class IndexedVideoReader:
    """Read source frames by index while avoiding unnecessary repeated decoding.

    This is used to supply talking-video frames at the correct time when the main
    and talking videos have different FPS values. It is also used to resample the
    endscreen to each output video's FPS.
    """

    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))

        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if not _is_valid_fps(self.fps):
            self.capture.release()
            raise ValueError(
                f"Video has invalid FPS metadata ({self.fps}): {video_path}"
            )
        if self.frame_count <= 0:
            self.capture.release()
            raise ValueError(f"Video contains no readable frames: {video_path}")

        self.duration_seconds = self.frame_count / self.fps
        self._current_index = -1
        self._current_frame = None

    def read_index(self, frame_index: int):
        """Return one frame by index, resetting when a looping source wraps."""

        if not 0 <= frame_index < self.frame_count:
            raise IndexError(
                f"Frame {frame_index} is outside 0..{self.frame_count - 1} "
                f"for {self.video_path}"
            )

        if frame_index < self._current_index:
            # This occurs when talking.mp4 loops back to its first frame.
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise RuntimeError(f"Cannot rewind video: {self.video_path}")
            self._current_index = -1
            self._current_frame = None

        while self._current_index < frame_index:
            success, frame = self.capture.read()
            if not success:
                raise RuntimeError(
                    f"Cannot read frame {self._current_index + 1} "
                    f"from {self.video_path}"
                )
            self._current_index += 1
            self._current_frame = frame

        # A copy protects the cached frame if Member 2 modifies its input in place.
        return self._current_frame.copy()

    def read_looped_time(self, time_seconds: float):
        """Return the frame at a time position, looping at the source duration."""

        looped_time = time_seconds % self.duration_seconds
        frame_index = min(
            int(looped_time * self.fps),
            self.frame_count - 1,
        )
        return self.read_index(frame_index)

    def release(self) -> None:
        """Release the underlying OpenCV video capture."""

        self.capture.release()


def _is_valid_fps(fps: float) -> bool:
    """Return whether an FPS value is finite and usable by VideoWriter."""

    return math.isfinite(fps) and fps > 0.0


def _load_member2_functions(module_name: str) -> Member2Functions:
    """Import and validate Member 2's module without duplicating their algorithms."""

    try:
        module: ModuleType = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Cannot import Member 2 module '{module_name}'. Place the module in "
            f"{TASK_A_DIR} or provide its importable module name."
        ) from error

    overlay_talking = getattr(module, "overlay_talking", None)
    add_watermarks = getattr(module, "add_watermarks", None)

    missing = []
    if not callable(overlay_talking):
        missing.append("overlay_talking(frame, talking_frame)")
    if not callable(add_watermarks):
        missing.append("add_watermarks(frame, watermark1, watermark2)")

    if missing:
        raise AttributeError(
            f"Member 2 module '{module_name}' is missing callable function(s): "
            + ", ".join(missing)
        )

    return Member2Functions(overlay_talking, add_watermarks, module_name)


def _load_watermarks():
    """Load both supplied watermark images for Member 2's future function."""

    watermarks = []
    for watermark_path in WATERMARK_PATHS:
        watermark = cv2.imread(str(watermark_path), cv2.IMREAD_UNCHANGED)
        if watermark is None:
            raise ValueError(f"Cannot load watermark image: {watermark_path}")
        watermarks.append(watermark)

    return tuple(watermarks)


def _resize_frame_to_fit(frame, target_size: tuple[int, int]):
    """Fit a frame into a target resolution without changing its aspect ratio.

    Black bars are added only when source and target aspect ratios differ. The
    current assignment assets are all 1280x720, but this keeps appending robust if
    a replacement endscreen has a different resolution in the future.
    """

    target_width, target_height = target_size
    source_height, source_width = frame.shape[:2]

    if (source_width, source_height) == target_size:
        return frame

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    left = horizontal_padding // 2
    right = horizontal_padding - left
    top = vertical_padding // 2
    bottom = vertical_padding - top

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def _validate_processed_frame(frame, expected_size: tuple[int, int]) -> None:
    """Fail clearly if an integration step returns an invalid output frame."""

    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("A processing step returned no valid frame.")
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(
            "A processing step must return a three-channel BGR colour frame."
        )

    actual_size = (frame.shape[1], frame.shape[0])
    if actual_size != expected_size:
        raise ValueError(
            f"A processing step changed the frame resolution from "
            f"{expected_size[0]}x{expected_size[1]} to "
            f"{actual_size[0]}x{actual_size[1]}."
        )


def append_endscreen(
    writer,
    endscreen_path: Path,
    target_fps: float,
    target_size: tuple[int, int],
) -> int:
    """Append the endscreen at the output FPS and resolution.

    Output video writers have a single fixed FPS. Therefore, the 30 FPS endscreen
    is sampled by time at the main video's FPS. This drops frames for 25 FPS output
    and duplicates frames for 50 FPS output while preserving playback duration.
    """

    endscreen = IndexedVideoReader(endscreen_path)
    try:
        target_frame_count = max(
            1,
            round(endscreen.duration_seconds * target_fps),
        )

        for output_index in range(target_frame_count):
            source_time = output_index / target_fps
            source_index = min(
                int(source_time * endscreen.fps),
                endscreen.frame_count - 1,
            )
            frame = endscreen.read_index(source_index)
            frame = _resize_frame_to_fit(frame, target_size)
            writer.write(frame)

        print(
            f"  Endscreen: {endscreen.fps:g} FPS -> {target_fps:g} FPS, "
            f"{target_frame_count} output frames"
        )
        return target_frame_count
    finally:
        endscreen.release()


def process_video(
    input_path: Path,
    output_path: Path,
    face_cascade,
    member2: Optional[Member2Functions] = None,
    watermarks=None,
) -> dict:
    """Process one main video, then append the supplied endscreen video."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Output path must not overwrite the original input video.")
    if not ENDSCREEN_VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Endscreen video does not exist: {ENDSCREEN_VIDEO_PATH}"
        )

    # Member 1's detector samples the video once before frame-by-frame processing.
    is_night = detect_day_night(str(input_path))

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open input video: {input_path}")

    writer = None
    talking_source = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if not _is_valid_fps(fps):
            raise ValueError(
                f"Input video has invalid FPS metadata ({fps}): {input_path}"
            )
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Input video has invalid resolution {width}x{height}: {input_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        if member2 is not None:
            if watermarks is None or len(watermarks) != 2:
                raise ValueError("Both watermark images are required for Member 2.")
            talking_source = IndexedVideoReader(TALKING_VIDEO_PATH)

        print(
            f"Processing {input_path.name}: {fps:g} FPS, "
            f"{width}x{height}, {reported_frame_count} reported frames"
        )

        main_frame_count = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            # Member 1 processing (do not replace these algorithms).
            frame = adjust_brightness(frame, is_night)
            frame = blur_faces(frame, face_cascade)

            if member2 is not None:
                # Member 3 supplies a correctly timed talking frame. Member 2 owns
                # both of the following image-processing algorithms.
                talking_frame = talking_source.read_looped_time(
                    main_frame_count / fps
                )
                frame = member2.overlay_talking(frame, talking_frame)
                frame = member2.add_watermarks(
                    frame,
                    watermarks[0],
                    watermarks[1],
                )

            _validate_processed_frame(frame, (width, height))
            writer.write(frame)
            main_frame_count += 1

            if main_frame_count % max(1, round(fps * 5)) == 0:
                print(f"  Processed {main_frame_count} main-video frames")

        if main_frame_count == 0:
            raise RuntimeError(f"No frames could be read from: {input_path}")
        if reported_frame_count > 0 and main_frame_count < reported_frame_count:
            raise RuntimeError(
                f"Video decoding ended early: read {main_frame_count} of "
                f"{reported_frame_count} reported frames from {input_path}"
            )

        endscreen_frame_count = append_endscreen(
            writer,
            ENDSCREEN_VIDEO_PATH,
            fps,
            (width, height),
        )
    finally:
        capture.release()
        if talking_source is not None:
            talking_source.release()
        if writer is not None:
            writer.release()

    total_frame_count = main_frame_count + endscreen_frame_count
    print(
        f"Saved {output_path} "
        f"({total_frame_count} frames, {total_frame_count / fps:.3f} seconds)"
    )

    return {
        "input": input_path,
        "output": output_path,
        "is_night": is_night,
        "fps": fps,
        "size": (width, height),
        "main_frames": main_frame_count,
        "endscreen_frames": endscreen_frame_count,
        "total_frames": total_frame_count,
    }


def _resolve_input_video(value: str) -> Path:
    """Resolve CLI video names relative to the repository, not the launch folder."""

    supplied_path = Path(value)
    if supplied_path.is_absolute():
        return supplied_path
    if len(supplied_path.parts) == 1:
        return VIDEO_DIR / supplied_path
    return REPOSITORY_ROOT / supplied_path


def _resolve_output_directory(value: Optional[Path]) -> Path:
    """Resolve an optional output directory relative to the repository root."""

    if value is None:
        return DEFAULT_OUTPUT_DIR
    if value.is_absolute():
        return value
    return REPOSITORY_ROOT / value


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for batch and sample processing."""

    parser = argparse.ArgumentParser(
        description=(
            "Process Task A videos with Member 1 and Member 3 functionality. "
            "With no video names, all four required videos are processed."
        )
    )
    parser.add_argument(
        "videos",
        nargs="*",
        metavar="VIDEO",
        help=(
            "Input video name(s), for example street.mp4. Bare names are read "
            "from input/videos. Defaults to the four required videos."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Relative paths are resolved from the repository "
            "root. Defaults to output/."
        ),
    )
    parser.add_argument(
        "--member2-module",
        default="overlay_watermark",
        help=(
            "Importable module containing overlay_talking and add_watermarks. "
            "Defaults to overlay_watermark."
        ),
    )
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run the requested videos and return a process-style status code."""

    parser = _build_argument_parser()
    options = parser.parse_args(arguments)

    video_values = options.videos or list(REQUIRED_VIDEO_NAMES)
    input_paths = [_resolve_input_video(value) for value in video_values]
    output_dir = _resolve_output_directory(options.output_dir)

    if not FACE_CASCADE_PATH.is_file():
        parser.error(f"Face cascade file does not exist: {FACE_CASCADE_PATH}")

    try:
        face_cascade = load_face_detector(str(FACE_CASCADE_PATH))
        member2 = _load_member2_functions(options.member2_module)
        watermarks = _load_watermarks()
        print("Talking overlay and watermark processing enabled.")
    except (AttributeError, ImportError, ValueError) as error:
        parser.error(str(error))

    failures = []
    for input_path in input_paths:
        output_path = output_dir / f"{input_path.stem}_processed.mp4"
        try:
            process_video(
                input_path,
                output_path,
                face_cascade,
                member2=member2,
                watermarks=watermarks,
            )
        except Exception as error:  # Report one input failure, then try the others.
            failures.append((input_path, error))
            print(f"ERROR processing {input_path}: {error}")

    if failures:
        print(f"Completed with {len(failures)} failed video(s).")
        return 1

    print(f"Completed successfully: {len(input_paths)} video(s) processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
