"""Frame-by-frame video and webcam inference using the existing YOLO3D pipeline."""

import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


VIDEO_EXTENSIONS = {'.avi', '.mov', '.mp4'}
OUTPUT_EXTENSIONS = VIDEO_EXTENSIONS


def parse_video_source(source):
    source = str(source)
    if source.isdigit():
        return int(source)

    path = Path(source).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Video file not found: {path}')
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f'Unsupported video format {path.suffix!r}; use MP4, AVI, or MOV')
    return path


def default_output_path(source):
    output_dir = ROOT / 'output_videos'
    if isinstance(source, int):
        name = f'webcam_{datetime.now():%Y%m%d_%H%M%S}.mp4'
    else:
        name = f'{Path(source).stem}_3d.mp4'
    return output_dir / name


def process_video(source, pipeline, output=None, show=True, frame_skip=0, window_name='YOLO3D video'):
    """Process a video/camera and save the annotated frames; press Q to stop display."""
    parsed_source = parse_video_source(source)
    if frame_skip < 0:
        raise ValueError('frame_skip must be zero or a positive integer')
    output_path = Path(output) if output is not None else default_output_path(parsed_source)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    if output_path.suffix.lower() not in OUTPUT_EXTENSIONS:
        raise ValueError('Output video extension must be .mp4, .avi, or .mov')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(parsed_source)
    if not capture.isOpened():
        capture.release()
        if isinstance(parsed_source, int):
            raise OSError(f'Could not open camera index {parsed_source}')
        raise OSError(f'Could not open video file: {parsed_source}')

    writer = None
    frame_number = 0
    started = time.perf_counter()
    processed_frames = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        delay = max(1, int(1000 / fps))
        while True:
            ok, frame = capture.read()
            if not ok:
                if frame_number == 0:
                    raise ValueError(f'No decodable frames found in source: {parsed_source}')
                break
            if frame is None or frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                raise ValueError(f'Invalid frame dimensions at frame {frame_number + 1}')

            frame_number += 1
            should_detect = frame_skip == 0 or (frame_number - 1) % (frame_skip + 1) == 0
            if should_detect:
                pipeline.detect_frame(frame)
                processed_frames += 1

            elapsed = max(time.perf_counter() - started, 1e-6)
            current_fps = processed_frames / elapsed
            frame_text = f'Frame {frame_number}'
            if total_frames > 0:
                frame_text += f' / {total_frames}'
            status = 'Processing' if should_detect else 'Detection skipped'
            cv2.putText(
                frame, f'{status} | {frame_text} | FPS {current_fps:.1f}',
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
            )

            if writer is None:
                height, width = frame.shape[:2]
                if width <= 0 or height <= 0:
                    raise ValueError('The video source reported invalid frame dimensions')
                codec = 'MJPG' if output_path.suffix.lower() == '.avi' else 'mp4v'
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
                )
                if not writer.isOpened():
                    raise OSError(f'Could not create output video: {output_path}')

            writer.write(frame)
            if show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1 if isinstance(parsed_source, int) else delay) & 0xFF
                if key == ord('q'):
                    break
    except cv2.error as error:
        raise RuntimeError(f'OpenCV failed while processing {parsed_source}: {error}') from error
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if writer is None:
        raise ValueError(f'No video frames were written for source: {parsed_source}')
    return output_path


def create_video_pipeline(
    reg_weights=ROOT / 'weights' / 'resnet18.pkl',
    model_select='resnet18',
    calib_file=ROOT / 'eval' / 'camera_cal' / 'calib_cam_to_cam.txt',
    device='',
    detector_weights=ROOT / 'yolov5s.pt',
    data=ROOT / 'data' / 'coco128.yaml',
    imgsz=(640, 640),
    classes=(2, 5, 7),
):
    from inference import YOLO3DPipeline

    return YOLO3DPipeline(
        reg_weights=reg_weights,
        model_select=model_select,
        calib_file=calib_file,
        device=device,
        detector_weights=detector_weights,
        data=data,
        imgsz=imgsz,
        classes=classes,
    )
