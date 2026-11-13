import argparse
import time
from pathlib import Path
from uuid import uuid4

import cv2
from flask import Flask, Response, render_template, request, send_from_directory, stream_with_context, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from inference import ROOT, detect3d
from video_processor import VIDEO_EXTENSIONS, create_video_pipeline, process_video

ALLOWED_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024


@app.route('/')
def start_page():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/upload', methods=['POST'])
def upload_file():
    upload = request.files.get('image')
    if upload is None or not upload.filename:
        return render_template('index.html', error='Choose an image file to upload.'), 400

    safe_name = secure_filename(upload.filename)
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or suffix not in ALLOWED_EXTENSIONS:
        return render_template('index.html', error='Unsupported image format. Use PNG, JPEG, BMP, TIFF, or WEBP.'), 400

    try:
        with Image.open(upload.stream) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return render_template('index.html', error='The uploaded file is not a valid image.'), 400

    upload.stream.seek(0)
    checkpoint = ROOT / 'weights' / 'resnet18.pkl'
    if not checkpoint.is_file():
        return render_template(
            'index.html',
            error='The trained YOLO3D ResNet18 checkpoint is missing. Run '
                  '`python get_weights.py --weights resnet18` from the weights folder.',
        ), 503

    upload_name = f'{uuid4().hex}{suffix}'
    upload_dir = ROOT / 'static' / 'uploads'
    result_dir = ROOT / 'static' / 'results'
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / upload_name
    result_name = f'{Path(upload_name).stem}_3d.png'
    result_path = result_dir / result_name
    upload.save(upload_path)

    try:
        detect3d(
            reg_weights=checkpoint,
            model_select='resnet18',
            source=upload_path,
            calib_file=ROOT / 'eval' / 'camera_cal' / 'calib_cam_to_cam.txt',
            save_result=True,
            show_result=False,
            output_path=result_dir,
            device='',
            detector_weights=ROOT / 'yolov5s.pt',
            data=ROOT / 'data' / 'coco128.yaml',
        )
        if not result_path.is_file():
            raise OSError(f'Inference completed without creating the expected result: {result_path}')
    except FileNotFoundError as error:
        app.logger.exception('YOLO3D inference resource is missing')
        return render_template('index.html', error=f'Inference resource is missing: {error}'), 503
    except (RuntimeError, TypeError, ValueError) as error:
        app.logger.exception('YOLO3D inference or checkpoint loading failed')
        return render_template('index.html', error=f'Inference failed: {error}'), 500
    except Exception:
        app.logger.exception('YOLO3D inference failed')
        return render_template(
            'index.html',
            error='Inference failed. Check the application console for details and try again.',
        ), 500
    finally:
        try:
            upload_path.unlink()
        except FileNotFoundError:
            pass

    return render_template(
        'index.html',
        init=True,
        detect=True,
        image_url=url_for('static', filename=f'results/{result_name}'),
    )


@app.route('/upload_video', methods=['POST'])
def upload_video():
    upload = request.files.get('video')
    if upload is None or not upload.filename:
        return render_template('index.html', error='Choose a video file to upload.'), 400

    safe_name = secure_filename(upload.filename)
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or suffix not in VIDEO_EXTENSIONS:
        return render_template('index.html', error='Unsupported video format. Use MP4, AVI, or MOV.'), 400

    upload_dir = ROOT / 'static' / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f'{uuid4().hex}{suffix}'
    output_path = ROOT / 'output_videos' / f'{upload_path.stem}_3d.mp4'
    upload.save(upload_path)
    try:
        pipeline = create_video_pipeline()
        process_video(upload_path, pipeline, output=output_path, show=False)
    except Exception as error:
        app.logger.exception('Uploaded video inference failed')
        return render_template('index.html', error=f'Video processing failed: {error}'), 500
    finally:
        try:
            upload_path.unlink()
        except FileNotFoundError:
            pass

    return render_template(
        'index.html',
        video_url=url_for('output_video', filename=output_path.name),
        output_name=output_path.name,
    )


@app.route('/output_videos/<path:filename>')
def output_video(filename):
    return send_from_directory(ROOT / 'output_videos', filename)


@app.route('/webcam')
def webcam_page():
    return render_template('index.html', webcam_url=url_for('webcam_feed'))


@app.route('/webcam_feed')
def webcam_feed():
    def generate():
        capture = cv2.VideoCapture(0)
        writer = None
        if not capture.isOpened():
            capture.release()
            app.logger.error('Could not open webcam index 0')
            return

        try:
            pipeline = create_video_pipeline()
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps != fps:
                fps = 30.0
            output_path = ROOT / 'output_videos' / f'webcam_{uuid4().hex}.mp4'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pipeline.detect_frame(frame)
                frames += 1
                current_fps = frames / max(time.perf_counter() - started, 1e-6)
                cv2.putText(
                    frame, f'Processing | Frame {frames} | FPS {current_fps:.1f}',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                )
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise OSError(f'Could not create webcam output video: {output_path}')
                writer.write(frame)
                encoded, buffer = cv2.imencode('.jpg', frame)
                if not encoded:
                    raise OSError('Could not encode webcam frame for display')
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        except (cv2.error, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
            app.logger.exception('Webcam inference failed')
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    return Response(
        stream_with_context(generate()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.errorhandler(413)
def request_too_large(_error):
    return render_template('index.html', error='Upload is too large. Maximum upload size is 512 MB.'), 413


def parse_video_args():
    parser = argparse.ArgumentParser(description='YOLO3D video and webcam vehicle detection')
    parser.add_argument('--source', help='MP4/AVI/MOV file path or camera index (for example 0)')
    parser.add_argument('--output', help='Output video path; defaults to output_videos/<source>_3d.mp4')
    parser.add_argument('--weights', nargs='+', default=[str(ROOT / 'yolov5s.pt')], help='YOLOv5 detector weights')
    parser.add_argument('--reg-weights', default=str(ROOT / 'weights' / 'resnet18.pkl'), help='YOLO3D regressor checkpoint')
    parser.add_argument('--model-select', default='resnet18', choices=('resnet', 'resnet18', 'vgg11'))
    parser.add_argument('--calib-file', default=str(ROOT / 'eval' / 'camera_cal' / 'calib_cam_to_cam.txt'))
    parser.add_argument('--data', default=str(ROOT / 'data' / 'coco128.yaml'))
    parser.add_argument('--device', default='', help='CUDA device (e.g. 0) or cpu; defaults to automatic selection')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='Inference size; one value or height width')
    parser.add_argument('--classes', nargs='+', type=int, default=[2, 5, 7], help='YOLO class IDs; default car, bus, truck')
    parser.add_argument('--frame-skip', type=int, default=0, help='Run inference every N+1 frames (default: every frame)')
    parser.add_argument('--no-display', action='store_true', help='Process and save without opening a display window')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_video_args()
    if options.source is None:
        app.run(debug=False, host='127.0.0.1', port=5020)
    else:
        image_size = options.imgsz * 2 if len(options.imgsz) == 1 else options.imgsz
        pipeline = create_video_pipeline(
            reg_weights=options.reg_weights,
            model_select=options.model_select,
            calib_file=options.calib_file,
            device=options.device,
            detector_weights=options.weights,
            data=options.data,
            imgsz=image_size,
            classes=options.classes,
        )
        result = process_video(
            options.source,
            pipeline,
            output=options.output,
            frame_skip=options.frame_skip,
            show=not options.no_display,
        )
        print(f'Processed video saved to: {result}')
