# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""

import argparse
import sys
from pathlib import Path
import glob

import numpy as np

# Initialize NumPy's MKL runtime before PyTorch initializes its bundled OpenMP runtime on Windows.
np.linalg.lstsq(np.eye(2), np.ones((2, 1)), rcond=None)

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import LoadImages
from utils.general import LOGGER, check_img_size, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device, time_sync

from torchvision.models import resnet18, vgg11

from script.Dataset import generate_bins, DetectedObject
from library.Calib import get_P
from library.Math import *
from library.Plotting import *
from script import ClassAverages
from script.Model import ResNet, ResNet18, VGG11

model_factory = {
    'resnet': lambda: resnet18(weights=None),
    'resnet18': lambda: resnet18(weights=None),
    'vgg11': lambda: vgg11(weights=None),
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def source_images(source):
    source_path = project_path(source)
    if source_path.is_file():
        paths = [source_path]
    elif source_path.is_dir():
        paths = sorted(path for path in source_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        paths = [
            Path(path) for path in sorted(glob.glob(str(source_path)))
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS
        ]

    if not paths:
        raise FileNotFoundError(f'No supported image files found at {source_path}')
    return paths


def load_regressor_checkpoint(checkpoint_path, regressor, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Checkpoint must be a mapping, got {type(checkpoint).__name__}')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError('Checkpoint model state must be a mapping of parameter names to tensors')

    try:
        regressor.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f'Checkpoint does not match the selected regressor architecture: {error}') from error


class Bbox:
    def __init__(self, box_2d, class_, confidence=None):
        self.box_2d = box_2d
        self.detected_class = class_
        self.confidence = confidence

class YOLO3DPipeline:
    """Load the 2D detector and 3D regressor once for image or video inference."""

    def __init__(
        self,
        reg_weights,
        model_select,
        calib_file,
        device='',
        detector_weights=ROOT / 'yolov5s.pt',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        classes=(2, 5, 7),
    ):
        self.device = select_device(device)
        self.calib_path = project_path(calib_file)
        if not self.calib_path.is_file():
            raise FileNotFoundError(f'Calibration file not found: {self.calib_path}')
        self.proj_matrix = get_P(str(self.calib_path))

        checkpoint_path = project_path(reg_weights)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Regressor checkpoint not found: {checkpoint_path}')
        if model_select not in model_factory:
            raise ValueError(f'Unsupported regressor model {model_select!r}; choose from {", ".join(model_factory)}')

        detector_weights = (
            [project_path(path) for path in detector_weights]
            if isinstance(detector_weights, (list, tuple))
            else project_path(detector_weights)
        )
        detector_paths = detector_weights if isinstance(detector_weights, list) else [detector_weights]
        for path in detector_paths:
            if not path.is_file():
                raise FileNotFoundError(f'Detector weights not found: {path}')
        data = project_path(data)
        if not data.is_file():
            raise FileNotFoundError(f'Detector dataset configuration not found: {data}')

        base_model = model_factory[model_select]()
        self.regressor = regressor_factory[model_select](model=base_model).to(self.device)
        load_regressor_checkpoint(checkpoint_path, self.regressor, self.device)
        self.regressor.eval()

        self.detector = DetectMultiBackend(detector_weights, device=self.device, dnn=False, data=data)
        self.stride = self.detector.stride
        self.pt = self.detector.pt
        self.imgsz = check_img_size(list(imgsz), s=self.stride)
        self.detector.warmup(imgsz=(1, 3, *self.imgsz), half=False)
        self.classes = list(classes) if classes is not None else None
        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)

    @torch.no_grad()
    def detect_frame(self, img):
        """Run genuine 2D + 3D inference on one OpenCV BGR frame, modifying it in place."""
        if img is None or img.ndim != 3 or img.shape[0] <= 0 or img.shape[1] <= 0:
            raise ValueError('Expected a non-empty three-channel image frame')

        resized = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        resized = resized.transpose((2, 0, 1))[::-1]
        resized = np.ascontiguousarray(resized)
        tensor = torch.from_numpy(resized).to(self.device).float() / 255.0
        tensor = tensor.unsqueeze(0)

        prediction = self.detector(tensor, augment=False, visualize=False)
        detections = non_max_suppression(prediction, classes=self.classes)
        for det in detections:
            if not len(det):
                continue
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, confidence, class_id in reversed(det):
                class_name = self.detector.names[int(class_id)]
                coords = [int(value) for value in torch.tensor(xyxy).view(-1).tolist()]
                box_2d = [(coords[0], coords[1]), (coords[2], coords[3])]
                supported = self.averages.recognized_class(class_name)
                if supported:
                    supported = self.averages.dimension_map[class_name.lower()]['count'] > 0
                if not supported:
                    cv2.rectangle(img, box_2d[0], box_2d[1], (0, 165, 255), 2)
                    cv2.putText(
                        img, f'{class_name} {float(confidence):.0%} | 3D unavailable',
                        (box_2d[0][0], max(18, box_2d[0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA
                    )
                    continue
                try:
                    detected = DetectedObject(img, class_name, box_2d, self.proj_matrix)
                    orient, orientation_confidence, dimensions = self.regressor(
                        detected.img.unsqueeze(0).to(self.device)
                    )
                    orient = orient.cpu().numpy()[0]
                    orientation_confidence = orientation_confidence.cpu().numpy()[0]
                    dimensions = dimensions.cpu().numpy()[0] + self.averages.get_item(class_name)

                    bin_index = int(np.argmax(orientation_confidence))
                    alpha = np.arctan2(orient[bin_index, 1], orient[bin_index, 0])
                    alpha += self.angle_bins[bin_index] - np.pi
                    location = plot3d(
                        img, self.proj_matrix, box_2d, dimensions, alpha, detected.theta_ray
                    )
                except (IndexError, ValueError, ZeroDivisionError, np.linalg.LinAlgError) as error:
                    LOGGER.warning('Skipping invalid 3D detection %s: %s', box_2d, error)
                    continue

                confidence_value = float(confidence)
                orientation = alpha + detected.theta_ray
                cv2.rectangle(img, box_2d[0], box_2d[1], (0, 255, 0), 2)
                label = f'{class_name} {confidence_value:.0%}'
                label += f' | {location[2]:.1f}m | yaw {np.degrees(orientation):.0f} deg'
                cv2.putText(
                    img, label, (box_2d[0][0], max(18, box_2d[0][1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                )
                size_label = 'H {:.1f} W {:.1f} L {:.1f}m'.format(*dimensions)
                cv2.putText(
                    img, size_label, (box_2d[0][0], min(img.shape[0] - 5, box_2d[1][1] + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
                )
        return img


@torch.no_grad()
def detect3d(
    reg_weights,
    model_select,
    source,
    calib_file,
    show_result,
    save_result,
    output_path,
    device='',
    detector_weights=ROOT / 'yolov5s.pt',
    data=ROOT / 'data/coco128.yaml',
    imgsz=(640, 640),
    classes=(2, 5, 7)
    ):

    imgs_path = source_images(source)
    pipeline = YOLO3DPipeline(
        reg_weights, model_select, calib_file, device, detector_weights, data, imgsz, classes
    )
    output_path = project_path(output_path) if output_path is not None else None
    if output_path is not None and save_result:
        output_path.mkdir(parents=True, exist_ok=True)

    for img_path in imgs_path:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f'Could not decode image: {img_path}')
        pipeline.detect_frame(img)

        if show_result:
            cv2.imshow('3d detection', img)
            cv2.waitKey(0)

        if save_result and output_path is not None:
            result_path = output_path / f'{img_path.stem}_3d.png'
            if not cv2.imwrite(str(result_path), img):
                raise OSError(f'Could not write inference result: {result_path}')

@torch.no_grad()
def detect2d(
    weights,
    source,
    data,
    imgsz,
    device,
    classes
    ):

    # array for boundingbox detection
    bbox_list = []

    # Directories
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(prediction=pred, classes=classes)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)  # integer class
                    label = names[c]

                    bbox_list.append(Bbox(bbox, label, float(conf)))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return bbox_list

def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    img_2d=None
    ):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes and resizing of boxes 

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='device to run on, e.g. 0, 0,1,2,3, or cpu (default: auto)')
    parser.add_argument('--classes', default=[2, 5, 7], nargs='+', type=int, help='vehicle class IDs: car=2, bus=5, truck=7')
    parser.add_argument('--reg_weights', type=str, default=ROOT / 'weights/resnet18.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet18', choices=tuple(regressor_factory), help='Regressor model')
    parser.add_argument('--calib_file', type=str, default=ROOT / 'eval/camera_cal/calib_cam_to_cam.txt', help='Calibration file or path')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output pat')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_file=opt.calib_file,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path,
        device=opt.device,
        detector_weights=opt.weights,
        data=opt.data,
        imgsz=opt.imgsz,
        classes=opt.classes
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)