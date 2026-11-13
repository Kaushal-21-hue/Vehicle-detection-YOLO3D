import argparse
import hashlib
from pathlib import Path
import sys

import gdown
import torch
from torchvision.models import resnet18, vgg11

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.Model import ResNet, ResNet18, VGG11

WEIGHT_IDS = {
    'resnet': '1Bw4gUsRBxy8XZDGchPJ_URQjbHItikjw',
    'resnet18': '1k_v1RrDO6da_NDhBtMZL5c0QSogCmiRn',
    'vgg11': '1vZcB-NaPUCovVA-pH-g-3NNJuUA948ni',
}
RESNET18_SIZE = 297699863
RESNET18_SHA256 = '186980839060ea2d4bc9e83de04b856384604f9a9aaf1c23a7d6143b24f21186'


def verify_checkpoint(name, path):
    size = path.stat().st_size
    if size < 1024 * 1024:
        raise ValueError(f'Checkpoint is unexpectedly small ({size} bytes): {path}')
    if name == 'resnet18':
        if size != RESNET18_SIZE:
            raise ValueError(f'Checkpoint size mismatch: expected {RESNET18_SIZE}, got {size} bytes')
        hasher = hashlib.sha256()
        with path.open('rb') as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != RESNET18_SHA256:
            raise ValueError(f'Checkpoint SHA-256 mismatch: {digest}')

    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('Checkpoint does not contain the expected model_state_dict')

    architectures = {
        'resnet': lambda: ResNet(model=resnet18(weights=None)),
        'resnet18': lambda: ResNet18(model=resnet18(weights=None)),
        'vgg11': lambda: VGG11(model=vgg11(weights=None)),
    }
    architectures[name]().load_state_dict(checkpoint['model_state_dict'], strict=True)
    return size


def get_weights(name):
    output = Path(__file__).resolve().parent / f'{name}.pkl'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        verify_checkpoint(name, output)
        print(f'Checkpoint already verified: {output}')
        return output

    temporary = output.with_suffix('.pkl.download')
    try:
        downloaded = gdown.download(id=WEIGHT_IDS[name], output=str(temporary), quiet=False)
        if not downloaded or not temporary.is_file():
            raise RuntimeError(f'Download did not produce a checkpoint for {name}')
        size = verify_checkpoint(name, temporary)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f'Verified checkpoint: {output} ({size:,} bytes)')
    return output


def main():
    parser = argparse.ArgumentParser(description='Download and validate official YOLO3D regressor weights')
    parser.add_argument('--weights', choices=tuple(WEIGHT_IDS), required=True, help='Regressor checkpoint to download')
    args = parser.parse_args()
    get_weights(args.weights)


if __name__ == '__main__':
    main()
