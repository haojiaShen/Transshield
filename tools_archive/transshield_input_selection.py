from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


IMAGE_SUFFIXES = {
    '.bmp',
    '.gif',
    '.jpeg',
    '.jpg',
    '.png',
    '.tif',
    '.tiff',
    '.webp',
}


def add_input_selection_args(parser, include_data_path=True):
    if include_data_path:
        parser.add_argument('--data-path', default='')
    parser.add_argument(
        '--image',
        action='append',
        default=[],
        help='Explicit image path. Can be repeated multiple times.',
    )
    parser.add_argument(
        '--image-list',
        default='',
        help='Text file with one image path per line.',
    )
    parser.add_argument(
        '--input-dir',
        default='',
        help='Directory containing images to process. Recursively scans matching files.',
    )
    parser.add_argument(
        '--glob-pattern',
        default='*',
        help='Glob pattern used under --input-dir. Default keeps all image suffixes.',
    )


def infer_target_from_path(path: Path):
    parent_name = path.parent.name
    return int(parent_name) if parent_name.isdigit() else None


def load_image_list(list_path: Path):
    sample_paths = []
    for raw_line in list_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        sample_paths.append(Path(line).expanduser().resolve())
    return sample_paths


def scan_input_dir(input_dir: Path, glob_pattern: str):
    sample_paths = []
    for path in sorted(input_dir.rglob(glob_pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            sample_paths.append(path.resolve())
    return sample_paths


def resolve_selected_sample_paths(
    *,
    data_path='',
    default_data_path='',
    image_paths=None,
    image_list='',
    input_dir='',
    glob_pattern='*',
    max_samples=0,
):
    image_paths = image_paths or []
    explicit_paths = [Path(path).expanduser().resolve() for path in image_paths if path]

    if image_list:
        explicit_paths.extend(load_image_list(Path(image_list).expanduser().resolve()))

    if input_dir:
        explicit_paths.extend(scan_input_dir(Path(input_dir).expanduser().resolve(), glob_pattern))

    selection_mode = 'imagefolder'
    source_root = None
    targets = None
    if explicit_paths:
        sample_paths = explicit_paths
        selection_mode = 'explicit_paths'
        targets = [infer_target_from_path(path) for path in sample_paths]
    else:
        resolved_data_path = data_path or default_data_path
        if not resolved_data_path:
            raise ValueError('one of --data-path / --image / --image-list / --input-dir is required')
        source_root = Path(resolved_data_path).expanduser().resolve()
        imagefolder = ImageFolder(root=str(source_root))
        sample_paths = [Path(path).resolve() for path, _target in imagefolder.samples]
        targets = [target for _path, target in imagefolder.samples]

    if max_samples and max_samples > 0:
        sample_paths = sample_paths[:max_samples]
        targets = targets[:max_samples] if targets is not None else None

    if not sample_paths:
        raise ValueError('no images matched the requested input selection')

    missing = [str(path) for path in sample_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'missing input images: {missing[:4]}')

    return {
        'selection_mode': selection_mode,
        'data_path': str(source_root) if source_root is not None else '',
        'sample_paths': [str(path) for path in sample_paths],
        'targets': targets,
    }


class SelectedImageDataset(Dataset):
    def __init__(self, sample_paths, transform):
        self.sample_paths = [Path(path).resolve() for path in sample_paths]
        self.transform = transform
        self.targets = [infer_target_from_path(path) for path in self.sample_paths]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        path = self.sample_paths[index]
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        target = self.targets[index]
        return tensor, str(path), (-1 if target is None else target)
