"""
Unified Data Manager - Merged Version
Integrated from data_manager.py, data_manager_multi_text.py, data_manager_unified.py
Supports multiple text feature types and datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """
    Custom collate function that can handle None values

    When some sample fields are None (e.g., when not using text features),
    these fields will be removed from the batch instead of trying to collate them
    """
    if not batch:
        return {}

    # Check keys from first sample
    first_item = batch[0]

    # Separate None and non-None keys
    result = {}

    for key in first_item.keys():
        # Collect all values for this key
        values = [item[key] for item in batch]

        # Check if all values are None
        if all(v is None for v in values):
            # All values are None, skip this key or set to None
            result[key] = None
            continue

        # Check if some values are None
        if any(v is None for v in values):
            # Some values are None, filter out None values (this should be avoided)
            logger.warning(f"Key '{key}' has mixed None and non-None values in batch")
            values = [v for v in values if v is not None]
            if not values:
                result[key] = None
                continue

        # For non-None values, use default collation
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], str):
            # String type kept as list
            result[key] = values
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            # Other types kept as list
            result[key] = values

    return result


class UnifiedDataManager:
    """
    Unified Data Manager

    Features:
    - Supports multiple datasets (WHU, Massachusetts, INRIA)
    - Supports multiple text feature types (Janus, Qwen, None)
    - Unified data loading interface
    - Flexible configuration options
    """

    # Text feature dimension mapping
    TEXT_DIM_MAP = {
        'janus': 2048,
        'qwen': 3584,
        'none': 0,
        None: 0
    }

    # Dataset path configuration
    DATASET_CONFIG = {
        'whu_building': {
            'root_key': 'whu_building',
            'splits': ['train', 'val', 'test'],
            'has_text': True
        },
        'massachusetts': {
            'root_key': 'massachusetts',
            'splits': ['train', 'val', 'test'],
            'has_text': True
        },
        'inria': {
            'root_key': 'inria',
            'splits': ['train', 'val', 'test'],
            'has_text': True
        }
    }

    def __init__(self, config: Dict, default_text_type: str = 'janus'):
        """
        Initialize unified data manager

        Args:
            config: Configuration dictionary containing dataset paths, etc.
            default_text_type: Default text feature type ('janus', 'qwen', 'none')
        """
        self.config = config
        self.default_text_type = default_text_type.lower() if default_text_type else 'none'

        # Validate configuration
        if self.default_text_type not in self.TEXT_DIM_MAP:
            raise ValueError(
                f"Invalid text_type: {default_text_type}. "
                f"Must be one of {list(self.TEXT_DIM_MAP.keys())}"
            )

        logger.info(f"UnifiedDataManager initialized with text_type={self.default_text_type}")

    def get_text_dim(self, text_type: Optional[str] = None) -> int:
        """Get text feature dimension"""
        text_type = (text_type or self.default_text_type).lower()
        return self.TEXT_DIM_MAP.get(text_type, 0)

    def get_dataloader(
            self,
            dataset_name: str,
            split: str,
            batch_size: int,
            shuffle: bool = True,
            num_workers: int = 4,
            alignment_type: str = 'none',
            text_type: Optional[str] = None,
            transform=None,
            sample_count: int = -1,
            text_feature_subdir: Optional[str] = None,
            text_subdir: Optional[str] = None,
            **kwargs
    ) -> DataLoader:
        """
        Create data loader

        Args:
            dataset_name: Dataset name
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading threads
            alignment_type: Alignment type ('none', 'simple', 'dynamic', 'generative')
            text_type: Text feature type (overrides default)
            transform: Data transform
            sample_count: Sample count limit (for debugging, -1 means use all)
            text_feature_subdir: Custom text feature subdirectory name
            text_subdir: Custom text description subdirectory name

        Returns:
            DataLoader instance
        """
        # Determine if text features are needed
        text_type = text_type or self.default_text_type
        load_text = (alignment_type != 'none') and (text_type != 'none')

        # Create dataset
        dataset = self._create_dataset(
            dataset_name=dataset_name,
            split=split,
            load_text=load_text,
            text_type=text_type,
            transform=transform,
            text_feature_subdir=text_feature_subdir,
            text_subdir=text_subdir
        )

        # If sample_count specified, limit sample count
        if sample_count > 0 and sample_count < len(dataset):
            from torch.utils.data import Subset
            import random
            indices = list(range(len(dataset)))
            if shuffle:
                random.shuffle(indices)
            indices = indices[:sample_count]
            dataset = Subset(dataset, indices)
            logger.info(f"Limited dataset to {sample_count} samples")

        # Filter out parameters not belonging to DataLoader
        dataloader_kwargs = {}
        valid_dataloader_params = {
            'batch_sampler', 'sampler', 'collate_fn', 'drop_last',
            'timeout', 'worker_init_fn', 'multiprocessing_context',
            'generator', 'prefetch_factor', 'persistent_workers', 'pin_memory'
        }
        for key, value in kwargs.items():
            if key in valid_dataloader_params:
                dataloader_kwargs[key] = value

        # If no custom collate_fn provided, use ours to handle None values
        if 'collate_fn' not in dataloader_kwargs:
            dataloader_kwargs['collate_fn'] = custom_collate_fn

        # Create data loader
        _dl_extra = {}
        if num_workers > 0:
            _dl_extra["prefetch_factor"] = self.config.get("hardware", {}).get("prefetch_factor", 2)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config.get('hardware', {}).get('pin_memory', True),
            persistent_workers=self.config.get('hardware', {}).get('persistent_workers', True) and num_workers > 0,
            **_dl_extra,
            **dataloader_kwargs
        )

        logger.info(
            f"Created dataloader: {dataset_name}/{split}, "
            f"batch_size={batch_size}, text_type={text_type}, "
            f"samples={len(dataset)}"
        )

        return dataloader

    def _create_dataset(
            self,
            dataset_name: str,
            split: str,
            load_text: bool,
            text_type: str,
            transform,
            text_feature_subdir: Optional[str] = None,
            text_subdir: Optional[str] = None
    ) -> Dataset:
        """Create dataset instance"""

        # Get dataset configuration
        if dataset_name not in self.DATASET_CONFIG:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {list(self.DATASET_CONFIG.keys())}"
            )

        dataset_cfg = self.DATASET_CONFIG[dataset_name]

        # Get dataset root directory
        data_config = self.config.get('data', {})
        datasets_config = self.config.get('datasets', {})

        if dataset_name in datasets_config:
            root_dir = Path(datasets_config[dataset_name].get('root_dir'))
        else:
            root_dir = Path(data_config.get('root_dir', 'data')) / dataset_name

        # Validate dataset path
        if not root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        # Create dataset
        dataset = BuildingSegmentationDataset(
            root_dir=root_dir,
            split=split,
            load_text=load_text,
            text_type=text_type,
            transform=transform,
            image_size=data_config.get('image_size', 512),
            text_feature_subdir=text_feature_subdir,
            text_subdir=text_subdir
        )

        return dataset

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get dataset information"""
        if dataset_name not in self.DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        cfg = self.DATASET_CONFIG[dataset_name]
        datasets_config = self.config.get('datasets', {})

        info = {
            'name': dataset_name,
            'splits': cfg['splits'],
            'has_text_features': cfg['has_text'],
            'supported_text_types': list(self.TEXT_DIM_MAP.keys())
        }

        # Add dataset-specific information
        if dataset_name in datasets_config:
            dataset_cfg = datasets_config[dataset_name]
            info.update({
                'root_dir': dataset_cfg.get('root_dir', 'N/A'),
                'description': dataset_cfg.get('description', 'N/A'),
                'train_samples': dataset_cfg.get('train_samples', 'N/A')
            })

        return info


class BuildingSegmentationDataset(Dataset):
    """
    Building Segmentation Dataset

    Supported directory structure:
    dataset_root/
        ├── train/
        │   ├── images/
        │   ├── labels/
        │   └── text_features_{text_type}/  (optional)
        ├── val/
        └── test/
    """

    def __init__(
            self,
            root_dir: Path,
            split: str,
            load_text: bool = False,
            text_type: str = 'janus',
            transform=None,
            image_size: int = 512,
            text_feature_subdir: Optional[str] = None,
            text_subdir: Optional[str] = None
    ):
        """
        Initialize dataset

        Args:
            root_dir: Dataset root directory
            split: Data split ('train', 'val', 'test')
            load_text: Whether to load text features
            text_type: Text feature type
            transform: Data transform
            image_size: Image size
            text_feature_subdir: Custom text feature subdirectory name
            text_subdir: Custom text description subdirectory name
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.load_text = load_text
        self.text_type = text_type.lower()
        self.transform = transform
        self.image_size = image_size

        # Setup paths
        self.split_dir = self.root_dir / split
        self.images_dir = self.split_dir / 'images'
        self.labels_dir = self.split_dir / 'labels'

        # Text feature path
        if self.load_text:
            # If custom subdirectory name provided, use it; otherwise use default format
            if text_feature_subdir:
                self.text_features_dir = self.split_dir / text_feature_subdir
            else:
                self.text_features_dir = self.split_dir / f'text_features_{self.text_type}'

            if not self.text_features_dir.exists():
                logger.warning(
                    f"Text features directory not found: {self.text_features_dir}. "
                    f"Will return None for text features."
                )
                self.load_text = False

        # Validate paths
        self._validate_paths()

        # Get sample list
        self.samples = self._get_samples()

        logger.info(
            f"Dataset initialized: {self.root_dir.name}/{split}, "
            f"samples={len(self.samples)}, load_text={self.load_text}"
        )

    def _validate_paths(self):
        """Validate required paths"""
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"labels directory not found: {self.labels_dir}")

    def _get_samples(self) -> List[Dict]:
        """Get sample list"""
        samples = []

        # Get all image files
        image_files = sorted(self.images_dir.glob('*.png')) + \
                      sorted(self.images_dir.glob('*.jpg')) + \
                      sorted(self.images_dir.glob('*.tif'))

        for img_path in image_files:
            # Find corresponding mask
            mask_path = self.labels_dir / img_path.name
            if not mask_path.exists():
                # Try different extension
                mask_path = self.labels_dir / f"{img_path.stem}.png"

            if not mask_path.exists():
                logger.warning(f"Mask not found for image: {img_path.name}")
                continue

            sample = {
                'image_path': img_path,
                'mask_path': mask_path,
                'name': img_path.stem
            }

            # Add text feature path
            if self.load_text:
                text_path = self.text_features_dir / f"{img_path.stem}.pt"
                if text_path.exists():
                    sample['text_path'] = text_path
                else:
                    sample['text_path'] = None

            samples.append(sample)

        if len(samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.images_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get single sample"""
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)

        # Load mask
        mask = Image.open(sample['mask_path']).convert('L')
        mask = np.array(mask)

        # Binarize mask
        mask = (mask > 127).astype(np.float32)

        # Load text feature (before transform, as transform may need it)
        text_feature = None
        if self.load_text and sample.get('text_path'):
            try:
                text_feature = torch.load(sample["text_path"], map_location="cpu")
                # Ensure it's 1D or 2D tensor
                if text_feature.dim() > 2:
                    text_feature = text_feature.squeeze()
            except Exception as e:
                logger.warning(f"Failed to load text feature for {sample['name']}: {e}")
                text_feature = None

        # Apply transform
        if self.transform:
            # BuildingExtractionTransforms expects dict input
            transform_input = {
                'image': image,
                'label': mask,
                'name': sample['name']
            }
            if text_feature is not None:
                transform_input['text_feature'] = text_feature

            transformed = self.transform(transform_input)
            image = transformed['image']
            mask = transformed['label']  # transform returns 'label'
            if 'text_feature' in transformed:
                text_feature = transformed['text_feature']
        else:
            # Default transform: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Build output
        output = {
            'image': image,
            'mask': mask,
            'label': mask,  # Compatibility
            'name': sample['name']
        }

        # Add text feature to output
        if text_feature is not None:
            output['text_features'] = text_feature
            output['text_feature'] = text_feature  # Compatibility
        else:
            output['text_features'] = None
            output['text_feature'] = None

        return output


# Usage example
if __name__ == '__main__':
    # Example configuration
    config = {
        'data': {
            'root_dir': 'data',
            'image_size': 512,
            'num_workers': 4
        },
        'datasets': {
            'whu_building': {
                'root_dir': 'data/whu_building',
                'description': 'WHU Building Dataset'
            }
        }
    }

    # Create data manager
    manager = UnifiedDataManager(config, default_text_type='janus')

    # Get data loader
    train_loader = manager.get_dataloader(
        dataset_name='whu_building',
        split='train',
        batch_size=2,
        shuffle=True,
        alignment_type='simple',
        text_type='janus'
    )

    # Test loading
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        if batch['text_features'] is not None:
            print(f"Text features shape: {batch['text_features'].shape}")
        break

    # Get dataset info
    info = manager.get_dataset_info('whu_building')
    print(f"\nDataset info: {info}")