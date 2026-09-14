# utils/transforms.py - Simplified version, removed overlapping features
"""
Professional Data Transform Tools - Optimized for Building Extraction
Simplified version, focused on core functionality, removed overlap with training scripts
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image, ImageFilter
import cv2
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BuildingExtractionTransforms:
    """Building Extraction Specialized Data Transform - Simplified Version"""

    def __init__(
            self,
            phase: str = 'train',
            image_size: int = 512,
            augmentation_config: Dict = None
    ):
        self.phase = phase
        self.image_size = image_size
        self.aug_config = augmentation_config or {}

        # Fixed binarization threshold - ensure consistency
        self.BINARY_THRESHOLD = 0.5

        logger.info(f"BuildingExtractionTransforms initialized: {phase} phase, size={image_size}")

    def __call__(self, sample: Dict) -> Dict:
        """Apply transforms - ensure train/val consistency"""

        try:
            image = sample['image']
            label = sample['label']

            # === Synchronized geometric transforms (training only) ===
            if self.phase == 'train':
                image, label = self._apply_sync_geometric_transforms(image, label)

            # === Image transform ===
            transformed_image = self._transform_image(image)

            # === Label transform (critical: ensure consistency) ===
            transformed_label = self._transform_label_consistently(label)

            result = {
                'image': transformed_image,
                'label': transformed_label,
                'name': sample.get('name', '')
            }

            # === Text feature processing ===
            if 'text_feature' in sample:
                result['text_feature'] = self._process_text_feature(sample['text_feature'])

            return result

        except Exception as e:
            logger.error(f"Transform error: {e}")
            logger.error(f"Sample keys: {list(sample.keys())}")
            raise

    def _apply_sync_geometric_transforms(self, image, label):
        """Synchronized geometric transforms - ensure image and label transform consistently"""

        # Ensure both are PIL format
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if isinstance(label, np.ndarray):
            # Process value range before converting label to PIL
            if label.max() <= 1.0:
                label_pil = Image.fromarray((label * 255).astype(np.uint8), mode='L')
            else:
                label_pil = Image.fromarray(label.astype(np.uint8), mode='L')
        else:
            label_pil = label

        # Synchronized geometric transforms
        # Horizontal flip
        if random.random() < self.aug_config.get('horizontal_flip', 0.5):
            image = TF.hflip(image)
            label_pil = TF.hflip(label_pil)

        # Vertical flip
        if random.random() < self.aug_config.get('vertical_flip', 0.5):
            image = TF.vflip(image)
            label_pil = TF.vflip(label_pil)

        # Random rotation
        rotation_degrees = self.aug_config.get('rotation_degrees', 15)
        if rotation_degrees > 0:
            angle = random.uniform(-rotation_degrees, rotation_degrees)
            if abs(angle) > 1:
                image = TF.rotate(image, angle, fill=0)
                label_pil = TF.rotate(label_pil, angle, fill=0)

        return image, label_pil

    def _transform_image(self, image):
        """Image transform"""

        # Ensure PIL format
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Color transforms (training only)
        if self.phase == 'train':
            image = self._apply_color_transforms(image)

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image

    def _apply_color_transforms(self, image):
        """Color transforms - training only"""

        color_jitter_config = self.aug_config.get('color_jitter', {})

        if random.random() < 0.8 and color_jitter_config:
            # Apply color jitter
            brightness = color_jitter_config.get('brightness', 0.15)
            contrast = color_jitter_config.get('contrast', 0.15)
            saturation = color_jitter_config.get('saturation', 0.1)
            hue = color_jitter_config.get('hue', 0.05)

            if brightness > 0:
                factor = random.uniform(1 - brightness, 1 + brightness)
                image = TF.adjust_brightness(image, factor)

            if contrast > 0:
                factor = random.uniform(1 - contrast, 1 + contrast)
                image = TF.adjust_contrast(image, factor)

            if saturation > 0:
                factor = random.uniform(1 - saturation, 1 + saturation)
                image = TF.adjust_saturation(image, factor)

            if hue > 0:
                factor = random.uniform(-hue, hue)
                image = TF.adjust_hue(image, factor)

        # Gaussian blur
        if random.random() < self.aug_config.get('gaussian_blur', 0.1):
            radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image

    def _transform_label_consistently(self, label):
        """Consistent label transform - critical function"""

        # Step 1: Convert to numpy
        if isinstance(label, Image.Image):
            label_np = np.array(label)
        elif isinstance(label, torch.Tensor):
            label_np = label.numpy()
        else:
            label_np = np.array(label)

        # Step 2: Handle dimensions
        while len(label_np.shape) > 2:
            if label_np.shape[0] == 1:
                label_np = label_np.squeeze(0)
            elif label_np.shape[-1] == 1:
                label_np = label_np.squeeze(-1)
            else:
                label_np = label_np[..., 0] if label_np.shape[-1] < label_np.shape[0] else label_np[0]
                break

        # Step 3: Normalize value range
        if label_np.max() > 1.0:
            label_np = label_np.astype(np.float32) / 255.0
        else:
            label_np = label_np.astype(np.float32)

        # Step 4: Unified binarization (critical step)
        label_binary = (label_np > self.BINARY_THRESHOLD).astype(np.float32)

        # Step 5: Resize
        label_pil = Image.fromarray((label_binary * 255).astype(np.uint8), mode='L')
        label_resized = label_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        # Step 6: Convert to tensor
        label_tensor = TF.to_tensor(label_resized)
        label_tensor = torch.clamp(label_tensor, 0.0, 1.0)

        return label_tensor

    def _process_text_feature(self, text_feature):
        """Process text feature"""
        if isinstance(text_feature, torch.Tensor):
            text_feature = text_feature.squeeze()
            if text_feature.dim() == 0:
                text_feature = text_feature.unsqueeze(0)
        else:
            text_feature = torch.tensor(text_feature, dtype=torch.float32)
            text_feature = text_feature.squeeze()

        return text_feature.float()


# Test Time Augmentation (TTA) - keep professional feature
class TestTimeAugmentation:
    """Test Time Augmentation - professional feature retained"""

    def __init__(self, image_size=512):
        self.image_size = image_size
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def augment(self, image, return_inverse_transforms=True):
        """Generate multiple augmented versions"""

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        augmented_images = []
        inverse_transforms = []

        # Original image
        aug_img = self.base_transform(image)
        augmented_images.append(aug_img)
        inverse_transforms.append(lambda x: x)

        # Horizontal flip
        h_flip_img = self.base_transform(TF.hflip(image))
        augmented_images.append(h_flip_img)
        inverse_transforms.append(lambda x: torch.flip(x, dims=[2]))

        # Vertical flip
        v_flip_img = self.base_transform(TF.vflip(image))
        augmented_images.append(v_flip_img)
        inverse_transforms.append(lambda x: torch.flip(x, dims=[1]))

        # 90 degree rotation
        rotate_img = self.base_transform(TF.rotate(image, 90))
        augmented_images.append(rotate_img)
        inverse_transforms.append(lambda x: torch.rot90(x, k=-1, dims=[1, 2]))

        if return_inverse_transforms:
            return augmented_images, inverse_transforms
        else:
            return augmented_images

    def merge_predictions(self, predictions, inverse_transforms):
        """Merge multiple prediction results"""

        # Apply inverse transforms
        transformed_preds = []
        for pred, inv_transform in zip(predictions, inverse_transforms):
            transformed_pred = inv_transform(pred)
            transformed_preds.append(transformed_pred)

        # Average predictions
        merged_prediction = torch.stack(transformed_preds).mean(dim=0)
        return merged_prediction


# Keep some professional helper tools
class AdvancedAugmentations:
    """Advanced Augmentation Tools - optional use"""

    @staticmethod
    def random_shadow(image, p=0.2):
        """Random shadow effect"""
        if random.random() < p:
            img_array = np.array(image)
            h, w = img_array.shape[:2]

            # Create random shadow region
            shadow_vertices = np.array([
                [random.randint(0, w // 2), random.randint(0, h // 2)],
                [random.randint(w // 2, w), random.randint(0, h // 2)],
                [random.randint(w // 2, w), random.randint(h // 2, h)],
                [random.randint(0, w // 2), random.randint(h // 2, h)]
            ])

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [shadow_vertices], 255)

            # Apply shadow
            shadow_intensity = random.uniform(0.3, 0.7)
            img_array = img_array.astype(np.float32)
            img_array[mask > 0] *= shadow_intensity
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            return Image.fromarray(img_array)
        return image

    @staticmethod
    def random_bright_spots(image, p=0.15):
        """Random bright spots"""
        if random.random() < p:
            img_array = np.array(image)
            h, w = img_array.shape[:2]

            # Random number and position of bright spots
            num_spots = random.randint(1, 3)
            for _ in range(num_spots):
                center_x = random.randint(w // 4, 3 * w // 4)
                center_y = random.randint(h // 4, 3 * h // 4)
                radius = random.randint(10, 30)

                # Create Gaussian bright spot
                y, x = np.ogrid[:h, :w]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

                brightness_factor = random.uniform(1.2, 1.5)
                img_array = img_array.astype(np.float32)
                img_array[mask] *= brightness_factor
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            return Image.fromarray(img_array)
        return image


if __name__ == "__main__":
    # Simplified test code
    print("Testing BuildingExtractionTransforms...")

    # Create test data
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_label = np.random.randint(0, 2, (512, 512), dtype=np.float32)

    sample = {
        'image': test_image,
        'label': test_label,
        'name': 'test_sample'
    }

    # Test transform
    transform = BuildingExtractionTransforms(
        phase='train',
        image_size=512,
        augmentation_config={
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,
            'rotation_degrees': 15,
            'color_jitter': {
                'brightness': 0.15,
                'contrast': 0.15,
                'saturation': 0.1,
                'hue': 0.05
            },
            'gaussian_blur': 0.1
        }
    )

    transformed_sample = transform(sample)

    print(f"Transform test passed!")
    print(f"   Original image shape: {test_image.shape}")
    print(f"   Transformed image shape: {transformed_sample['image'].shape}")
    print(f"   Transformed label shape: {transformed_sample['label'].shape}")
    print(f"   Label value range: [{transformed_sample['label'].min():.3f}, {transformed_sample['label'].max():.3f}]")

    # Test TTA
    tta = TestTimeAugmentation(image_size=512)
    augmented_images, inverse_transforms = tta.augment(test_image)
    print(f"TTA test passed: generated {len(augmented_images)} augmented images")

    print("All tests completed successfully!")