"""
Unified Loss Function - Merged Version
Integrated from segmentation_loss.py, enhanced_loss.py, cleaned_enhanced_loss.py, improved_boundary_loss.py
Provides flexible configurable loss function combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class UnifiedSegmentationLoss(nn.Module):
    """
    Unified Segmentation Loss Function

    Features:
    - Multiple base losses: BCE, Dice, Focal, Tversky
    - Optional boundary loss: Adaptive, Distance-based
    - Flexible loss weight configuration
    - Automatic weight adjustment

    Loss Components:
    1. BCE Loss: Standard binary cross-entropy
    2. Dice Loss: Focuses on overlapping regions
    3. Focal Loss: Handles class imbalance
    4. Tversky Loss: Adjustable FP/FN weights
    5. Boundary Loss: Enhances boundary prediction (optional)
    """

    def __init__(
            self,
            # Base loss configuration
            use_bce: bool = True,
            use_dice: bool = True,
            use_focal: bool = True,
            use_tversky: bool = True,

            # Boundary loss configuration
            use_boundary: bool = False,
            boundary_type: str = 'adaptive',  # 'adaptive', 'distance', 'none'

            # Loss weights
            bce_weight: float = 1.0,
            dice_weight: float = 1.0,
            focal_weight: float = 1.0,
            tversky_weight: float = 1.0,
            boundary_weight: float = 0.1,

            # Focal Loss parameters
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,

            # Tversky Loss parameters
            tversky_alpha: float = 0.3,  # FP weight
            tversky_beta: float = 0.7,  # FN weight

            # Auto adjustment
            auto_adjust: bool = True,
            adjust_interval: int = 100,

            # Other
            smooth: float = 1e-6
    ):
        """
        Initialize unified loss function

        Args:
            use_*: Whether to use each loss component
            boundary_type: Boundary loss type
            *_weight: Weight for each loss component
            focal_alpha/gamma: Focal Loss parameters
            tversky_alpha/beta: FP/FN weights for Tversky Loss
            auto_adjust: Whether to auto-adjust weights
            smooth: Smoothing term to avoid division by zero
        """
        super().__init__()

        # Base loss switches
        self.use_bce = use_bce
        self.use_dice = use_dice
        self.use_focal = use_focal
        self.use_tversky = use_tversky
        self.use_boundary = use_boundary

        # Boundary loss type
        self.boundary_type = boundary_type.lower()
        if self.boundary_type not in ['adaptive', 'distance', 'none']:
            raise ValueError(f"Invalid boundary_type: {boundary_type}")

        # Loss weights
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight

        # Focal Loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Tversky Loss parameters
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        # Auto adjustment
        self.auto_adjust = auto_adjust
        self.adjust_interval = adjust_interval
        self.step_counter = 0

        # Other parameters
        self.smooth = smooth

        # BCE loss function
        if self.use_bce:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

        # Record weight history (for auto adjustment)
        self.weight_history = {
            'bce': [],
            'dice': [],
            'focal': [],
            'tversky': [],
            'boundary': []
        }

        self._log_config()

    def _log_config(self):
        """Log configuration info"""
        components = []
        if self.use_bce:
            components.append(f"BCE({self.bce_weight:.2f})")
        if self.use_dice:
            components.append(f"Dice({self.dice_weight:.2f})")
        if self.use_focal:
            components.append(f"Focal({self.focal_weight:.2f})")
        if self.use_tversky:
            components.append(f"Tversky({self.tversky_weight:.2f})")
        if self.use_boundary:
            components.append(f"Boundary({self.boundary_weight:.2f},{self.boundary_type})")

        logger.info(
            f"UnifiedSegmentationLoss initialized: {' + '.join(components)}, "
            f"auto_adjust={self.auto_adjust}"
        )

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            return_components: bool = False
    ) -> torch.Tensor:
        """
        Calculate loss

        Args:
            predictions: Predicted logits [B, C, H, W] (typically C=1)
            targets: Ground truth labels [B, C, H, W] (0-1 range)
            return_components: Whether to return each component's loss value

        Returns:
            total_loss or (total_loss, loss_dict)
        """
        # Ensure targets are float type
        targets = targets.float()

        # If targets are in 0-255 range, normalize to 0-1
        if targets.max() > 1.0:
            targets = targets / 255.0

        # Binarize (ensure only 0 and 1)
        targets = (targets > 0.5).float()

        # Calculate each loss component
        loss_components = {}
        total_loss = torch.tensor(0.0, device=predictions.device)

        # 1. BCE Loss
        if self.use_bce:
            bce_loss = self.bce_loss(predictions, targets)
            loss_components['bce_loss'] = bce_loss
            total_loss += self.bce_weight * bce_loss

        # 2. Dice Loss
        if self.use_dice:
            dice_loss = self.dice_loss(predictions, targets)
            loss_components['dice_loss'] = dice_loss
            total_loss += self.dice_weight * dice_loss

        # 3. Focal Loss
        if self.use_focal:
            focal_loss = self.focal_loss(predictions, targets)
            loss_components['focal_loss'] = focal_loss
            total_loss += self.focal_weight * focal_loss

        # 4. Tversky Loss
        if self.use_tversky:
            tversky_loss = self.tversky_loss(predictions, targets)
            loss_components['tversky_loss'] = tversky_loss
            total_loss += self.tversky_weight * tversky_loss

        # 5. Boundary Loss
        if self.use_boundary and self.boundary_type != 'none':
            if self.boundary_type == 'adaptive':
                boundary_loss = self.adaptive_boundary_loss(predictions, targets)
            elif self.boundary_type == 'distance':
                boundary_loss = self.distance_boundary_loss(predictions, targets)
            else:
                boundary_loss = torch.tensor(0.0, device=predictions.device)

            loss_components['boundary_loss'] = boundary_loss
            total_loss += self.boundary_weight * boundary_loss

        loss_components['total_loss'] = total_loss

        # Auto adjust weights
        if self.auto_adjust and self.training:
            self.step_counter += 1
            if self.step_counter % self.adjust_interval == 0:
                self._adjust_weights(loss_components)

        if return_components:
            return total_loss, loss_components
        return total_loss

    def dice_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Dice Loss (Soft Dice)

        Formula: 1 - (2*TP + smooth) / (2*TP + FP + FN + smooth)
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice

    def focal_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal Loss

        Formula: -alpha*(1-p_t)^gamma * log(p_t)
        """
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )

        # Calculate probabilities
        probs = torch.sigmoid(predictions)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Focal weight
        focal_weight = (1 - pt) ** self.focal_gamma

        # Alpha weight
        alpha_weight = torch.where(
            targets == 1,
            self.focal_alpha,
            1 - self.focal_alpha
        )

        # Final loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        return focal_loss.mean()

    def tversky_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Tversky Loss

        Formula: 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        When alpha=beta=0.5, equivalent to Dice Loss
        Increase alpha to reduce FP, increase beta to reduce FN
        """
        # Convert to probabilities
        probs = torch.sigmoid(predictions)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate TP, FP, FN
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()

        # Tversky coefficient
        tversky = (tp + self.smooth) / (
                tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
        )

        return 1.0 - tversky

    def adaptive_boundary_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptive Boundary Loss

        Uses morphological operations to extract boundaries, adaptive weighting
        """
        # Extract boundary
        boundary_targets = self._extract_boundary(targets)

        # Calculate BCE loss on boundary region
        boundary_pred = predictions * boundary_targets
        boundary_true = targets * boundary_targets

        # Boundary BCE
        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_pred,
            boundary_true,
            reduction='none'
        )

        # Adaptive weight: boundary pixels have higher weight
        weights = 1.0 + 2.0 * boundary_targets  # Boundary weight x3
        weighted_loss = (boundary_bce * weights).mean()

        return weighted_loss

    def distance_boundary_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Distance-based Boundary Loss

        Uses distance transform, higher weight closer to boundary
        """
        # Extract boundary
        boundary_targets = self._extract_boundary(targets)

        # Calculate distance map (simplified: use Gaussian blur approximation)
        distance_map = F.avg_pool2d(
            boundary_targets,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # Normalize distance map
        distance_map = distance_map / (distance_map.max() + self.smooth)

        # Boundary BCE
        bce = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction='none'
        )

        # Distance weighted
        weighted_loss = (bce * (1.0 + distance_map)).mean()

        return weighted_loss

    def _extract_boundary(
            self,
            masks: torch.Tensor,
            kernel_size: int = 3
    ) -> torch.Tensor:
        """
        Extract boundary

        Uses morphological erosion: boundary = mask - eroded_mask
        """
        # Simplified erosion: use average pooling
        kernel = torch.ones(
            1, 1, kernel_size, kernel_size,
            device=masks.device
        ) / (kernel_size * kernel_size)

        # Dilation = 1 - erosion(1 - mask)
        inverted = 1.0 - masks
        padding = kernel_size // 2

        eroded_inv = F.conv2d(
            inverted,
            kernel,
            padding=padding
        )
        eroded = 1.0 - eroded_inv

        # Boundary = original - eroded
        boundary = masks - eroded
        boundary = torch.clamp(boundary, 0, 1)

        return boundary

    def _adjust_weights(self, loss_components: Dict[str, torch.Tensor]):
        """
        Auto adjust weights

        Dynamically adjust weights based on relative magnitude of each loss component
        """
        # Record current loss values
        for key in ['bce', 'dice', 'focal', 'tversky', 'boundary']:
            loss_key = f'{key}_loss'
            if loss_key in loss_components:
                self.weight_history[key].append(
                    loss_components[loss_key].item()
                )

        # Adjust every 100 steps
        if len(self.weight_history['bce']) >= 100:
            # Calculate averages
            avg_losses = {}
            for key in ['bce', 'dice', 'focal', 'tversky', 'boundary']:
                if len(self.weight_history[key]) > 0:
                    avg_losses[key] = sum(self.weight_history[key][-100:]) / 100

            # Normalize weights (simple strategy: inversely proportional to loss magnitude)
            if len(avg_losses) > 0:
                total = sum(avg_losses.values())
                if total > 0:
                    scale = len(avg_losses) / total

                    if self.use_bce and 'bce' in avg_losses:
                        self.bce_weight = max(0.1, avg_losses['bce'] * scale)
                    if self.use_dice and 'dice' in avg_losses:
                        self.dice_weight = max(0.1, avg_losses['dice'] * scale)
                    if self.use_focal and 'focal' in avg_losses:
                        self.focal_weight = max(0.1, avg_losses['focal'] * scale)
                    if self.use_tversky and 'tversky' in avg_losses:
                        self.tversky_weight = max(0.1, avg_losses['tversky'] * scale)
                    if self.use_boundary and 'boundary' in avg_losses:
                        self.boundary_weight = max(0.05, avg_losses['boundary'] * scale * 0.5)

            # Clear history
            for key in self.weight_history:
                self.weight_history[key] = []

    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return {
            'bce_weight': self.bce_weight,
            'dice_weight': self.dice_weight,
            'focal_weight': self.focal_weight,
            'tversky_weight': self.tversky_weight,
            'boundary_weight': self.boundary_weight
        }


# Preset configurations
class LossPresets:
    """Common loss function configuration presets"""

    @staticmethod
    def basic():
        """Basic configuration: BCE + Dice"""
        return UnifiedSegmentationLoss(
            use_bce=True,
            use_dice=True,
            use_focal=False,
            use_tversky=False,
            use_boundary=False
        )

    @staticmethod
    def standard():
        """Standard configuration: BCE + Dice + Focal"""
        return UnifiedSegmentationLoss(
            use_bce=True,
            use_dice=True,
            use_focal=True,
            use_tversky=False,
            use_boundary=False,
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5
        )

    @staticmethod
    def full():
        """Full configuration: All loss components"""
        return UnifiedSegmentationLoss(
            use_bce=True,
            use_dice=True,
            use_focal=True,
            use_tversky=True,
            use_boundary=False,
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5,
            tversky_weight=0.5
        )

    @staticmethod
    def with_boundary():
        """With boundary enhancement: Standard + Adaptive Boundary"""
        return UnifiedSegmentationLoss(
            use_bce=True,
            use_dice=True,
            use_focal=True,
            use_tversky=False,
            use_boundary=True,
            boundary_type='adaptive',
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5,
            boundary_weight=0.1
        )


# Usage example
if __name__ == '__main__':
    # Create loss function
    criterion = LossPresets.with_boundary()

    # Test
    predictions = torch.randn(2, 1, 256, 256)  # Logits
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()

    # Calculate loss
    total_loss, components = criterion(predictions, targets, return_components=True)

    print(f"Total Loss: {total_loss.item():.4f}")
    print("\nLoss Components:")
    for name, value in components.items():
        if name != 'total_loss':
            print(f"  {name}: {value.item():.4f}")

    print(f"\nCurrent Weights: {criterion.get_weights()}")