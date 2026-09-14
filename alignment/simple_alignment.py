import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SimpleFeatureConcatenation(nn.Module):
    """Simple feature concatenation alignment model, used as baseline method"""

    def __init__(
            self,
            image_dim: int = 1024,
            text_dim: int = 2048,
            output_dim: int = 1024,
            dropout: float = 0.1
    ):
        super().__init__()

        # Text feature dimensionality reduction to match image feature channels
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, image_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Simple feature fusion network
        self.fusion_network = nn.Sequential(
            nn.Conv2d(image_dim * 2, output_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Additional processing for residual connection (if input/output dimensions differ)
        self.residual_proj = None
        if image_dim != output_dim:
            self.residual_proj = nn.Conv2d(image_dim, output_dim, kernel_size=1)

    def _ensure_tensor_type(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is float32 type"""
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return x

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        try:
            # Ensure input data type is float32
            image_features = self._ensure_tensor_type(image_features)
            text_features = self._ensure_tensor_type(text_features)

            B, C, H, W = image_features.shape

            # Text feature projection
            text_projected = self.text_projection(text_features)  # [B, image_dim]

            # Expand text features to same spatial dimensions as image features
            text_expanded = text_projected.view(B, -1, 1, 1).expand(-1, -1, H, W)

            # Simple concatenation
            concat_features = torch.cat([image_features, text_expanded], dim=1)

            # Fused features
            fused_features = self.fusion_network(concat_features)

            # Residual connection
            if self.residual_proj is not None:
                residual = self.residual_proj(image_features)
            else:
                residual = image_features

            output = fused_features + residual

            return output

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Image features shape: {image_features.shape}")
            logger.error(f"Text features shape: {text_features.shape}")
            raise

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict:
        """Simple loss calculation, mainly for training supervision"""
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        return {'total_loss': loss}