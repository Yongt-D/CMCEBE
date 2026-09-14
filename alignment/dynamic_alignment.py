import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DynamicFeatureAlignmentModel(nn.Module):
    def __init__(
            self,
            image_dim: int = 1024,
            text_dim: int = 2048,
            embed_dim: int = 512,
            num_heads: int = 8,
            dropout: float = 0.1,
            use_layer_norm: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.image_dim = image_dim
        self.text_dim = text_dim

        # Image feature projection layer - removed BatchNorm, using simple structure
        self.image_projection = nn.Sequential(
            nn.Conv2d(image_dim, embed_dim, 1),
            nn.ReLU()
        )

        # Text feature projection layer - fix: handle possible extra dimensions
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.ReLU()
        )

        # Multi-head attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feature enhancement blocks
        self.image_enhancement = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU()
        )

        self.text_enhancement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        # Feature fusion network
        self.fusion_network = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, image_dim, 1)
        )

        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Conv2d(image_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )

    def _ensure_tensor_type(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor type is float32"""
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return x

    def _preprocess_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        """Preprocess text features to ensure correct dimensions"""
        # Remove extra dimensions
        while text_features.dim() > 2:
            if text_features.size(1) == 1:
                text_features = text_features.squeeze(1)
            else:
                break

        # Ensure it's 2D: [B, text_dim]
        if text_features.dim() != 2:
            raise ValueError(f"Text features should be 2D after preprocessing, got shape: {text_features.shape}")

        return text_features

    def forward(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        try:
            # Ensure input data type is float32
            image_features = self._ensure_tensor_type(image_features)
            text_features = self._ensure_tensor_type(text_features)

            B, C, H, W = image_features.shape

            # Preprocess text features
            text_features = self._preprocess_text_features(text_features)

            # 1. Feature projection
            image_proj = self.image_projection(image_features)  # [B, embed_dim, H, W]
            text_proj = self.text_projection(text_features)  # [B, embed_dim]

            # 2. Feature enhancement
            enhanced_image = self.image_enhancement(image_proj)
            enhanced_text = self.text_enhancement(text_proj)

            # 3. Prepare attention input - fix dimension issues
            image_flat = enhanced_image.flatten(2).transpose(1, 2)  # [B, HW, embed_dim]
            text_expand = enhanced_text.unsqueeze(1)  # [B, 1, embed_dim]

            # Expand text features to same sequence length as image_flat
            seq_len = image_flat.size(1)  # HW
            text_for_attention = text_expand.expand(B, seq_len, self.embed_dim)  # [B, HW, embed_dim]

            # 4. Apply attention mechanism - fix: ensure all inputs are 3D
            attn_output, attn_weights = self.cross_attention(
                query=image_flat,  # [B, HW, embed_dim]
                key=text_for_attention,  # [B, HW, embed_dim]
                value=text_for_attention,  # [B, HW, embed_dim]
                key_padding_mask=attention_mask,
                need_weights=True
            )

            # 5. Reshape attention output
            attn_output = attn_output.transpose(1, 2).reshape(B, self.embed_dim, H, W)

            # 6. Feature fusion
            concat_features = torch.cat([enhanced_image, attn_output], dim=1)
            fused_features = self.fusion_network(concat_features)

            # 7. Predict confidence
            confidence = self.confidence_head(fused_features)

            # 8. Weighted features
            weighted_features = fused_features * confidence

            # 9. Residual connection
            output_features = weighted_features + image_features

            return output_features

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Image features shape: {image_features.shape}")
            logger.error(f"Text features shape: {text_features.shape}")
            raise

    def compute_loss(
            self,
            weighted_features: torch.Tensor,
            labels: torch.Tensor,
            auxiliary_outputs: Dict = None
    ) -> Dict[str, torch.Tensor]:
        """Loss function calculation"""
        main_loss = F.binary_cross_entropy_with_logits(weighted_features, labels)
        total_loss = main_loss

        return {
            'total_loss': total_loss,
            'main_loss': main_loss
        }


def train_dynamic_feature_alignment(
        model: DynamicFeatureAlignmentModel,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int = 10,
        scheduler=None,
        validation_loader=None
):
    """Training loop"""
    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for batch_idx, (images, texts, labels) in enumerate(train_loader):
            # Data preparation
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, texts)
            loss_dict = model.compute_loss(outputs, labels)
            total_loss = loss_dict['total_loss']

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())

            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, '
                            f'Loss: {total_loss.item():.4f}')

        # Validation phase
        if validation_loader is not None:
            val_loss = validate_model(model, validation_loader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_dynamic_alignment.pth')

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')


def validate_model(
        model: DynamicFeatureAlignmentModel,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device
) -> float:
    """Validation function"""
    model.eval()
    total_val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, texts, labels in val_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            loss = model.compute_loss(outputs, labels)['total_loss']

            total_val_loss += loss.item()
            num_batches += 1

    return total_val_loss / num_batches if num_batches > 0 else float('inf')