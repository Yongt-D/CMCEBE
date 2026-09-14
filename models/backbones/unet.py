# models/backbones/unet.py
"""
U-Net主干网络 - 专为FGCA设计
支持中间特征提取和特征注入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double Convolution Block"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling Block"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling Block"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net实现 - 专为FGCA设计

    特点:
    1. 支持中间特征提取
    2. 支持特征注入
    3. 可配置的特征通道数
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            base_channels: int = 64,
            bilinear: bool = True,
            dropout: float = 0.1,
            return_features: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.return_features = return_features

        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2, dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout)

        # 解码器
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout)

        # 输出层
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # 特征注入相关
        self.aligned_features = None
        self.aligned_features_layer = "bottleneck"

        # 特征融合层（用于注入对齐特征）
        self.feature_fusion = None

        logger.info(f"U-Net initialized: {base_channels} base channels, bilinear={bilinear}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准前向传播"""
        if self.return_features:
            return self.forward_with_features(x)[0]  # 只返回预测结果
        else:
            return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """标准U-Net前向传播"""
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出
        x = self.outc(x)

        return x

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播并返回中间特征

        Returns:
            predictions: 最终预测结果
            features: 中间特征字典
        """
        features = {}

        # 编码器
        x1 = self.inc(x)
        features['encoder_0'] = x1

        x2 = self.down1(x1)
        features['encoder_1'] = x2

        x3 = self.down2(x2)
        features['encoder_2'] = x3

        x4 = self.down3(x3)
        features['encoder_3'] = x4

        x5 = self.down4(x4)
        features['bottleneck'] = x5
        features['encoder_4'] = x5  # 兼容性别名

        # 如果有对齐特征注入
        if self.aligned_features is not None:
            x5 = self._inject_aligned_features(x5, self.aligned_features)
            features['bottleneck_aligned'] = x5

        # 解码器
        x = self.up1(x5, x4)
        features['decoder_0'] = x

        x = self.up2(x, x3)
        features['decoder_1'] = x

        x = self.up3(x, x2)
        features['decoder_2'] = x

        x = self.up4(x, x1)
        features['decoder_3'] = x

        # 输出
        predictions = self.outc(x)
        features['predictions'] = predictions

        return predictions, features

    def set_aligned_features(
            self,
            aligned_features: torch.Tensor,
            layer: str = "bottleneck"
    ) -> torch.Tensor:
        """
        注入对齐后的特征并重新执行部分前向传播

        Args:
            aligned_features: FGCA对齐后的特征
            layer: 注入的层级

        Returns:
            更新后的预测结果
        """
        self.aligned_features = aligned_features
        self.aligned_features_layer = layer

        # 重新执行前向传播（带特征注入）
        # 这里简化实现，实际中可以优化为只执行部分层
        dummy_input = torch.randn(
            aligned_features.shape[0],
            self.in_channels,
            512, 512  # 假设输入尺寸
        ).to(aligned_features.device)

        return self.forward(dummy_input)

    def _inject_aligned_features(
            self,
            original_features: torch.Tensor,
            aligned_features: torch.Tensor
    ) -> torch.Tensor:
        """
        将对齐特征注入到原始特征中

        Args:
            original_features: U-Net的原始特征
            aligned_features: FGCA对齐后的特征

        Returns:
            融合后的特征
        """
        B, C_orig, H, W = original_features.shape
        B_align, C_align = aligned_features.shape[:2]

        # 确保批次维度匹配
        if B != B_align:
            logger.warning(f"Batch size mismatch: {B} vs {B_align}")
            return original_features

        # 调整对齐特征的空间尺寸
        if aligned_features.dim() == 2:
            # [B, C] -> [B, C, H, W]
            aligned_features = aligned_features.view(B_align, C_align, 1, 1)
            aligned_features = aligned_features.expand(-1, -1, H, W)
        elif aligned_features.dim() == 4:
            # [B, C, H', W'] -> [B, C, H, W]
            aligned_features = F.interpolate(
                aligned_features,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        # 通道维度对齐
        if C_align != C_orig:
            if self.feature_fusion is None:
                self.feature_fusion = nn.Conv2d(
                    C_align, C_orig, kernel_size=1, bias=False
                ).to(aligned_features.device)

            aligned_features = self.feature_fusion(aligned_features)

        # 特征融合策略
        fusion_weight = 0.3  # 对齐特征的权重
        fused_features = (1 - fusion_weight) * original_features + fusion_weight * aligned_features

        return fused_features

    def get_feature_channels(self, layer: str) -> int:
        """获取指定层的特征通道数"""
        channel_map = {
            'encoder_0': self.base_channels,
            'encoder_1': self.base_channels * 2,
            'encoder_2': self.base_channels * 4,
            'encoder_3': self.base_channels * 8,
            'encoder_4': self.base_channels * 16 // (2 if self.bilinear else 1),
            'bottleneck': self.base_channels * 16 // (2 if self.bilinear else 1),
        }
        return channel_map.get(layer, self.base_channels)

    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in [
            *self.inc.parameters(),
            *self.down1.parameters(),
            *self.down2.parameters(),
            *self.down3.parameters(),
            *self.down4.parameters()
        ]:
            param.requires_grad = False
        logger.info("U-Net encoder frozen")

    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in [
            *self.inc.parameters(),
            *self.down1.parameters(),
            *self.down2.parameters(),
            *self.down3.parameters(),
            *self.down4.parameters()
        ]:
            param.requires_grad = True
        logger.info("U-Net encoder unfrozen")