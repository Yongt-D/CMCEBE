"""
CMCE V5: Cross-Modal Co-Evolution V5 - Stable & Optimized Version
跨模态协同进化 V5 - 稳定优化版

V5 设计理念（基于 V3/V4 经验）：
1. 回归稳定：移除 V4 的自适应模块，使用固定但优化的参数
2. 折中参数：取 V3/V4 之间的中间值
3. 学习率优化：改进的调度策略（作为创新点）
4. 训练效率：100轮训练代替150轮

V5 vs V3 vs V4:
- V3: dropout=0.12, residual=0.06, max_change=0.15 → 稳定但seed=123差
- V4: 自适应dropout/residual, 可学习约束 → seed=123好但seed=666崩溃
- V5: dropout=0.11, residual=0.07, max_change=0.12 → 折中稳定

核心创新（保持）：
1. 双向闭环精炼
2. 迭代收敛机制
3. 不确定性感知
4. 可证明的收敛性

目标：
- 均值 IoU >= 90.40%
- 标准差 <= 0.35%
- 均值 Gap <= 1.40%

目标期刊：TIP / Information Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# V5 核心模块（简化稳定版）
# ============================================================================

class SemanticAnchorDecomposerV5(nn.Module):
    """语义锚点分解器 V5 - 简化稳定版"""

    def __init__(self, text_dim: int, hidden_dim: int, num_anchors: int = 3, dropout: float = 0.11):
        super().__init__()

        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors

        # 共享基础编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 每个锚点的特化层
        self.anchor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_anchors)
        ])

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_encoder(text_embeddings)

        anchors = []
        for head in self.anchor_heads:
            anchor = head(shared_features)
            anchors.append(anchor)

        return torch.stack(anchors, dim=1)


class VisualContextEncoderV5(nn.Module):
    """视觉上下文编码器 V5"""

    def __init__(self, scale_channels: Tuple[int, ...], context_dim: int = 256, dropout: float = 0.11):
        super().__init__()

        self.num_scales = len(scale_channels)
        self.context_dim = context_dim

        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for ch in scale_channels
        ])

        self.fusion = nn.Sequential(
            nn.Linear(context_dim * self.num_scales, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim)
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        scale_contexts = []
        for feat, encoder in zip(scale_features, self.scale_encoders):
            ctx = encoder(feat)
            scale_contexts.append(ctx)

        concat = torch.cat(scale_contexts, dim=-1)
        context = self.fusion(concat)

        return context


class InconsistencyDetectorV5(nn.Module):
    """不一致性检测器 V5"""

    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 256, dropout: float = 0.11):
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.inconsistency_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.inconsistency_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_context: torch.Tensor,
                text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.visual_proj(visual_context)
        t = self.text_proj(text_embedding)

        diff = v - t
        features = torch.cat([v, t, diff], dim=-1)

        inconsistency_vec = self.inconsistency_net(features)
        score = self.inconsistency_score(inconsistency_vec)

        return inconsistency_vec, score


class TextRefinementModuleV5(nn.Module):
    """
    文本精炼模块 V5 - 简化稳定版

    V5 特点：
    1. 固定参数（不使用自适应）
    2. 更保守的文本变化约束 (0.12 vs V3的0.15)
    3. 折中的 residual_scale (0.07)
    """

    def __init__(self, text_dim: int, hidden_dim: int = 256, dropout: float = 0.11,
                 residual_scale: float = 0.07, max_change_ratio: float = 0.12):
        super().__init__()

        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.max_change_ratio = max_change_ratio

        self.text_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, text_dim),
            nn.LayerNorm(text_dim)
        )

        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim),
            nn.Sigmoid()
        )

        # V5: 固定残差缩放
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

    def forward(self, text_embedding: torch.Tensor,
                inconsistency_vec: torch.Tensor,
                inconsistency_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adjustment = self.text_adapter(inconsistency_vec)

        gate_input = torch.cat([inconsistency_vec, inconsistency_score], dim=-1)
        gate = self.refine_gate(gate_input)

        # 计算原始变化量
        raw_change = self.residual_scale * gate * adjustment

        # 约束文本变化幅度
        text_norm = text_embedding.norm(dim=-1, keepdim=True) + 1e-6
        change_norm = raw_change.norm(dim=-1, keepdim=True)
        max_allowed_change = self.max_change_ratio * text_norm

        scale_factor = torch.where(
            change_norm > max_allowed_change,
            max_allowed_change / (change_norm + 1e-6),
            torch.ones_like(change_norm)
        )
        constrained_change = raw_change * scale_factor

        refined = text_embedding + constrained_change
        text_change = constrained_change.norm(dim=-1).mean()

        return refined, text_change


class DeformableCrossModalOffsetV5(nn.Module):
    """可变形跨模态偏移模块 V5"""

    def __init__(self, visual_dim: int, anchor_dim: int, hidden_dim: int = 64,
                 max_offset: float = 4.0, dropout: float = 0.11):
        super().__init__()

        self.visual_dim = visual_dim
        self.max_offset = max_offset

        self.offset_net = nn.Sequential(
            nn.Linear(visual_dim + anchor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )

        # V5: 固定偏移缩放
        self.offset_scale = nn.Parameter(torch.tensor(max_offset * 0.4))

    def forward(self, visual_feat: torch.Tensor,
                anchor_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = visual_feat.shape

        visual_flat = visual_feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        anchor_expanded = anchor_feat.unsqueeze(1).unsqueeze(1)
        anchor_expanded = anchor_expanded.expand(B, H, W, -1).reshape(B * H * W, -1)

        combined = torch.cat([visual_flat, anchor_expanded], dim=-1)
        offset = self.offset_net(combined)
        offset = offset.view(B, H, W, 2)

        offset = offset * self.offset_scale

        return offset


class DeformableVisualRefinerV5(nn.Module):
    """可变形视觉精炼器 V5 - 简化稳定版"""

    def __init__(self, visual_dim: int, text_dim: int, anchor_dim: int,
                 hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.11,
                 max_offset: float = 4.0, use_deformable: bool = True,
                 residual_scale: float = 0.07, confidence_baseline: float = 0.65):
        super().__init__()

        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_deformable = use_deformable
        self.confidence_baseline = confidence_baseline

        if use_deformable:
            self.offset_module = DeformableCrossModalOffsetV5(
                visual_dim=visual_dim,
                anchor_dim=anchor_dim,
                hidden_dim=hidden_dim // 4,
                max_offset=max_offset,
                dropout=dropout
            )

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.confidence_gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.visual_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, visual_dim, 1, bias=False),
            nn.GroupNorm(min(32, visual_dim), visual_dim)
        )

        self.dropout = nn.Dropout(dropout)
        # V5: 固定残差缩放
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

    def _apply_deformable_sampling(self, feat: torch.Tensor,
                                    offset: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device)
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        offset_normalized = offset / torch.tensor([W/2, H/2], device=feat.device)
        deformed_grid = base_grid + offset_normalized

        sampled = F.grid_sample(
            feat, deformed_grid,
            mode='bilinear', padding_mode='border', align_corners=False
        )

        return sampled

    def forward(self, visual_feat: torch.Tensor,
                refined_text: torch.Tensor,
                anchor_feat: torch.Tensor,
                iteration: int = 0) -> torch.Tensor:
        B, C, H, W = visual_feat.shape
        residual = visual_feat

        if self.use_deformable:
            offset = self.offset_module(visual_feat, anchor_feat)
            visual_deformed = self._apply_deformable_sampling(visual_feat, offset)
        else:
            visual_deformed = visual_feat

        visual_encoded = self.visual_encoder(visual_deformed)
        text_encoded = self.text_encoder(refined_text)

        visual_flat = visual_encoded.flatten(2).transpose(1, 2)
        text_expanded = text_encoded.unsqueeze(1)

        Q = self.q_proj(visual_flat)
        K = self.k_proj(text_expanded)
        V = self.v_proj(text_expanded)

        Q = Q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, H*W, self.hidden_dim)
        out = self.out_proj(out)

        visual_global = visual_flat.mean(dim=1)
        confidence_input = torch.cat([visual_global, text_encoded], dim=-1)
        confidence = self.confidence_gate(confidence_input)
        confidence = self.confidence_baseline + (1 - self.confidence_baseline) * confidence

        out = out.transpose(1, 2).view(B, self.hidden_dim, H, W)
        refined = self.visual_decoder(out)

        # V5: 固定残差缩放 + 迭代衰减
        decay = 1.0 / (1.0 + 0.15 * iteration)
        effective_scale = self.residual_scale * decay * confidence.view(B, 1, 1, 1)

        return residual + effective_scale * refined


class VisualRefinementModuleV5(nn.Module):
    """多尺度视觉精炼模块 V5"""

    def __init__(self, scale_channels: Tuple[int, ...], text_dim: int,
                 anchor_dim: int, hidden_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.11, use_deformable: bool = True,
                 residual_scale: float = 0.07, confidence_baseline: float = 0.65):
        super().__init__()

        self.num_scales = len(scale_channels)
        max_offsets = [1.5, 3.0, 4.5, 6.0][:self.num_scales]

        self.scale_refiners = nn.ModuleList([
            DeformableVisualRefinerV5(
                visual_dim=ch,
                text_dim=text_dim,
                anchor_dim=anchor_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_offset=max_offset,
                use_deformable=use_deformable,
                residual_scale=residual_scale,
                confidence_baseline=confidence_baseline
            ) for ch, max_offset in zip(scale_channels, max_offsets)
        ])

    def forward(self, scale_features: List[torch.Tensor],
                refined_text: torch.Tensor,
                anchor_feat: torch.Tensor,
                iteration: int = 0) -> List[torch.Tensor]:
        refined_features = []
        for feat, refiner in zip(scale_features, self.scale_refiners):
            refined = refiner(feat, refined_text, anchor_feat, iteration)
            refined_features.append(refined)
        return refined_features


class AlignmentStateEstimatorV5(nn.Module):
    """对齐状态估计器 V5"""

    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 256, dropout: float = 0.11):
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.alignment_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_context: torch.Tensor,
                text_embedding: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(visual_context)
        t = self.text_proj(text_embedding)

        v_norm = F.normalize(v, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)

        concat = torch.cat([v_norm, t_norm], dim=-1)
        score = self.alignment_scorer(concat)

        return score


class ScaleAnchorAffinityModuleV5(nn.Module):
    """尺度-锚点亲和度模块 V5"""

    def __init__(self, num_scales: int, num_anchors: int, learnable: bool = True):
        super().__init__()

        self.num_scales = num_scales
        self.num_anchors = num_anchors

        init_affinity = torch.zeros(num_scales, num_anchors)
        if num_anchors == 3 and num_scales == 4:
            init_affinity[0] = torch.tensor([0.7, 0.2, 0.1])
            init_affinity[1] = torch.tensor([0.4, 0.4, 0.2])
            init_affinity[2] = torch.tensor([0.2, 0.4, 0.4])
            init_affinity[3] = torch.tensor([0.1, 0.3, 0.6])

        if learnable:
            self.affinity_logits = nn.Parameter(torch.log(init_affinity + 1e-6))
        else:
            self.register_buffer('affinity_logits', torch.log(init_affinity + 1e-6))

    def forward(self) -> torch.Tensor:
        return F.softmax(self.affinity_logits, dim=-1)


class AdaptiveScaleFusionV5(nn.Module):
    """自适应尺度融合 V5"""

    def __init__(self, scale_channels: Tuple[int, ...], learnable: bool = True, dropout: float = 0.11):
        super().__init__()
        self.num_scales = len(scale_channels)
        self.output_channels = scale_channels[0]

        self.channel_projections = nn.ModuleList()
        for channels in scale_channels:
            if channels != self.output_channels:
                self.channel_projections.append(
                    nn.Sequential(
                        nn.Conv2d(channels, self.output_channels, 1, bias=False),
                        nn.GroupNorm(min(32, self.output_channels), self.output_channels)
                    )
                )
            else:
                self.channel_projections.append(nn.Identity())

        if learnable:
            init_weights = torch.tensor([0.10, 0.20, 0.35, 0.35])[:self.num_scales]
            self.scale_weights = nn.Parameter(torch.log(init_weights))
        else:
            self.register_buffer('scale_weights', torch.ones(self.num_scales) / self.num_scales)

        self.fusion_refine = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, self.output_channels), self.output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, self.output_channels), self.output_channels)
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.scale_weights, dim=0)
        target_size = scale_features[0].shape[-2:]

        fused = 0
        for i, feat in enumerate(scale_features):
            feat = self.channel_projections[i](feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused = fused + weights[i] * feat

        return self.fusion_refine(fused)


class AnchorCoherenceModuleV5(nn.Module):
    """锚点一致性模块 V5"""

    def __init__(self, hidden_dim: int, num_anchors: int, dropout: float = 0.11):
        super().__init__()

        self.anchor_alignment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def compute_coherence_loss(self, anchors: torch.Tensor) -> torch.Tensor:
        aligned = self.anchor_alignment(anchors)
        aligned_norm = F.normalize(aligned, p=2, dim=-1)
        sim_matrix = torch.bmm(aligned_norm, aligned_norm.transpose(1, 2))
        target = torch.eye(anchors.shape[1], device=anchors.device).unsqueeze(0)
        target = target.expand(anchors.shape[0], -1, -1)
        coherence_loss = F.mse_loss(sim_matrix, target)
        return coherence_loss


# ============================================================================
# 主模块：CMCE
# ============================================================================

class CMCE(nn.Module):
    """
    CMCE V5: Cross-Modal Co-Evolution - Stable & Optimized Version
    跨模态协同进化 V5 - 稳定优化版

    V5 核心理念：
    1. 稳定性优先：移除自适应模块，使用固定优化参数
    2. 折中策略：dropout=0.11, residual=0.07, max_change=0.12
    3. 效率优化：配合100轮训练和改进的学习率调度

    与 V3/V4 对比：
    - V3: 稳定但 seed=123 表现差
    - V4: seed=123 好但 seed=666 崩溃
    - V5: 取折中参数，追求全面稳定
    """

    def __init__(self,
                 scale_channels: Tuple[int, ...] = (256, 512, 512, 512),
                 text_dim: int = 2048,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_anchors: int = 3,
                 max_iterations: int = 2,
                 convergence_threshold: float = 0.02,
                 dropout: float = 0.11,  # V5: 折中值
                 learnable_fusion: bool = True,
                 learnable_affinity: bool = True,
                 coherence_loss_weight: float = 0.1,
                 refinement_loss_weight: float = 0.05,
                 convergence_loss_weight: float = 0.02,
                 text_consistency_weight: float = 0.025,  # V5: 折中值
                 use_deformable: bool = True,
                 residual_scale: float = 0.07,  # V5: 折中值
                 confidence_baseline: float = 0.65,
                 max_text_change_ratio: float = 0.12):  # V5: 更保守
        super().__init__()

        self.num_scales = len(scale_channels)
        self.scale_channels = scale_channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_deformable = use_deformable

        # 损失权重
        self.coherence_loss_weight = coherence_loss_weight
        self.refinement_loss_weight = refinement_loss_weight
        self.convergence_loss_weight = convergence_loss_weight
        self.text_consistency_weight = text_consistency_weight

        # 1. 语义锚点分解器
        self.anchor_decomposer = SemanticAnchorDecomposerV5(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            dropout=dropout
        )

        # 2. 尺度-锚点亲和度
        self.scale_anchor_affinity = ScaleAnchorAffinityModuleV5(
            num_scales=self.num_scales,
            num_anchors=num_anchors,
            learnable=learnable_affinity
        )

        # 3. 锚点一致性模块
        self.anchor_coherence = AnchorCoherenceModuleV5(
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            dropout=dropout
        )

        # 4. 视觉上下文编码器
        self.visual_context_encoder = VisualContextEncoderV5(
            scale_channels=scale_channels,
            context_dim=hidden_dim,
            dropout=dropout
        )

        # 5. 不一致性检测器
        self.inconsistency_detector = InconsistencyDetectorV5(
            visual_dim=hidden_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 6. 文本精炼模块 V5
        self.text_refiner = TextRefinementModuleV5(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            residual_scale=residual_scale,
            max_change_ratio=max_text_change_ratio
        )

        # 7. 可变形视觉精炼模块 V5
        self.visual_refiner = VisualRefinementModuleV5(
            scale_channels=scale_channels,
            text_dim=text_dim,
            anchor_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_deformable=use_deformable,
            residual_scale=residual_scale,
            confidence_baseline=confidence_baseline
        )

        # 8. 对齐状态估计器
        self.alignment_estimator = AlignmentStateEstimatorV5(
            visual_dim=hidden_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 9. 尺度融合
        self.scale_fusion = AdaptiveScaleFusionV5(
            scale_channels=scale_channels,
            learnable=learnable_fusion,
            dropout=dropout
        )

        logger.info(f"CMCE V5 initialized (Stable & Optimized Version):")
        logger.info(f"  - Scales: {len(scale_channels)}")
        logger.info(f"  - Anchors: {num_anchors}")
        logger.info(f"  - Max iterations: {max_iterations}")
        logger.info(f"  - Convergence threshold: {convergence_threshold}")
        logger.info(f"  - Dropout: {dropout} (V5: balanced)")
        logger.info(f"  - Residual scale: {residual_scale} (V5: balanced)")
        logger.info(f"  - Max text change ratio: {max_text_change_ratio} (V5: conservative)")
        logger.info(f"  - Deformable: {use_deformable}")
        logger.info(f"  - Confidence baseline: {confidence_baseline}")

    def forward(self, scale_features: List[torch.Tensor],
                text_embeddings: torch.Tensor,
                compute_contrastive: bool = False
                ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        B = text_embeddings.shape[0]

        # Step 0: 语义锚点分解
        anchors = self.anchor_decomposer(text_embeddings)
        affinity = self.scale_anchor_affinity()

        # 为每个尺度计算加权锚点
        scale_anchors = []
        for s in range(self.num_scales):
            weights = affinity[s].view(1, self.num_anchors, 1)
            weighted_anchor = (anchors * weights).sum(dim=1)
            scale_anchors.append(weighted_anchor)

        # 初始化
        current_text = text_embeddings
        original_text = text_embeddings
        current_features = scale_features
        alignment_scores = []
        inconsistency_scores = []
        text_changes = []
        prev_alignment = None

        # 协同进化迭代
        for t in range(self.max_iterations):
            # Step 1: 提取视觉上下文
            visual_context = self.visual_context_encoder(current_features)

            # Step 2: 检测不一致性
            inconsistency_vec, inconsistency_score = self.inconsistency_detector(
                visual_context, current_text
            )
            inconsistency_scores.append(inconsistency_score)

            # Step 3: 精炼文本
            refined_text, text_change = self.text_refiner(
                current_text, inconsistency_vec, inconsistency_score
            )
            text_changes.append(text_change)

            # Step 4: 精炼视觉
            refined_features = []
            for s, (feat, anchor) in enumerate(zip(current_features, scale_anchors)):
                refiner = self.visual_refiner.scale_refiners[s]
                refined = refiner(feat, refined_text, anchor, iteration=t)
                refined_features.append(refined)

            # Step 5: 评估对齐状态
            refined_context = self.visual_context_encoder(refined_features)
            alignment_score = self.alignment_estimator(refined_context, refined_text)
            alignment_scores.append(alignment_score)

            # 更新状态
            current_text = refined_text
            current_features = refined_features

            # 收敛检测（推理时）
            if not self.training:
                if prev_alignment is not None:
                    delta = (alignment_score - prev_alignment).abs().mean()
                    if delta < self.convergence_threshold:
                        break
                prev_alignment = alignment_score

        # 计算辅助损失
        if compute_contrastive:
            aux_loss = self._compute_auxiliary_loss(
                anchors, alignment_scores, inconsistency_scores,
                text_changes, original_text, current_text
            )
            return current_features, aux_loss
        else:
            return current_features, None

    def _compute_auxiliary_loss(self,
                                anchors: torch.Tensor,
                                alignment_scores: List[torch.Tensor],
                                inconsistency_scores: List[torch.Tensor],
                                text_changes: List[torch.Tensor],
                                original_text: torch.Tensor,
                                final_text: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=anchors.device)

        # 1. 锚点一致性损失
        coherence_loss = self.anchor_coherence.compute_coherence_loss(anchors)
        loss = loss + self.coherence_loss_weight * coherence_loss

        # 2. 对齐损失
        final_alignment = alignment_scores[-1]
        alignment_loss = (1 - final_alignment).mean()
        loss = loss + self.refinement_loss_weight * alignment_loss

        # 3. 收敛损失
        if len(alignment_scores) > 1:
            convergence_loss = torch.tensor(0.0, device=anchors.device)
            for i in range(1, len(alignment_scores)):
                decrease = F.relu(alignment_scores[i-1] - alignment_scores[i])
                convergence_loss = convergence_loss + decrease.mean()
            loss = loss + self.convergence_loss_weight * convergence_loss

        # 4. 不一致性正则化
        avg_inconsistency = torch.stack(inconsistency_scores).mean()
        inconsistency_reg = avg_inconsistency ** 2
        loss = loss + 0.01 * inconsistency_reg

        # 5. 文本一致性损失
        text_sim = F.cosine_similarity(
            F.normalize(original_text, dim=-1),
            F.normalize(final_text, dim=-1),
            dim=-1
        ).mean()
        text_consistency_loss = 1 - text_sim
        loss = loss + self.text_consistency_weight * text_consistency_loss

        # 6. 惩罚过大的文本变化
        if text_changes:
            avg_text_change = sum(text_changes) / len(text_changes)
            loss = loss + 0.008 * avg_text_change  # V5: 折中值

        return loss

    def get_evolution_trace(self, scale_features: List[torch.Tensor],
                            text_embeddings: torch.Tensor) -> Dict:
        """获取协同进化过程的详细轨迹"""
        trace = {
            'alignment_scores': [],
            'inconsistency_scores': [],
            'text_changes': [],
            'iterations': 0
        }

        anchors = self.anchor_decomposer(text_embeddings)
        affinity = self.scale_anchor_affinity()
        scale_anchors = []
        for s in range(self.num_scales):
            weights = affinity[s].view(1, self.num_anchors, 1)
            weighted_anchor = (anchors * weights).sum(dim=1)
            scale_anchors.append(weighted_anchor)

        current_text = text_embeddings
        current_features = scale_features
        prev_text = text_embeddings.clone()
        prev_alignment = None

        for t in range(self.max_iterations):
            visual_context = self.visual_context_encoder(current_features)
            inconsistency_vec, inconsistency_score = self.inconsistency_detector(
                visual_context, current_text
            )
            refined_text, _ = self.text_refiner(
                current_text, inconsistency_vec, inconsistency_score
            )

            refined_features = []
            for s, (feat, anchor) in enumerate(zip(current_features, scale_anchors)):
                refiner = self.visual_refiner.scale_refiners[s]
                refined = refiner(feat, refined_text, anchor, iteration=t)
                refined_features.append(refined)

            refined_context = self.visual_context_encoder(refined_features)
            alignment_score = self.alignment_estimator(refined_context, refined_text)

            trace['alignment_scores'].append(alignment_score.mean().item())
            trace['inconsistency_scores'].append(inconsistency_score.mean().item())
            trace['text_changes'].append(
                (refined_text - prev_text).norm(dim=-1).mean().item()
            )
            trace['iterations'] = t + 1

            if prev_alignment is not None:
                delta = abs(alignment_score.mean().item() - prev_alignment)
                if delta < self.convergence_threshold:
                    break
            prev_alignment = alignment_score.mean().item()

            current_text = refined_text
            current_features = refined_features
            prev_text = refined_text.clone()

        return trace


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing CMCE V5 Module (Stable & Optimized Version)")
    print("=" * 80)

    batch_size = 2
    scale_channels = (256, 512, 512, 512)
    text_dim = 2048
    spatial_sizes = [(128, 128), (64, 64), (32, 32), (32, 32)]

    scale_features = [
        torch.randn(batch_size, channels, *size)
        for channels, size in zip(scale_channels, spatial_sizes)
    ]
    text_embeddings = torch.randn(batch_size, text_dim)

    model = CMCE(
        scale_channels=scale_channels,
        text_dim=text_dim,
        hidden_dim=256,
        num_heads=4,
        num_anchors=3,
        max_iterations=2,
        convergence_threshold=0.02,
        dropout=0.11,  # V5
        use_deformable=True,
        residual_scale=0.07,  # V5
        confidence_baseline=0.65,
        max_text_change_ratio=0.12  # V5
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nForward pass...")
    enhanced_features, aux_loss = model(
        scale_features, text_embeddings, compute_contrastive=True
    )

    print(f"\nInput scales:")
    for i, feat in enumerate(scale_features):
        print(f"  Scale {i}: {feat.shape}")

    print(f"\nEnhanced scales:")
    for i, feat in enumerate(enhanced_features):
        print(f"  Scale {i}: {feat.shape}")

    print(f"\nAuxiliary loss: {aux_loss.item():.6f}")

    print("\nEvolution trace:")
    trace = model.get_evolution_trace(scale_features, text_embeddings)
    for i, (align, incon, change) in enumerate(zip(
        trace['alignment_scores'],
        trace['inconsistency_scores'],
        trace['text_changes']
    )):
        print(f"  Iteration {i+1}: alignment={align:.4f}, "
              f"inconsistency={incon:.4f}, text_change={change:.4f}")

    fused = model.scale_fusion(enhanced_features)
    print(f"\nFused features: {fused.shape}")

    print("\n" + "=" * 80)
    print("CMCE V5 KEY FEATURES")
    print("=" * 80)
    print("V5 Philosophy: Stability First + Balanced Parameters")
    print()
    print("Parameter Comparison:")
    print("  V3: dropout=0.12, residual=0.06, max_change=0.15")
    print("  V4: adaptive dropout/residual, learnable constraint")
    print("  V5: dropout=0.11, residual=0.07, max_change=0.12 (balanced)")
    print()
    print("Training Optimization:")
    print("  - 100 epochs (vs 150)")
    print("  - Improved LR scheduling")
    print("  - Warmup + Cosine with fixed period")
    print("=" * 80)
    print("\nCMCE V5 test passed!")
