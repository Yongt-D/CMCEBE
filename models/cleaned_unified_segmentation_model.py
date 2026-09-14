# models/cleaned_unified_segmentation_model.py

from typing import Dict, Optional, Any, List, Tuple
import logging
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import alignment modules
from alignment.simple_alignment import SimpleFeatureConcatenation
from alignment.dynamic_alignment import DynamicFeatureAlignmentModel
from alignment.generative_alignment import MultimodalGenerativeAlignmentModel
from alignment.cmce import CMCE  # CMCE is the final stable version (formerly v5)
from alignment.daca import DACA  # DACA: Dynamic Anchor-aware Cross-modal Alignment (new)
# Import cleaned loss function
from utils.unified_loss import UnifiedSegmentationLoss
from models.backbones.unet import UNet

logger = logging.getLogger(__name__)


# ---------------- Factory (unchanged) ----------------
class AlignmentFactory:
    @staticmethod
    def create_alignment(alignment_type: str, config: Dict):
        if alignment_type in ('simple',):
            return SimpleFeatureConcatenation(
                image_dim=config.get('image_channels', 512),
                text_dim=config.get('text_dim', 2048),
                output_dim=config.get('output_dim', config.get('image_channels', 512)),
                dropout=config.get('dropout', 0.1),
            )
        elif alignment_type in ('dynamic',):
            return DynamicFeatureAlignmentModel(
                image_dim=config.get('image_channels', 512),
                text_dim=config.get('text_dim', 2048),
                embed_dim=config.get('embed_dim', 256),
                num_heads=config.get('num_attention_heads', 8),
                dropout=config.get('dropout', 0.1),
                use_layer_norm=config.get('use_layer_norm', True),
            )
        elif alignment_type in ('generative',):
            return MultimodalGenerativeAlignmentModel(input_dim=config.get('image_channels', 512),
                                                      text_dim=config.get('text_dim', 2048),
                                                      latent_dim=config.get('latent_dim', 256),
                                                      hidden_dim=config.get('hidden_dim', 384),
                                                      num_layers=config.get('num_layers', 3),
                                                      dropout=config.get('dropout', 0.1),
                                                      use_vae=config.get('use_vae', True))
        elif alignment_type in ('none', None):
            return None
        # Note: 'cmce', 'daca' have separate initialization logic in the main model
        elif alignment_type in ('cmce', 'daca'):
            # CMCE/DACA have their own initialization path, returning None here is safe
            return None
        else:
            raise ValueError(f'Unsupported alignment type: {alignment_type}')


# ---------------- Cleaned Main Model (modified) ----------------
class CleanedUnifiedSegmentationModel(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1,
            base_channels: int = 64,
            text_dim: int = 2048,
            alignment_type: str = 'none',
            alignment_config: Dict = None,
            alignment_insertion_layer: Any = "bottleneck",
            # Removed boundary loss and semantic consistency related parameters
            attn_gate_alpha: float = 0.6,
            attn_gate_beta: float = 0.4,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.text_dim = text_dim
        self.alignment_type = alignment_type
        self.attn_gate_alpha = float(attn_gate_alpha)
        self.attn_gate_beta = float(attn_gate_beta)

        # Simplified mixing parameters (for old alignment methods)
        self.align_mix_min = float((alignment_config or {}).get('align_mix_min', 0.10))
        self.align_mix_max = float((alignment_config or {}).get('align_mix_max', 0.30))

        if isinstance(alignment_insertion_layer, (list, tuple)):
            self.alignment_layers: List[str] = list(alignment_insertion_layer)
        else:
            self.alignment_layers = [str(alignment_insertion_layer)]

        # +++ CMCE initialization logic +++
        self.cmce_module = None
        self.cmce_seg_head = None
        self.auxiliary_loss_weight = 1.0
        # UNet feature layers used by CMCE (need to correspond to 'scale_channels' in config)
        self.cmce_feature_keys = ['encoder_2', 'encoder_3', 'encoder_4', 'bottleneck']
        if self.alignment_type == 'cmce':
            if CMCE is None:
                raise ImportError("alignment_type='cmce' but CMCE module cannot be imported.")

            self.requires_text_features = True
            self.backbone = UNet(in_channels=in_channels, out_channels=num_classes,
                                 base_channels=base_channels, return_features=True)

            cmce_cfg = dict(alignment_config or {})
            cmce_scale_channels = cmce_cfg.get('scale_channels')
            if cmce_scale_channels is None:
                cmce_scale_channels = (128, 256, 512, 1024)
                logger.warning(f"'scale_channels' not specified in CMCE config, using default: {cmce_scale_channels}")

            if 'feature_keys' in cmce_cfg:
                self.cmce_feature_keys = list(cmce_cfg['feature_keys'])
                if len(self.cmce_feature_keys) != len(cmce_scale_channels):
                    raise ValueError(f"CMCE: Number of 'feature_keys' ({len(self.cmce_feature_keys)}) "
                                     f"must match 'scale_channels' ({len(cmce_scale_channels)}).")

            # Initialize CMCE module (Stable & Optimized Version, formerly V5)
            logger.info("Using CMCE (Stable: balanced parameters + improved LR scheduling)")
            self.cmce_module = CMCE(
                scale_channels=tuple(cmce_scale_channels),
                text_dim=text_dim,
                hidden_dim=cmce_cfg.get('hidden_dim', 256),
                num_heads=cmce_cfg.get('num_heads', 4),
                num_anchors=cmce_cfg.get('num_anchors', 3),
                dropout=cmce_cfg.get('dropout', 0.11),  # V5: balanced
                max_iterations=cmce_cfg.get('max_iterations', 2),
                convergence_threshold=cmce_cfg.get('convergence_threshold', 0.02),
                learnable_fusion=cmce_cfg.get('learnable_fusion', True),
                learnable_affinity=cmce_cfg.get('learnable_affinity', True),
                coherence_loss_weight=cmce_cfg.get('coherence_loss_weight', 0.1),
                refinement_loss_weight=cmce_cfg.get('refinement_loss_weight', 0.05),
                convergence_loss_weight=cmce_cfg.get('convergence_loss_weight', 0.02),
                text_consistency_weight=cmce_cfg.get('text_consistency_weight', 0.025),  # V5: balanced
                use_deformable=cmce_cfg.get('use_deformable', True),
                residual_scale=cmce_cfg.get('residual_scale', 0.07),  # V5: balanced
                confidence_baseline=cmce_cfg.get('confidence_baseline', 0.65),
                max_text_change_ratio=cmce_cfg.get('max_text_change_ratio', 0.12)  # V5: conservative
            )

            fused_channels = cmce_scale_channels[0]
            self.cmce_seg_head = nn.Conv2d(fused_channels, num_classes, kernel_size=1)

            self.contrastive_loss_weight = float(cmce_cfg.get('coherence_loss_weight', 0.1))
            logger.info(f"CMCE module initialized. Using feature layers: {self.cmce_feature_keys}")
            logger.info(f"CMCE dropout: {cmce_cfg.get('dropout', 0.11)}")
            logger.info(f"CMCE residual_scale: {cmce_cfg.get('residual_scale', 0.07)}")
            logger.info(f"CMCE max_text_change_ratio: {cmce_cfg.get('max_text_change_ratio', 0.12)}")

            self.alignment_modules = nn.ModuleDict()
            self.feature_fusions = nn.ModuleDict()
            self.align_to_logits = nn.ModuleDict()

        # +++ DACA initialization logic (Dynamic Anchor-aware Cross-modal Alignment) +++
        elif self.alignment_type == 'daca':
            if DACA is None:
                raise ImportError("alignment_type='daca' but DACA module cannot be imported.")

            self.requires_text_features = True
            self.backbone = UNet(in_channels=in_channels, out_channels=num_classes,
                                 base_channels=base_channels, return_features=True)

            daca_cfg = dict(alignment_config or {})
            daca_scale_channels = daca_cfg.get('scale_channels')
            if daca_scale_channels is None:
                daca_scale_channels = (128, 256, 512, 1024)
                logger.warning(f"'scale_channels' not specified in DACA config, using default: {daca_scale_channels}")

            if 'feature_keys' in daca_cfg:
                self.cmce_feature_keys = list(daca_cfg['feature_keys'])
                if len(self.cmce_feature_keys) != len(daca_scale_channels):
                    raise ValueError(f"DACA: Number of 'feature_keys' ({len(self.cmce_feature_keys)}) "
                                     f"must match 'scale_channels' ({len(daca_scale_channels)}).")

            # Initialize DACA module
            logger.info("Using DACA (Dynamic Anchor-aware Cross-modal Alignment)")
            self.cmce_module = DACA(
                scale_channels=tuple(daca_scale_channels),
                text_dim=text_dim,
                hidden_dim=daca_cfg.get('hidden_dim', 256),
                num_heads=daca_cfg.get('num_heads', 4),
                max_anchors=daca_cfg.get('max_anchors', 5),
                dropout=daca_cfg.get('dropout', 0.10),
                temperature=daca_cfg.get('temperature', 0.07),
                contrastive_weight=daca_cfg.get('contrastive_weight', 0.1),
                complexity_reg_weight=daca_cfg.get('complexity_reg_weight', 0.02)
            )

            # DACA fusion output needs a new segmentation head
            fused_channels = daca_scale_channels[0]
            self.cmce_seg_head = nn.Conv2d(fused_channels, num_classes, kernel_size=1)

            # Use contrastive loss weight
            self.contrastive_loss_weight = float(daca_cfg.get('contrastive_weight', 0.1))
            logger.info(f"DACA module initialized. Using feature layers: {self.cmce_feature_keys}")
            logger.info(f"DACA max_anchors: {daca_cfg.get('max_anchors', 5)}")
            logger.info(f"DACA contrastive_weight: {daca_cfg.get('contrastive_weight', 0.1)}")

            # Clear old alignment modules to prevent confusion
            self.alignment_modules = nn.ModuleDict()
            self.feature_fusions = nn.ModuleDict()
            self.align_to_logits = nn.ModuleDict()

        else:
            # --- Original logic (for none, simple, dynamic, generative) ---
            self.requires_text_features = (alignment_type != 'none')
            self.backbone = UNet(in_channels=in_channels, out_channels=num_classes,
                                 base_channels=base_channels, return_features=self.requires_text_features)

            self.alignment_modules = nn.ModuleDict()
            self.feature_fusions = nn.ModuleDict()
            self.align_to_logits = nn.ModuleDict()

            if self.requires_text_features:
                ch = base_channels * 8  # bottleneck channels for UNet(64) -> 512 (old logic assumption)
                # Note: If old alignment also needs multi-scale, ch here needs modification

                cfg = dict(alignment_config or {})
                cfg.update({'image_channels': ch, 'text_dim': text_dim})

                for layer in self.alignment_layers:
                    # Fix: If alignment_type is 'mscta', AlignmentFactory returns None
                    align_module = AlignmentFactory.create_alignment(alignment_type, cfg)
                    if align_module is None and alignment_type != 'none':
                        logger.warning(f"Cannot create alignment module for type '{alignment_type}' (layer: {layer}).")
                        continue

                    self.alignment_modules[layer] = align_module

                    # Simplified feature fusion
                    self.feature_fusions[layer] = nn.Sequential(
                        nn.Conv2d(ch, ch, 1),
                        nn.ReLU(True),
                        nn.Conv2d(ch, ch, 1)
                    )

                    # Simplified logits projection
                    self.align_to_logits[layer] = nn.Sequential(
                        nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(True),
                        nn.Conv2d(ch, self.num_classes, 1)
                    )

        # D1 baseline (restored - H1/I1 proved loss function is already optimal)
        self.criterion = UnifiedSegmentationLoss(
            use_bce=True,
            use_dice=True,
            use_focal=True,
            use_tversky=True,
            use_boundary=False,  # ❌ Exp1/H1/I1 proved harmful

            # D1 optimal loss weights (DO NOT MODIFY)
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5,
            tversky_weight=0.5,

            # D1 optimal parameters
            focal_alpha=0.25,
            focal_gamma=2.0,

            # D1 optimal Tversky parameters
            tversky_alpha=0.3,
            tversky_beta=0.7,

            auto_adjust=True
        )
        self.injection_progress: float = 1.0

    def set_injection_progress(self, p: float) -> None:
        self.injection_progress = float(max(0.0, min(1.0, p)))

    def get_model_info(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        align = 0

        # +++ Modified: Include CMCE/DACA parameter statistics +++
        if self.alignment_type in ('cmce', 'daca'):
            if self.cmce_module:
                align += sum(p.numel() for p in self.cmce_module.parameters())
            if self.cmce_seg_head:
                align += sum(p.numel() for p in self.cmce_seg_head.parameters())
        else:
            for m in self.alignment_modules.values():
                if m: align += sum(p.numel() for p in m.parameters())
            for m in self.align_to_logits.values():
                if m: align += sum(p.numel() for p in m.parameters())
        # --- End modification ---

        return {
            'backbone': 'UNet',
            'base_channels': self.base_channels,
            'num_classes': self.num_classes,
            'alignment_type': self.alignment_type,
            'alignment_insertion_layer': self.alignment_layers if self.alignment_type not in ('cmce', 'daca') else self.cmce_feature_keys,
            'requires_text_features': self.requires_text_features,
            'params_total': int(total),
            'params_trainable': int(trainable),
            'params_alignment_branch': int(align),
            'cleaned_components': [
                'removed_boundary_loss',
                'removed_semantic_consistency_loss',
                'simplified_loss_computation',
                'kept_effective_losses: BCE+Dice+Focal+Tversky'
            ]
        }

    # --- (Old helper methods, unchanged) ---
    def _pick_target(self, feats: Dict[str, torch.Tensor], layer: str) -> Optional[torch.Tensor]:
        if layer == 'bottleneck':
            return feats.get('bottleneck', feats.get('encoder_4'))
        if layer == 'encoder':
            return feats.get('encoder_3')
        return None

    def _attention_gate(self, layer: str, fused: torch.Tensor, base_shape: Tuple[int, int]) -> torch.Tensor:
        H, W = base_shape
        attn_gate = None
        try:
            maps = self.alignment_modules[layer].get_attention_maps() if hasattr(self.alignment_modules[layer],
                                                                                 'get_attention_maps') else None
            if maps and ('spatial' in maps or 'channel' in maps):
                spatial = maps.get('spatial', torch.zeros_like(fused[:, :1]))
                channel = maps.get('channel', torch.zeros_like(fused[:, :1]))
                g = self.attn_gate_alpha * spatial + self.attn_gate_beta * channel
                attn_gate = torch.sigmoid(F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False))
        except Exception as e:
            logger.debug(f"attention gate fallback: {e}")

        if attn_gate is None:
            g = torch.sigmoid(torch.mean(fused, dim=1, keepdim=True))
            attn_gate = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        return attn_gate

    def _inject_from_layer(self, layer: str, target: torch.Tensor, text: torch.Tensor, base_logits: torch.Tensor):
        """Simplified feature injection (removed complex calibration mechanism)"""
        aligned = self.alignment_modules[layer](target, text)
        enhanced = self.feature_fusions[layer](aligned)

        mix = self.align_mix_min + (self.align_mix_max - self.align_mix_min) * float(self.injection_progress)
        mix = max(0.0, min(0.95, mix))
        fused = (1.0 - mix) * target + mix * enhanced

        H, W = base_logits.shape[-2:]
        gate = self._attention_gate(layer, fused, (H, W))
        inj = self.align_to_logits[layer](fused)

        if inj.shape[-2:] != (H, W):
            inj = F.interpolate(inj, size=(H, W), mode='bilinear', align_corners=False)
        if gate.shape[-2:] != (H, W):
            gate = F.interpolate(gate, size=(H, W), mode='bilinear', align_corners=False)

        return base_logits + self.injection_progress * (gate * inj)

    # --- (Old helper methods end) ---

    # +++ Modified forward method +++
    def forward(self, images: torch.Tensor, text_features: Optional[torch.Tensor] = None,
                return_intermediate: bool = False):
        inter: Dict[str, Any] = {}

        # =======================================================
        # +++ 1. CMCE/DACA specific logic path +++
        # =======================================================
        if self.alignment_type in ('cmce', 'daca'):
            if text_features is None or not self.cmce_module:
                # Fallback: If no text or module not initialized, run standard U-Net
                logits = self.backbone(images)
                if return_intermediate:
                    inter['alignment_type'] = self.alignment_type
                    inter['text_features_used'] = False
                    inter[
                        'fallback_reason'] = 'text_features_missing' if text_features is None else 'mscta_module_missing'
                return (logits, inter) if return_intermediate else logits

            # 1. Get features from U-Net encoder
            # We ignore U-Net logits (first return value), only use encoder features (feats)
            _, feats = self.backbone.forward_with_features(images)

            # 2. Map U-Net features to MSCTA expected input list
            try:
                mscta_input_features = [feats[key] for key in self.cmce_feature_keys]
            except KeyError as e:
                logger.error(f"MSCTA failed: U-Net feature key '{e}' not found. Available keys: {list(feats.keys())}")
                # Fallback to standard U-Net
                logits = self.backbone(images)
                if return_intermediate:
                    inter['alignment_error'] = f"Missing feature key: {e}"
                return (logits, inter) if return_intermediate else logits

            # 3. Run through MSCTA module
            # Compute contrastive loss during training (compute_contrastive=True)
            # Can set to False during inference (controlled by return_intermediate)
            compute_loss = self.training or return_intermediate

            # MSCTA forward pass
            enhanced_features, contrastive_loss = self.cmce_module.forward(
                mscta_input_features,
                text_features,
                compute_contrastive=compute_loss
            )

            # 4. Get adaptively fused feature map
            # scale_fusion automatically upsamples and fuses
            fused_map = self.cmce_module.scale_fusion(enhanced_features)

            # 5. Generate logits through new segmentation head
            logits = self.cmce_seg_head(fused_map)

            # 6. Upsample logits to input image size (MSCTA fused map may not be full size)
            if logits.shape[-2:] != images.shape[-2:]:
                logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)

            # 7. Store intermediate results
            if return_intermediate:
                inter['alignment_type'] = 'mscta'
                inter['text_features_used'] = True
                # Only store loss when needed (e.g., during training)
                if compute_loss:
                    inter['contrastive_loss'] = contrastive_loss
                inter['fused_map_shape'] = fused_map.shape
                inter['enhanced_features_shapes'] = [f.shape for f in enhanced_features]

            return (logits, inter) if return_intermediate else logits

        # =======================================================
        # +++ 2. Original logic (none, simple, dynamic, generative) +++
        # =======================================================
        if not self.requires_text_features:
            logits = self.backbone(images)
            return (logits, inter) if return_intermediate else logits

        if text_features is None:
            logits = self.backbone(images)
            if return_intermediate:
                inter.update({'alignment_type': self.alignment_type, 'text_features_used': False,
                              'fallback_reason': 'text_features_missing'})
                return logits, inter
            return logits

        if hasattr(self.backbone, 'forward_with_features'):
            logits, feats = self.backbone.forward_with_features(images)
            inter['unet_features'] = feats
        else:
            logits = self.backbone(images)
            feats = None

        if self.alignment_modules and feats is not None:
            for layer in self.alignment_layers:
                tgt = self._pick_target(feats, layer)
                if tgt is None or layer not in self.alignment_modules:
                    continue
                try:
                    logits = self._inject_from_layer(layer, tgt, text_features, logits)
                    inter['text_features_used'] = True
                except Exception as e:
                    logger.warning(f"inject fail @ {layer}: {e}")
                    inter['alignment_error'] = str(e)
                    inter['text_features_used'] = False

        if return_intermediate:
            inter['alignment_type'] = self.alignment_type
            inter['alignment_layers'] = self.alignment_layers
            return logits, inter
        return logits

    # +++ Modified compute_loss method +++
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                     intermediate_results: Optional[Dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Use cleaned loss function, optionally add contrastive loss"""

        # 1. Compute standard segmentation loss
        # criterion returns (total_seg_loss, components_dict)
        seg_total_loss, loss_components = self.criterion(predictions, targets, return_components=True)

        # 2. Check and add MSCTA contrastive loss
        if intermediate_results and 'contrastive_loss' in intermediate_results:
            c_loss = intermediate_results['contrastive_loss']

            if c_loss is not None and torch.is_tensor(c_loss) and c_loss.requires_grad:
                c_loss_weighted = c_loss * self.contrastive_loss_weight

                # Store scalar values for logging
                loss_components['contrastive_loss'] = c_loss.item()
                loss_components['contrastive_loss_w'] = c_loss_weighted.item()

                # Add contrastive loss to total loss
                # 'total_loss' key will be used by CleanedUnifiedTrainer for backpropagation
                loss_components['total_loss'] = seg_total_loss + c_loss_weighted
            else:
                # If 'contrastive_loss' exists but invalid (e.g., None or 0)
                loss_components['contrastive_loss'] = 0.0
                loss_components['total_loss'] = seg_total_loss

        return loss_components


# ---------------- Cleaned Trainer (unchanged) ----------------
# (CleanedUnifiedTrainer class code remains unchanged,
#  because it already uses loss_components['total_loss']
#  to get the total loss, so it automatically handles MSCTA combined loss)
# --------------------------------------------------------
class CleanedUnifiedTrainer:
    def __init__(self, model: CleanedUnifiedSegmentationModel, device, config: Dict, logger=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.logger = logger

        tr = config.get('training', {})
        lr = float(tr.get('learning_rate', 1e-4))
        wd = float(tr.get('weight_decay', 1e-4))
        t0 = int(tr.get('scheduler_t0', 10))
        t_mult = int(tr.get('scheduler_tmult', 2))
        eta_min = float(tr.get('min_lr', 3e-6))
        self.injection_warmup_epochs = int(tr.get('injection_warmup_epochs', 10))

        # Simplified optimizer setup
        # +++ Modified: Add separate parameter groups for CMCE/DACA +++
        if self.model.alignment_type in ('cmce', 'daca'):
            align_params = []
            if self.model.cmce_module:
                align_params += list(self.model.cmce_module.parameters())
            if self.model.cmce_seg_head:
                align_params += list(self.model.cmce_seg_head.parameters())

            base_params = [p for n, p in self.model.named_parameters() if
                           p.requires_grad and 'cmce_module' not in n and 'cmce_seg_head' not in n]

            self.optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': lr, 'weight_decay': wd},
                {'params': align_params, 'lr': lr, 'weight_decay': wd},  # CMCE/DACA module uses same learning rate as backbone
            ])

        elif self.model.alignment_type != 'none' and len(self.model.alignment_modules) > 0:
            align_params = []
            for m in self.model.alignment_modules.values():
                if m: align_params += list(m.parameters())
            for m in self.model.align_to_logits.values():
                if m: align_params += list(m.parameters())

            base_params = [p for n, p in self.model.named_parameters() if
                           p.requires_grad and 'alignment_modules' not in n and 'align_to_logits' not in n]

            self.optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': lr, 'weight_decay': wd},
                {'params': align_params, 'lr': lr * 0.5, 'weight_decay': wd},  # Old alignment uses 0.5x learning rate
            ])
        else:
            # 'none' alignment type
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        # --- Optimizer modification end ---

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
        )

        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.best_val_iou = 0.0
        self.best_epoch = -1
        self.best_eval_threshold = 0.5

        # Simplified validation thresholds
        self.val_thresholds = [round(0.4 + 0.01 * i, 2) for i in range(21)]

    def _parse_batch(self, batch):
        dev = self.device
        img = msk = txt = None

        if isinstance(batch, (list, tuple)):
            img, msk = batch[0].to(dev), batch[1].to(dev)
            if len(batch) > 2 and batch[2] is not None:
                txt = batch[2].to(dev)
        elif isinstance(batch, dict):
            for k in ['image', 'img', 'images', 'x']:
                if k in batch:
                    img = batch[k].to(dev)
                    break
            for k in ['mask', 'masks', 'label', 'labels', 'target', 'targets', 'y']:
                if k in batch:
                    msk = batch[k].to(dev)
                    break
            for k in ['text_features', 'text_feature', 'txt', 'text']:
                if k in batch and batch[k] is not None:
                    txt = batch[k].to(dev)
                    break
            if img is None or msk is None:
                raise KeyError(f"Batch dict missing keys: {list(batch.keys())}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        return img, msk, txt

    def _run_epoch(self, loader, is_training=True):
        self.model.train(is_training)
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)

        if is_training:
            if self.injection_warmup_epochs > 0 and hasattr(self.model, 'set_injection_progress'):
                ep_done = len(self.train_history.get('iou', []))
                progress = min(1.0, (ep_done + 1) / float(self.injection_warmup_epochs))
                # set_injection_progress only affects old alignment methods, but calling it is harmless
                self.model.set_injection_progress(progress)

            for batch in loader:
                images, masks, text_features = self._parse_batch(batch)
                self.optimizer.zero_grad()

                predictions, inter = self.model.forward(images, text_features, return_intermediate=True)
                loss_comps = self.model.compute_loss(predictions, masks, inter)
                total_loss = loss_comps['total_loss']

                for k, v in loss_comps.items():
                    if isinstance(v, torch.Tensor):
                        epoch_losses[k].append(v.item())
                    elif isinstance(v, (int, float)):
                        epoch_losses[k].append(v)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    probs = torch.sigmoid(predictions)
                    preds = (probs > 0.5).float()
                    tp = (preds * masks).sum().item()
                    fp = (preds * (1 - masks)).sum().item()
                    fn = ((1 - preds) * masks).sum().item()
                    iou = tp / (tp + fp + fn + 1e-6)
                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)
                    epoch_metrics['iou'].append(iou)
                    epoch_metrics['precision'].append(precision)
                    epoch_metrics['recall'].append(recall)
                    epoch_metrics['f1'].append(f1)

            avg = {'lr': self.optimizer.param_groups[0]['lr']}
            for k, v in epoch_losses.items():
                avg[k] = sum(v) / max(1, len(v))
            for k, v in epoch_metrics.items():
                avg[k] = sum(v) / max(1, len(v))
            # Add 'loss' key for logging (alias of 'total_loss')
            if 'total_loss' in avg:
                avg['loss'] = avg['total_loss']
            return avg

        # Validation process
        self.model.eval()
        epoch_losses.clear()
        scan_tp = [0.0 for _ in self.val_thresholds]
        scan_fp = [0.0 for _ in self.val_thresholds]
        scan_fn = [0.0 for _ in self.val_thresholds]

        with torch.no_grad():
            for batch in loader:
                images, masks, text_features = self._parse_batch(batch)
                masks = masks.float()
                if masks.max() > 1.0:
                    masks = (masks > 127.5).float()

                logits = self.model.forward(images, text_features, return_intermediate=False)
                loss_comps = self.model.compute_loss(logits, masks, None)

                for k, v in loss_comps.items():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().item() if v.numel() == 1 else float(v.mean().item())
                    epoch_losses[k] = epoch_losses.get(k, []) + [float(v)]

                probs = torch.sigmoid(logits)
                for i, th in enumerate(self.val_thresholds):
                    preds = (probs > th).float()
                    scan_tp[i] += float((preds * masks).sum().item())
                    scan_fp[i] += float((preds * (1.0 - masks)).sum().item())
                    scan_fn[i] += float(((1.0 - preds) * masks).sum().item())

        best_iou, best_idx = -1.0, -1
        best_prec = best_rec = best_f1 = 0.0

        for i, th in enumerate(self.val_thresholds):
            denom = scan_tp[i] + scan_fp[i] + scan_fn[i] + 1e-6
            iou = scan_tp[i] / denom
            precision = scan_tp[i] / (scan_tp[i] + scan_fp[i] + 1e-6)
            recall = scan_tp[i] / (scan_tp[i] + scan_fn[i] + 1e-6)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

            if iou > best_iou:
                best_iou, best_idx = iou, i
                best_prec, best_rec, best_f1 = precision, recall, f1

        avg = {
            'iou': float(best_iou),
            'precision': float(best_prec),
            'recall': float(best_rec),
            'f1': float(best_f1),
            'best_threshold': float(self.val_thresholds[best_idx])
        }

        for k, v in epoch_losses.items():
            avg[k] = sum(v) / max(1, len(v))
        # Add 'loss' key for logging (alias of 'total_loss')
        if 'total_loss' in avg:
            avg['loss'] = avg['total_loss']
        return avg

    def train(self, train_loader, val_loader, num_epochs, save_dir):
        import os
        import traceback
        os.makedirs(save_dir, exist_ok=True)

        try:
            for ep in range(num_epochs):
                try:
                    if self.logger:
                        self.logger.info(f"Starting Epoch {ep+1}/{num_epochs}...")

                    # F1: Update training progress for Progressive Contrastive Learning
                    if hasattr(self.model, 'alignment_module') and hasattr(self.model.alignment_module, 'set_training_progress'):
                        self.model.alignment_module.set_training_progress(ep, num_epochs)

                    trm = self._run_epoch(train_loader, True)

                    # Check for NaN in training metrics
                    if torch.isnan(torch.tensor(trm.get('loss', 0.0))) or torch.isinf(torch.tensor(trm.get('loss', 0.0))):
                        if self.logger:
                            self.logger.error(f"NaN/Inf detected in training loss at epoch {ep+1}!")
                            self.logger.error(f"Training metrics: {trm}")
                        # Save emergency checkpoint
                        torch.save({
                            'epoch': ep,
                            'model_state_dict': self.model.state_dict(),
                            'error': 'NaN/Inf in training loss'
                        }, f"{save_dir}/emergency_epoch_{ep+1}.pth")
                        break

                    valm = self._run_epoch(val_loader, False)

                    # Check for NaN in validation metrics
                    if torch.isnan(torch.tensor(valm.get('loss', 0.0))) or torch.isinf(torch.tensor(valm.get('loss', 0.0))):
                        if self.logger:
                            self.logger.error(f"NaN/Inf detected in validation loss at epoch {ep+1}!")
                            self.logger.error(f"Validation metrics: {valm}")
                        break

                    self.scheduler.step()

                    for k, v in trm.items():
                        self.train_history[k].append(v)
                    for k, v in valm.items():
                        self.val_history[k].append(v)

                    # Check for best model
                    is_best = valm['iou'] > self.best_val_iou
                    if is_best:
                        self.best_val_iou = valm['iou']
                        self.best_epoch = ep
                        self.best_eval_threshold = float(valm.get('best_threshold', 0.5))

                        torch.save({
                            'epoch': ep,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_val_iou': self.best_val_iou,
                            'config': self.config,
                            'train_history': dict(self.train_history),
                            'val_history': dict(self.val_history),
                            'best_threshold': self.best_eval_threshold,
                        }, f"{save_dir}/best_model.pth")

                    # Save periodic checkpoint every 10 epochs
                    if (ep + 1) % 10 == 0:
                        checkpoint_path = f"{save_dir}/checkpoint_epoch_{ep+1}.pth"
                        torch.save({
                            'epoch': ep,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_val_iou': self.best_val_iou,
                            'train_history': dict(self.train_history),
                            'val_history': dict(self.val_history),
                        }, checkpoint_path)
                        if self.logger:
                            self.logger.info(f"Saved checkpoint: {checkpoint_path}")

                    # Log epoch results - only essential metrics
                    if self.logger:
                        best_marker = " *" if is_best else ""
                        self.logger.info(
                            f"Epoch {ep+1:3d}/{num_epochs} | "
                            f"Train: Loss={trm.get('loss', 0):.4f} IoU={trm.get('iou', 0):.4f} F1={trm.get('f1', 0):.4f} "
                            f"Pre={trm.get('precision', 0):.4f} Rec={trm.get('recall', 0):.4f} | "
                            f"Val: IoU={valm['iou']:.4f} F1={valm['f1']:.4f} "
                            f"Pre={valm['precision']:.4f} Rec={valm['recall']:.4f}{best_marker}"
                        )
                        # Flush logger to ensure output is written
                        if hasattr(self.logger, 'handlers'):
                            for handler in self.logger.handlers:
                                handler.flush()

                except RuntimeError as e:
                    if self.logger:
                        self.logger.error(f"RuntimeError at epoch {ep+1}: {str(e)}")
                        self.logger.error(traceback.format_exc())
                    # Save emergency checkpoint
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'error': str(e)
                    }, f"{save_dir}/emergency_epoch_{ep+1}.pth")
                    raise

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Unexpected error at epoch {ep+1}: {str(e)}")
                        self.logger.error(traceback.format_exc())
                    # Save emergency checkpoint
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'error': str(e)
                    }, f"{save_dir}/emergency_epoch_{ep+1}.pth")
                    raise

        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("Training interrupted by user (Ctrl+C)")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Training crashed: {str(e)}")
                self.logger.error(traceback.format_exc())
            raise
        finally:
            # Final summary
            if self.logger:
                self.logger.info("=" * 70)
                self.logger.info(
                    f"Training Complete! Best Val: IoU={self.best_val_iou:.4f} "
                    f"(Epoch {self.best_epoch + 1}, threshold={self.best_eval_threshold:.2f})"
                )
                self.logger.info("=" * 70)

        return self.train_history, self.val_history

    def evaluate_test_set(self, test_loader):
        """
        训练结束后评估测试集性能

        Args:
            test_loader: 测试集数据加载器

        Returns:
            dict: 测试集评估结果
        """
        if self.logger:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("EVALUATING ON TEST SET")
            self.logger.info("=" * 70)
            self.logger.info(f"Using threshold: {self.best_eval_threshold:.2f}")
            self.logger.info(f"Test set size: {len(test_loader.dataset)}")

        self.model.eval()
        tp = fp = fn = 0.0

        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(test_loader, desc="Testing", disable=not self.logger):
                images, masks, text_features = self._parse_batch(batch)
                masks = masks.float()
                if masks.max() > 1.0:
                    masks = (masks > 127.5).float()

                logits = self.model.forward(images, text_features, return_intermediate=False)
                probs = torch.sigmoid(logits)
                preds = (probs > self.best_eval_threshold).float()

                tp += float((preds * masks).sum().item())
                fp += float((preds * (1.0 - masks)).sum().item())
                fn += float(((1.0 - preds) * masks).sum().item())

        # 计算指标
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        test_results = {
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(self.best_eval_threshold)
        }

        if self.logger:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("TEST SET RESULTS")
            self.logger.info("=" * 70)
            self.logger.info(f"IoU:       {test_results['iou']:.4f}")
            self.logger.info(f"Precision: {test_results['precision']:.4f}")
            self.logger.info(f"Recall:    {test_results['recall']:.4f}")
            self.logger.info(f"F1:        {test_results['f1']:.4f}")
            self.logger.info(f"Threshold: {test_results['threshold']:.2f}")

            # 计算val-test gap
            val_test_gap = self.best_val_iou - test_results['iou']
            self.logger.info(f"\nGeneralization Gap (Val-Test): {val_test_gap:+.4f} ({val_test_gap/self.best_val_iou*100:+.2f}%)")

            if val_test_gap > 0.02:
                self.logger.info("⚠️  WARNING: Large val-test gap detected! Model may be overfitting.")
            elif val_test_gap < 0:
                self.logger.info("✓ Test performance exceeds validation (good generalization)")
            else:
                self.logger.info("✓ Good generalization to test set")

            self.logger.info("=" * 70)

        return test_results
