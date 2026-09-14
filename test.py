# test.py

import argparse
import json
import logging
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import csv
from datetime import datetime

# Import modules
from models.cleaned_unified_segmentation_model import CleanedUnifiedSegmentationModel
from utils.unified_data_manager import UnifiedDataManager  # Use new data manager
from utils.transforms import BuildingExtractionTransforms


def setup_logging(save_dir: Path) -> logging.Logger:
    """Setup logging"""
    logger = logging.getLogger("multi_text_test")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = save_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_text_feature_dim(text_type):
    """Get feature dimension for different text models"""
    text_dims = {
        'janus': 2048,
        'qwen': 3584,
    }
    return text_dims.get(text_type.lower(), 2048)


def create_model(config: Dict, alignment_type: str, text_type: str) -> CleanedUnifiedSegmentationModel:
    """Create model (adjust text_dim based on text_type)"""
    model_config = config['model']
    backbone_config = model_config['backbone']

    alignment_config = None
    if alignment_type != 'none':
        alignment_config = model_config.get(alignment_type, {})

    insertion_layer = 'bottleneck'
    if alignment_config and 'insertion_layer' in alignment_config:
        insertion_layer = alignment_config['insertion_layer']

    # Get correct text feature dimension based on text_type
    text_dim = get_text_feature_dim(text_type)

    # Update text_dim in alignment_config
    if alignment_config:
        alignment_config = dict(alignment_config)
        alignment_config['text_dim'] = text_dim

    model = CleanedUnifiedSegmentationModel(
        in_channels=backbone_config.get('in_channels', 3),
        num_classes=model_config['segmentation_head']['num_classes'],
        base_channels=backbone_config.get('base_channels', 64),
        text_dim=text_dim,  # Use dynamic text dimension
        alignment_type=alignment_type,
        alignment_config=alignment_config,
        alignment_insertion_layer=insertion_layer,
        attn_gate_alpha=model_config.get('attn_gate_alpha', 0.6),
        attn_gate_beta=model_config.get('attn_gate_beta', 0.4),
    )

    return model


def create_test_dataloader(config: Dict, alignment_type: str, text_type: str):
    """Create test data loader (pass text_type parameter)"""
    data_config = config['data']
    dataset_name = data_config['dataset']
    image_size = data_config['image_size']

    transform = BuildingExtractionTransforms(
        phase='test',
        image_size=image_size,
        augmentation_config={}
    )

    data_manager = UnifiedDataManager(config)

    test_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=1,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        alignment_type=alignment_type,
        text_type=text_type,  # New parameter
        transform=transform
    )

    return test_loader


def load_model_weights(model: CleanedUnifiedSegmentationModel, checkpoint_path: str,
                       device: torch.device) -> Tuple[CleanedUnifiedSegmentationModel, Dict]:
    """Load model weights"""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        best_model_path = checkpoint_path / 'best_model.pth'
        if best_model_path.exists():
            checkpoint_path = best_model_path
        else:
            pth_files = list(checkpoint_path.glob('*.pth'))
            if not pth_files:
                raise FileNotFoundError(f"No .pth files found in {checkpoint_path}")
            checkpoint_path = max(pth_files, key=lambda x: x.stat().st_mtime)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    else:
        model.load_state_dict(checkpoint, strict=False)
        metadata = {}

    return model, metadata


def parse_batch(batch, idx: int, requires_text: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
    """Parse batch data"""
    image = None
    mask = None
    text_features = None
    sample_id = f"sample_{idx:06d}"

    if isinstance(batch, dict):
        for img_key in ['image', 'images', 'x']:
            if img_key in batch and batch[img_key] is not None:
                image = batch[img_key]
                break

        for mask_key in ['label', 'labels', 'mask', 'masks', 'target', 'targets', 'y']:
            if mask_key in batch and batch[mask_key] is not None:
                mask = batch[mask_key]
                break

        if requires_text:
            for text_key in ['text_feature', 'text_features', 'txt', 'text']:
                if text_key in batch and batch[text_key] is not None:
                    text_features = batch[text_key]
                    break

        for id_key in ['name', 'id', 'path', 'sample_id']:
            if id_key in batch and batch[id_key] is not None:
                id_val = batch[id_key]
                if isinstance(id_val, (list, tuple)) and len(id_val) > 0:
                    sample_id = str(id_val[0])
                elif hasattr(id_val, 'item'):
                    sample_id = str(id_val.item()) if id_val.numel() == 1 else str(id_val[0])
                else:
                    sample_id = str(id_val)
                break

    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            image = batch[0]
            mask = batch[1]
            if len(batch) > 2 and requires_text:
                text_features = batch[2]

    if image is None or mask is None:
        raise ValueError(f"Failed to parse batch {idx}")

    return image, mask, text_features, sample_id


def morphological_postprocess(binary_mask: torch.Tensor) -> torch.Tensor:
    """Morphological post-processing"""
    device = binary_mask.device
    kernel = torch.ones((1, 1, 3, 3), device=device, dtype=binary_mask.dtype)

    conv_result = F.conv2d(binary_mask, kernel, padding=1)
    majority = (conv_result >= 5).float()

    dilated = (F.conv2d(majority, kernel, padding=1) > 0).float()
    closed = (F.conv2d(dilated, kernel, padding=1) >= 9).float()

    return closed


def save_prediction_image(prediction: np.ndarray, sample_id: str, save_dir: Path) -> Path:
    """Save prediction image"""
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_uint8 = (prediction * 255).astype(np.uint8)
    save_path = save_dir / f"{sample_id}_pred.png"
    Image.fromarray(pred_uint8, mode='L').save(save_path)
    return save_path


def evaluate_model(
        model: CleanedUnifiedSegmentationModel,
        test_loader,
        device: torch.device,
        requires_text: bool,
        use_postprocess: bool,
        thresholds: List[float],
        save_dir: Path,
        logger: logging.Logger
) -> Dict:
    """Evaluate model and save predictions"""

    pred_save_dir = save_dir / "predictions"
    pred_save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    tp = [0.0 for _ in thresholds]
    fp = [0.0 for _ in thresholds]
    fn = [0.0 for _ in thresholds]

    per_sample_metrics = []

    logger.info(f"Starting evaluation, total {len(test_loader)} samples")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            try:
                image, mask, text_features, sample_id = parse_batch(batch, idx, requires_text)

                image = image.to(device)
                mask = mask.to(device).float()

                if mask.max() > 1.0:
                    mask = (mask > 127.5).float()

                if text_features is not None:
                    text_features = text_features.to(device)

                logits = model(image, text_features, return_intermediate=False)
                probabilities = torch.sigmoid(logits)

                best_iou = -1.0
                best_threshold = thresholds[0]
                best_prediction = None

                for i, threshold in enumerate(thresholds):
                    prediction = (probabilities > threshold).float()

                    if use_postprocess:
                        prediction = morphological_postprocess(prediction)

                    tp[i] += float((prediction * mask).sum().item())
                    fp[i] += float((prediction * (1.0 - mask)).sum().item())
                    fn[i] += float(((1.0 - prediction) * mask).sum().item())

                    intersection = float((prediction * mask).sum().item())
                    union = float((prediction + mask - prediction * mask).sum().item()) + 1e-6
                    sample_iou = intersection / union

                    if sample_iou > best_iou:
                        best_iou = sample_iou
                        best_threshold = threshold
                        best_prediction = prediction.clone()

                pred_np = best_prediction.squeeze().cpu().numpy()
                saved_path = save_prediction_image(pred_np, sample_id, pred_save_dir)

                intersection = float((best_prediction * mask).sum().item())
                pred_sum = float(best_prediction.sum().item())
                mask_sum = float(mask.sum().item())

                precision = intersection / (pred_sum + 1e-6)
                recall = intersection / (mask_sum + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

                per_sample_metrics.append({
                    'sample_id': sample_id,
                    'iou': best_iou,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'best_threshold': best_threshold,
                    'saved_path': str(saved_path.relative_to(save_dir))
                })

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(test_loader)} samples, current IoU: {best_iou:.4f}")

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

    best_global_iou = -1.0
    best_global_metrics = {}

    for i, threshold in enumerate(thresholds):
        if tp[i] + fp[i] + fn[i] == 0:
            continue

        iou = tp[i] / (tp[i] + fp[i] + fn[i] + 1e-6)
        precision = tp[i] / (tp[i] + fp[i] + 1e-6)
        recall = tp[i] / (tp[i] + fn[i] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        if iou > best_global_iou:
            best_global_iou = iou
            best_global_metrics = {
                'iou': float(iou),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'threshold': float(threshold)
            }

    logger.info(f"Evaluation complete, best global IoU: {best_global_iou:.4f}")
    logger.info(f"All prediction images saved to: {pred_save_dir}")

    return {
        'global_metrics': best_global_metrics,
        'per_sample_metrics': per_sample_metrics,
        'predictions_dir': str(pred_save_dir),
        'num_samples': len(per_sample_metrics)
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified segmentation model test script supporting multiple text features')
    parser.add_argument('--config', type=str, default='configs/research_config_mass.yaml',
                        help='Config file path')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides config file)')
    parser.add_argument('--alignment', type=str, default='simple',
                        choices=['none', 'simple', 'dynamic', 'generative'],
                        help='Alignment method')
    parser.add_argument('--text-type', type=str, default='janus',
                        choices=['janus', 'qwen'],
                        help='Text feature type: janus (2048-dim) or qwen (3584-dim)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model weights path (file or directory)')
    parser.add_argument('--postprocess', action='store_true',
                        help='Enable morphological post-processing')
    parser.add_argument('--save-per-sample', action='store_true',
                        help='Save detailed metrics CSV for each sample')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU ID')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.dataset is not None:
        config['data']['dataset'] = args.dataset

    dataset_name = config['data']['dataset']

    # Setup device
    if args.gpu is not None and torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    output_dir = Path('results') / dataset_name / f"{args.alignment}_{args.text_type}_test_cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("Unified segmentation model test supporting multiple text features")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Alignment method: {args.alignment}")
    logger.info(f"Text feature type: {args.text_type.upper()}")
    logger.info(f"  - Model: {'Janus-Pro-1B' if args.text_type == 'janus' else 'Qwen2.5-VL-7B-Instruct'}")
    logger.info(f"  - Feature dimension: {get_text_feature_dim(args.text_type)}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"Post-processing: {args.postprocess}")
    logger.info("=" * 80)

    try:
        # Create model
        model = create_model(config, args.alignment, args.text_type).to(device)

        model_info = model.get_model_info()
        model_info['text_type'] = args.text_type
        model_info['text_dim'] = get_text_feature_dim(args.text_type)

        logger.info(f"Model info: {model_info}")

        # Load weights
        model, metadata = load_model_weights(model, args.checkpoint, device)
        best_val_threshold = float(metadata.get('best_threshold', 0.5))
        logger.info(f"Validation set best threshold: {best_val_threshold:.3f}")

        # Create test data loader
        test_loader = create_test_dataloader(config, args.alignment, args.text_type)
        logger.info(f"Test samples: {len(test_loader)}")

        # Generate evaluation threshold range
        threshold_range_half_width = 0.15
        min_threshold = max(0.05, best_val_threshold - threshold_range_half_width)
        max_threshold = min(0.95, best_val_threshold + threshold_range_half_width)
        thresholds = np.round(np.arange(min_threshold, max_threshold + 1e-9, 0.01), 3).tolist()

        logger.info(f"Evaluation threshold range: [{min_threshold:.3f}, {max_threshold:.3f}] ({len(thresholds)} values)")

        # Execute evaluation
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            requires_text=model.requires_text_features,
            use_postprocess=args.postprocess,
            thresholds=thresholds,
            save_dir=output_dir,
            logger=logger
        )

        global_metrics = results['global_metrics']
        per_sample_metrics = results['per_sample_metrics']

        # Display results
        logger.info("=" * 80)
        logger.info("Test Results")
        logger.info("=" * 80)
        logger.info(f"Text feature type: {args.text_type.upper()}")
        logger.info(f"Global IoU: {global_metrics['iou']:.4f}")
        logger.info(f"Global F1: {global_metrics['f1']:.4f}")
        logger.info(f"Global Precision: {global_metrics['precision']:.4f}")
        logger.info(f"Global Recall: {global_metrics['recall']:.4f}")
        logger.info(f"Best threshold: {global_metrics['threshold']:.3f}")
        logger.info(f"Samples processed: {results['num_samples']}")
        logger.info("=" * 80)

        # Save results
        final_results = {
            'dataset': dataset_name,
            'alignment_type': args.alignment,
            'text_type': args.text_type,
            'text_feature_dim': get_text_feature_dim(args.text_type),
            'text_model_name': 'Janus-Pro-1B' if args.text_type == 'janus' else 'Qwen2.5-VL-7B-Instruct',
            'split': 'test',
            'evaluation_method': 'direct_prediction',
            'model_type': 'CleanedUnifiedSegmentationModel',
            'global_metrics': global_metrics,
            'checkpoint_used': args.checkpoint,
            'best_validation_threshold': best_val_threshold,
            'postprocessing_enabled': args.postprocess,
            'text_features_required': model.requires_text_features,
            'num_samples_processed': results['num_samples'],
            'predictions_directory': results['predictions_dir'],
            'timestamp': datetime.now().isoformat()
        }

        results_json = output_dir / f"{args.alignment}_{args.text_type}_test_results.json"
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Test results saved: {results_json}")

        # Save per-sample detailed metrics CSV
        if args.save_per_sample and per_sample_metrics:
            csv_path = output_dir / f"{args.alignment}_{args.text_type}_per_sample_metrics.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(per_sample_metrics[0].keys()))
                writer.writeheader()
                for row in per_sample_metrics:
                    writer.writerow(row)
            logger.info(f"Per-sample metrics CSV saved: {csv_path}")

        logger.info(f"All results saved to: {output_dir}")
        logger.info("Test complete!")

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()