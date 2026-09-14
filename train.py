# train.py


import argparse
import yaml
import logging
import torch
import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from models.cleaned_unified_segmentation_model import CleanedUnifiedSegmentationModel, CleanedUnifiedTrainer
from utils.unified_data_manager import UnifiedDataManager
from utils.transforms import BuildingExtractionTransforms


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_name(config, alignment_type, text_type, unified_prompt=False, seed=None):
    """Generate experiment name (including text_type, prompt type, and seed info)"""
    base_name = config.get('experiment', {}).get('name', 'cleaned_unified_segmentation')
    prompt_suffix = 'unified' if unified_prompt else 'original'

    # Add seed suffix to avoid checkpoint conflicts
    if seed is not None:
        return f"{base_name}_{alignment_type}_{text_type}_{prompt_suffix}_seed{seed}_cleaned"
    else:
        return f"{base_name}_{alignment_type}_{text_type}_{prompt_suffix}_cleaned"


def setup_logging(dataset_name, experiment_name):
    """Setup logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path('logs') / dataset_name / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / f'{experiment_name}_{timestamp}.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    return logger


def get_text_feature_dim(text_type):
    """Get feature dimension for different text models"""
    text_dims = {
        'janus': 2048,
        'qwen': 3584,
    }
    return text_dims.get(text_type.lower(), 2048)


def create_cleaned_model(config, alignment_type='none', text_type='janus'):
    """Create model (adjust text_dim based on text_type)"""
    model_config = config['model']
    backbone_config = model_config['backbone']
    alignment_config = model_config.get(alignment_type, {}) if alignment_type != 'none' else None

    insertion_layer = 'bottleneck'
    if alignment_config and 'insertion_layer' in alignment_config:
        insertion_layer = alignment_config['insertion_layer']

    text_dim = get_text_feature_dim(text_type)

    if alignment_config:
        alignment_config = dict(alignment_config)
        alignment_config['text_dim'] = text_dim

    model = CleanedUnifiedSegmentationModel(
        in_channels=backbone_config['in_channels'],
        num_classes=model_config['segmentation_head']['num_classes'],
        base_channels=backbone_config['base_channels'],
        text_dim=text_dim,
        alignment_type=alignment_type,
        alignment_config=alignment_config,
        alignment_insertion_layer=insertion_layer,
        attn_gate_alpha=model_config.get('attn_gate_alpha', 0.6),
        attn_gate_beta=model_config.get('attn_gate_beta', 0.4),
    )
    return model


def create_data_loaders(config, alignment_type='none', text_type='janus', unified_prompt=False):
    """
    Create data loaders

    Fix: Added unified_prompt parameter
    """
    data_config = config['data']
    training_config = config['training']
    data_manager = UnifiedDataManager(config)
    dataset_name = data_config['dataset']
    image_size = data_config['image_size']

    # Data augmentation config
    aug_cfg = config.get('augmentation', {})
    train_transform = BuildingExtractionTransforms(
        phase='train',
        image_size=image_size,
        augmentation_config=aug_cfg
    )
    val_transform = BuildingExtractionTransforms(
        phase='val',
        image_size=image_size,
        augmentation_config={}
    )

    # Select feature directory based on unified_prompt
    if unified_prompt:
        # Use unified prompt features
        if text_type == 'janus':
            feature_subdir = 'unified_janus_features'
            text_subdir = 'unified_janus_texts'
        else:  # qwen
            feature_subdir = 'unified_qwen_features'
            text_subdir = 'unified_qwen_texts'
    else:
        # Use original features
        if text_type == 'janus':
            feature_subdir = 'text_features'
            text_subdir = 'text_descriptions'
        else:  # qwen
            feature_subdir = 'qwen_text_features'
            text_subdir = 'qwen_descriptions'

    # Training set
    train_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        sample_count=config['datasets'][dataset_name].get('train_samples', -1),
        alignment_type=alignment_type,
        text_type=text_type,
        transform=train_transform,
        # Pass feature directory info
        text_feature_subdir=feature_subdir,
        text_subdir=text_subdir
    )

    # Validation set
    val_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=training_config.get('val_batch_size', training_config['batch_size']),
        shuffle=False,
        num_workers=data_config['num_workers'],
        alignment_type=alignment_type,
        text_type=text_type,
        transform=val_transform,
        # Pass feature directory info
        text_feature_subdir=feature_subdir,
        text_subdir=text_subdir
    )

    # Test set (for post-training evaluation)
    test_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=training_config.get('val_batch_size', training_config['batch_size']),
        shuffle=False,
        num_workers=data_config['num_workers'],
        alignment_type=alignment_type,
        text_type=text_type,
        transform=val_transform,
        # Pass feature directory info
        text_feature_subdir=feature_subdir,
        text_subdir=text_subdir
    )

    return train_loader, val_loader, test_loader


def save_training_results(dataset_name, experiment_name, train_history, val_history,
                          model_info, config, dirs, text_type, unified_prompt):
    """Save training results (including text_type and prompt info)"""
    results_dir = dirs['results']
    results_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        'dataset': dataset_name,
        'experiment_name': experiment_name,
        'text_feature_type': text_type,
        'text_feature_dim': get_text_feature_dim(text_type),
        'prompt_type': 'unified' if unified_prompt else 'original',
        'model_info': model_info,
        'config': config,
        'train_history': {
            k: [float(x) if torch.is_tensor(x) else float(x) if isinstance(x, (int, float)) else x for x in v]
            for k, v in train_history.items()
        },
        'val_history': {
            k: [float(x) if torch.is_tensor(x) else float(x) if isinstance(x, (int, float)) else x for x in v]
            for k, v in val_history.items()
        },
        'timestamp': datetime.now().isoformat(),
        'text_model_info': {
            'janus': 'Janus-Pro-1B (2048-dim features)',
            'qwen': 'Qwen2.5-VL-7B-Instruct (3584-dim features)'
        }.get(text_type, 'Unknown'),
        'cleaning_notes': 'Removed boundary_loss and semantic_consistency_loss',
        'effective_components': ['BCE_loss', 'Dice_loss', 'Focal_loss', 'Tversky_loss']
    }

    results_file = results_dir / f'{experiment_name}_training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    return results_file


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified segmentation training script supporting multiple text features')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Config file path')
    parser.add_argument('--alignment', type=str, default='none',
                        choices=['none', 'simple', 'dynamic', 'generative', 'cmce', 'daca'],
                        help='Alignment method (cmce: stable 90.25%%; daca: dynamic anchor-aware, targeting 91.0%%+)')
    parser.add_argument('--text-type', type=str, default='janus',
                        choices=['janus', 'qwen'],
                        help='Text feature type')
    parser.add_argument('--unified-prompt', action='store_true',
                        help='Use features generated with unified prompts')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU ID')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')
    parser.add_argument('--loss-type', type=str, default='full',
                        choices=['full', 'bce', 'bce_dice', 'bce_dice_focal', 'seg_only'],
                        help='Loss ablation type: full=BCE+Dice+Focal+Tversky+aux, bce/bce_dice/bce_dice_focal=cumulative seg loss only, seg_only=full seg but no CMCE aux losses')
    parser.add_argument('--no-aux-loss', action='store_true',
                        help='Disable CMCE auxiliary losses (coherence/alignment/convergence/text)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['config_path'] = args.config

    # Loss ablation: disable CMCE auxiliary losses if requested
    if (getattr(args, 'no_aux_loss', False) or getattr(args, 'loss_type', 'full') == 'seg_only') and 'cmce' in config.get('model', {}):
        config['model']['cmce']['coherence_loss_weight'] = 0.0
        config['model']['cmce']['refinement_loss_weight'] = 0.0
        config['model']['cmce']['convergence_loss_weight'] = 0.0
        config['model']['cmce']['text_consistency_weight'] = 0.0

    # Set random seed for reproducibility
    # Command line argument takes precedence over config file
    seed = args.seed if args.seed is not None else config.get('experiment', {}).get('random_seed', 42)
    set_seed(seed)

    dataset_name = config['data']['dataset']
    # Pass seed to experiment name to avoid checkpoint conflicts
    _loss_suffix = '' if getattr(args, 'loss_type', 'full') == 'full' else f'_loss{args.loss_type}'
    _loss_suffix += '_noaux' if getattr(args, 'no_aux_loss', False) else ''
    experiment_name = get_experiment_name(config, args.alignment, args.text_type, args.unified_prompt, seed) + _loss_suffix
    logger = setup_logging(dataset_name, experiment_name)

    logger.info("=" * 80)
    logger.info(f"Starting training - Experiment: {experiment_name}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Alignment method: {args.alignment}")
    logger.info(f"Text feature type: {args.text_type.upper()}")
    logger.info(f"  - Model: {'Janus-Pro-1B' if args.text_type == 'janus' else 'Qwen2.5-VL-7B-Instruct'}")
    logger.info(f"  - Feature dimension: {get_text_feature_dim(args.text_type)}")
    logger.info(f"Prompt type: {'Unified prompt' if args.unified_prompt else 'Original prompt'}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Config file: {args.config}")
    logger.info("=" * 80)

    # Create directories
    dirs = {
        'checkpoints': Path('checkpoints') / dataset_name / experiment_name,
        'visualizations': Path('visualizations') / dataset_name / experiment_name,
        'results': Path('results') / dataset_name
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    if args.debug:
        config['training']['num_epochs'] = 5
        config['training']['batch_size'] = 4
        config['datasets'][dataset_name]['train_samples'] = 100
        logger.info("Debug mode: Reduced training scale")

    # Device
    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    logger.info(f"Device: {device}")

    # Create data loaders - Fix: Pass unified_prompt parameter
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        args.alignment,
        args.text_type,
        args.unified_prompt  # Pass parameter
    )

    # Create model
    model = create_cleaned_model(config, args.alignment, args.text_type).to(device)

    # Loss ablation: replace criterion if not full
    _lt = getattr(args, 'loss_type', 'full')
    if _lt not in ('full', 'seg_only'):
        from utils.unified_loss import UnifiedSegmentationLoss
        _loss_cfgs = {
            'bce':            dict(use_bce=True,  use_dice=False, use_focal=False, use_tversky=False),
            'bce_dice':       dict(use_bce=True,  use_dice=True,  use_focal=False, use_tversky=False),
            'bce_dice_focal': dict(use_bce=True,  use_dice=True,  use_focal=True,  use_tversky=False, focal_weight=0.5),
        }
        model.criterion = UnifiedSegmentationLoss(**_loss_cfgs[_lt])
        logger.info(f'Loss type overridden to: {_lt}')

    model_info = model.get_model_info()
    model_info['text_type'] = args.text_type
    model_info['text_dim'] = get_text_feature_dim(args.text_type)
    model_info['prompt_type'] = 'unified' if args.unified_prompt else 'original'

    logger.info(f"Model info: {model_info}")
    logger.info(f"Training samples: {len(train_loader)} batches")
    logger.info(f"Validation samples: {len(val_loader)} batches")
    logger.info(f"Test samples: {len(test_loader)} batches")

    # Trainer
    trainer = CleanedUnifiedTrainer(model, device, config, logger)

    logger.info("=" * 80 + "\nStarting training\n" + "=" * 80)

    # Start training
    train_history, val_history = trainer.train(
        train_loader, val_loader, config['training']['num_epochs'], str(dirs['checkpoints'])
    )

    # ============================================================================
    # 训练结束后自动评估测试集
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("POST-TRAINING TEST SET EVALUATION")
    logger.info("=" * 80)

    test_results = trainer.evaluate_test_set(test_loader)

    # 保存测试结果到checkpoint
    checkpoint_path = dirs['checkpoints'] / 'best_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint['test_results'] = test_results
        checkpoint['test_iou'] = test_results['iou']
        checkpoint['val_test_gap'] = trainer.best_val_iou - test_results['iou']
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Test results saved to checkpoint: {checkpoint_path}")

    # Save results
    results_file = save_training_results(
        dataset_name, experiment_name, dict(train_history), dict(val_history),
        model_info, config, dirs, args.text_type, args.unified_prompt
    )

    logger.info(f"Training results saved: {results_file}")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Text feature type: {args.text_type.upper()}")
    logger.info(f"Prompt type: {'Unified' if args.unified_prompt else 'Original'}")
    logger.info(f"Alignment method: {args.alignment}")
    logger.info(f"\nValidation (best):")
    logger.info(f"  IoU: {trainer.best_val_iou:.4f}")
    logger.info(f"  Threshold: {trainer.best_eval_threshold:.2f}")
    logger.info(f"\nTest set:")
    logger.info(f"  IoU: {test_results['iou']:.4f}")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall: {test_results['recall']:.4f}")
    logger.info(f"  F1: {test_results['f1']:.4f}")
    logger.info(f"\nGeneralization:")
    logger.info(f"  Val-Test Gap: {trainer.best_val_iou - test_results['iou']:+.4f} ({(trainer.best_val_iou - test_results['iou'])/trainer.best_val_iou*100:+.2f}%)")
    logger.info(f"\nModel saved at: {dirs['checkpoints']}/best_model.pth")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()