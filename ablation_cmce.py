# ablation_cmce.py
"""
CMCE 消融实验脚本

支持的消融类型:
1. num_anchors: 语义锚点数量 (1, 2, 3, 4, 5)
2. max_iterations: 最大迭代次数 (1, 2, 3)
3. deformable: 可变形卷积开关 (True, False)
4. dropout: dropout率 (0.05, 0.08, 0.11, 0.14, 0.17)
5. residual_scale: 残差缩放 (0.03, 0.05, 0.07, 0.09, 0.11)
6. text_change: 文本变化约束 (0.06, 0.09, 0.12, 0.15, 0.18)
7. num_scales: 多尺度数量 (1, 2, 3, 4)
8. loss_weights: 损失权重组合

用法:
    # 单个实验
    python ablation_cmce.py --ablation num_anchors --value 1 --gpu 0 --seed 42
    python ablation_cmce.py --ablation deformable --value false --gpu 1 --seed 42
    python ablation_cmce.py --ablation full --gpu 0 --seed 42  # 运行完整CMCE作为baseline

    # 批量运行（在单GPU上串行运行一组实验）
    python ablation_cmce.py --batch core --gpu 0 --seed 42
    python ablation_cmce.py --batch anchors --gpu 1 --seed 42

    # 查看可用选项
    python ablation_cmce.py --list
"""

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
from copy import deepcopy

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from models.cleaned_unified_segmentation_model import CleanedUnifiedSegmentationModel, CleanedUnifiedTrainer
from utils.unified_data_manager import UnifiedDataManager
from utils.transforms import BuildingExtractionTransforms


# ============================================================================
# 预定义的消融实验组（方便在不同服务器上分配）
# ============================================================================

ABLATION_GROUPS = {
    # 核心组件消融（最重要，优先运行）
    'core': [
        ('full', ''),                    # baseline
        ('no_coevolution', ''),          # 无协同进化
        ('no_text_refine', ''),          # 无文本精炼
        ('no_visual_refine', ''),        # 无视觉精炼
        ('deformable', 'false'),         # 无可变形卷积
    ],

    # 锚点数量消融
    'anchors': [
        ('num_anchors', '1'),
        ('num_anchors', '2'),
        ('num_anchors', '4'),
        ('num_anchors', '5'),
    ],

    # 迭代次数消融
    'iterations': [
        ('max_iterations', '1'),
        ('max_iterations', '3'),
    ],

    # 多尺度消融
    'scales': [
        ('num_scales', '1'),
        ('num_scales', '2'),
        ('num_scales', '3'),
    ],

    # 超参数敏感性分析
    'hyperparams': [
        ('dropout', '0.05'),
        ('dropout', '0.17'),
        ('residual_scale', '0.03'),
        ('residual_scale', '0.11'),
        ('text_change', '0.06'),
        ('text_change', '0.18'),
    ],

    # 论文必需的消融（精简版）
    'paper_essential': [
        ('full', ''),                    # baseline (必须)
        ('no_coevolution', ''),          # 核心消融
        ('no_text_refine', ''),          # 核心消融
        ('no_visual_refine', ''),        # 核心消融
    ],

    # 服务器1推荐（约5个实验）
    'server1': [
        ('full', ''),
        ('no_coevolution', ''),
        ('no_text_refine', ''),
        ('num_anchors', '1'),
        ('max_iterations', '1'),
    ],

    # 服务器2推荐（约5个实验）
    'server2': [
        ('no_visual_refine', ''),
        ('deformable', 'false'),
        ('num_anchors', '2'),
        ('num_scales', '1'),
        ('num_scales', '2'),
    ],

    # 服务器3推荐（约4个实验）
    'server3': [
        ('max_iterations', '3'),
        ('num_anchors', '4'),
        ('dropout', '0.05'),
        ('residual_scale', '0.03'),
    ],
}


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(experiment_name, log_dir):
    """Setup logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger(experiment_name)
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


def get_ablation_config(base_config, ablation_type, ablation_value):
    """
    根据消融类型修改配置

    Returns:
        config: 修改后的配置
        ablation_desc: 消融描述字符串
    """
    config = deepcopy(base_config)
    cmce_config = config['model']['cmce']

    if ablation_type == 'full':
        return config, 'full_cmce'

    elif ablation_type == 'num_anchors':
        num_anchors = int(ablation_value)
        cmce_config['num_anchors'] = num_anchors
        return config, f'anchors_{num_anchors}'

    elif ablation_type == 'max_iterations':
        max_iter = int(ablation_value)
        cmce_config['max_iterations'] = max_iter
        return config, f'iter_{max_iter}'

    elif ablation_type == 'deformable':
        use_deform = ablation_value.lower() in ('true', '1', 'yes')
        cmce_config['use_deformable'] = use_deform
        return config, f'deform_{"on" if use_deform else "off"}'

    elif ablation_type == 'dropout':
        dropout = float(ablation_value)
        cmce_config['dropout'] = dropout
        return config, f'dropout_{dropout:.2f}'

    elif ablation_type == 'residual_scale':
        res_scale = float(ablation_value)
        cmce_config['residual_scale'] = res_scale
        return config, f'resscale_{res_scale:.2f}'

    elif ablation_type == 'text_change':
        text_change = float(ablation_value)
        cmce_config['max_text_change_ratio'] = text_change
        return config, f'textchange_{text_change:.2f}'

    elif ablation_type == 'num_scales':
        num_scales = int(ablation_value)
        if num_scales == 1:
            cmce_config['scale_channels'] = [512]
            cmce_config['feature_keys'] = ['bottleneck']
        elif num_scales == 2:
            cmce_config['scale_channels'] = [512, 512]
            cmce_config['feature_keys'] = ['encoder_4', 'bottleneck']
        elif num_scales == 3:
            cmce_config['scale_channels'] = [256, 512, 512]
            cmce_config['feature_keys'] = ['encoder_3', 'encoder_4', 'bottleneck']
        else:
            cmce_config['scale_channels'] = [256, 512, 512, 512]
            cmce_config['feature_keys'] = ['encoder_2', 'encoder_3', 'encoder_4', 'bottleneck']
        return config, f'scales_{num_scales}'

    elif ablation_type == 'loss_weights':
        weights = [float(w) for w in ablation_value.split('_')]
        if len(weights) == 4:
            cmce_config['coherence_loss_weight'] = weights[0]
            cmce_config['refinement_loss_weight'] = weights[1]
            cmce_config['convergence_loss_weight'] = weights[2]
            cmce_config['text_consistency_weight'] = weights[3]
        return config, f'loss_{ablation_value}'

    elif ablation_type == 'no_text_refine':
        cmce_config['max_text_change_ratio'] = 0.001
        cmce_config['text_consistency_weight'] = 0.0
        return config, 'no_text_refine'

    elif ablation_type == 'no_visual_refine':
        cmce_config['residual_scale'] = 0.001
        return config, 'no_visual_refine'

    elif ablation_type == 'no_coevolution':
        cmce_config['max_iterations'] = 1
        cmce_config['max_text_change_ratio'] = 0.001
        return config, 'no_coevolution'

    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")


def get_text_feature_dim(text_type):
    """Get feature dimension for different text models"""
    text_dims = {
        'janus': 2048,
        'qwen': 3584,
    }
    return text_dims.get(text_type.lower(), 2048)


def create_model(config, text_type='janus'):
    """Create CMCE model with ablation config"""
    model_config = config['model']
    backbone_config = model_config['backbone']
    cmce_config = model_config.get('cmce', {})

    text_dim = get_text_feature_dim(text_type)
    cmce_config = dict(cmce_config)
    cmce_config['text_dim'] = text_dim

    model = CleanedUnifiedSegmentationModel(
        in_channels=backbone_config['in_channels'],
        num_classes=model_config['segmentation_head']['num_classes'],
        base_channels=backbone_config['base_channels'],
        text_dim=text_dim,
        alignment_type='cmce',
        alignment_config=cmce_config,
        alignment_insertion_layer=cmce_config.get('feature_keys', ['encoder_2', 'encoder_3', 'encoder_4', 'bottleneck']),
    )
    return model


def create_data_loaders(config, text_type='janus'):
    """Create data loaders"""
    data_config = config['data']
    training_config = config['training']
    data_manager = UnifiedDataManager(config)
    dataset_name = data_config['dataset']
    image_size = data_config['image_size']

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

    feature_subdir = 'text_features'
    text_subdir = 'text_descriptions'

    train_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        sample_count=config['datasets'][dataset_name].get('train_samples', -1),
        alignment_type='cmce',
        text_type=text_type,
        transform=train_transform,
        text_feature_subdir=feature_subdir,
        text_subdir=text_subdir
    )

    val_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=training_config.get('val_batch_size', training_config['batch_size']),
        shuffle=False,
        num_workers=data_config['num_workers'],
        alignment_type='cmce',
        text_type=text_type,
        transform=val_transform,
        text_feature_subdir=feature_subdir,
        text_subdir=text_subdir
    )

    test_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=training_config.get('val_batch_size', training_config['batch_size']),
        shuffle=False,
        num_workers=data_config['num_workers'],
        alignment_type='cmce',
        text_type=text_type,
        transform=val_transform,
        text_feature_subdir=feature_subdir,
        text_subdir=text_subdir
    )

    return train_loader, val_loader, test_loader


def run_single_ablation(args, ablation_type, ablation_value):
    """Run a single ablation experiment"""
    # Load base config
    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # Apply ablation
    config, ablation_desc = get_ablation_config(base_config, ablation_type, ablation_value)

    # Set seed
    seed = args.seed
    set_seed(seed)

    # Experiment name
    experiment_name = f"ablation_{ablation_desc}_seed{seed}"
    dataset_name = config['data']['dataset']

    # Setup directories
    dirs = {
        'checkpoints': Path('checkpoints') / dataset_name / 'ablation' / experiment_name,
        'logs': Path('logs') / dataset_name / 'ablation' / experiment_name,
        'results': Path('results') / dataset_name / 'ablation'
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    results_file = dirs['results'] / f'{experiment_name}_results.json'
    if results_file.exists() and not args.force:
        print(f"[SKIP] {ablation_desc} already completed. Use --force to re-run.")
        with open(results_file, 'r') as f:
            return json.load(f)

    # Setup logging
    logger = setup_logging(experiment_name, dirs['logs'])

    logger.info("=" * 70)
    logger.info(f"CMCE ABLATION: {ablation_desc}")
    logger.info("=" * 70)
    logger.info(f"Ablation: {ablation_type} = {ablation_value if ablation_value else 'N/A'}")
    logger.info(f"Seed: {seed} | GPU: {args.gpu}")

    # Log CMCE config
    cmce_config = config['model']['cmce']
    logger.info(f"CMCE Config: {json.dumps(cmce_config, indent=2)}")

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, args.text_type)
    logger.info(f"Data: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

    # Create model
    model = create_model(config, args.text_type).to(device)
    model_info = model.get_model_info()
    logger.info(f"Model params: {model_info['params_total']:,} total, {model_info['params_alignment_branch']:,} alignment")

    # Trainer
    trainer = CleanedUnifiedTrainer(model, device, config, logger)

    # Train
    num_epochs = config['training']['num_epochs']
    if args.debug:
        num_epochs = 3
        logger.info(f"DEBUG MODE: {num_epochs} epochs only")

    train_history, val_history = trainer.train(
        train_loader, val_loader, num_epochs, str(dirs['checkpoints'])
    )

    # Test
    test_results = trainer.evaluate_test_set(test_loader)

    # Save results
    results = {
        'experiment_name': experiment_name,
        'ablation_type': ablation_type,
        'ablation_value': ablation_value,
        'ablation_desc': ablation_desc,
        'seed': seed,
        'gpu': args.gpu,
        'cmce_config': cmce_config,
        'model_info': model_info,
        'best_val_iou': float(trainer.best_val_iou),
        'best_epoch': int(trainer.best_epoch),
        'test_iou': float(test_results['iou']),
        'test_precision': float(test_results['precision']),
        'test_recall': float(test_results['recall']),
        'test_f1': float(test_results['f1']),
        'val_test_gap': float(trainer.best_val_iou - test_results['iou']),
        'timestamp': datetime.now().isoformat()
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info(f"RESULT: {ablation_desc}")
    logger.info(f"  Val IoU: {trainer.best_val_iou:.4f} | Test IoU: {test_results['iou']:.4f}")
    logger.info(f"  Gap: {trainer.best_val_iou - test_results['iou']:+.4f}")
    logger.info("=" * 70)

    return results


def run_batch_ablation(args):
    """Run a batch of ablation experiments on single GPU"""
    group_name = args.batch
    if group_name not in ABLATION_GROUPS:
        print(f"Error: Unknown group '{group_name}'")
        print(f"Available groups: {list(ABLATION_GROUPS.keys())}")
        return

    experiments = ABLATION_GROUPS[group_name]
    print(f"\n{'='*60}")
    print(f"BATCH ABLATION: {group_name}")
    print(f"{'='*60}")
    print(f"Experiments: {len(experiments)}")
    print(f"GPU: {args.gpu} | Seed: {args.seed}")
    print(f"{'='*60}\n")

    all_results = []
    start_time = datetime.now()

    for i, (abl_type, abl_val) in enumerate(experiments, 1):
        exp_name = f"{abl_type}_{abl_val}" if abl_val else abl_type
        print(f"\n[{i}/{len(experiments)}] Running: {exp_name}")
        print("-" * 40)

        try:
            result = run_single_ablation(args, abl_type, abl_val)
            all_results.append(result)
            print(f"✓ Completed: Test IoU = {result['test_iou']:.4f}")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            all_results.append({
                'ablation_type': abl_type,
                'ablation_value': abl_val,
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {group_name}")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f} minutes")
    print(f"\nResults:")

    for r in sorted(all_results, key=lambda x: x.get('test_iou', 0), reverse=True):
        if 'test_iou' in r:
            print(f"  {r['ablation_desc']}: IoU={r['test_iou']:.4f}")
        else:
            print(f"  {r['ablation_type']}: FAILED")

    # Save batch summary
    summary_file = Path('results') / 'ablation' / f'batch_{group_name}_seed{args.seed}.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            'group': group_name,
            'seed': args.seed,
            'gpu': args.gpu,
            'elapsed_minutes': elapsed,
            'results': all_results
        }, f, indent=2)

    print(f"\nSummary saved: {summary_file}")


def print_help():
    """Print detailed help information"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CMCE 消融实验脚本使用指南                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

【单个实验运行】
    python ablation_cmce.py --ablation <type> --value <val> --gpu <id> --seed 42

    示例:
    python ablation_cmce.py --ablation full --gpu 0 --seed 42           # baseline
    python ablation_cmce.py --ablation no_coevolution --gpu 0 --seed 42 # 无协同进化
    python ablation_cmce.py --ablation num_anchors --value 1 --gpu 0    # 1个锚点
    python ablation_cmce.py --ablation num_scales --value 2 --gpu 1     # 2个尺度

【批量实验运行】（推荐用于多服务器）
    python ablation_cmce.py --batch <group> --gpu <id> --seed 42

    示例:
    python ablation_cmce.py --batch server1 --gpu 0 --seed 42  # 服务器1运行
    python ablation_cmce.py --batch server2 --gpu 0 --seed 42  # 服务器2运行
    python ablation_cmce.py --batch server3 --gpu 1 --seed 42  # 服务器3运行

【可用的实验组】
""")
    for group, exps in ABLATION_GROUPS.items():
        exp_names = [f"{t}_{v}" if v else t for t, v in exps]
        print(f"  {group:18} ({len(exps)}个): {', '.join(exp_names[:3])}{'...' if len(exps) > 3 else ''}")

    print("""
【可用的消融类型】
    full            - 完整CMCE (baseline)
    no_coevolution  - 禁用协同进化
    no_text_refine  - 禁用文本精炼
    no_visual_refine- 禁用视觉精炼
    num_anchors     - 锚点数量 (1, 2, 3*, 4, 5)
    max_iterations  - 迭代次数 (1, 2*, 3)
    deformable      - 可变形卷积 (true*, false)
    num_scales      - 尺度数量 (1, 2, 3, 4*)
    dropout         - Dropout率 (0.05, 0.08, 0.11*, 0.14, 0.17)
    residual_scale  - 残差缩放 (0.03, 0.05, 0.07*, 0.09, 0.11)
    text_change     - 文本变化 (0.06, 0.09, 0.12*, 0.15, 0.18)

    (* 表示默认值)

【多服务器分配建议】
    服务器A: python ablation_cmce.py --batch server1 --gpu 0 --seed 42
    服务器B: python ablation_cmce.py --batch server2 --gpu 0 --seed 42
    服务器C: python ablation_cmce.py --batch server3 --gpu 0 --seed 42

【其他选项】
    --force     强制重新运行已完成的实验
    --debug     调试模式 (仅3轮)
    --text-type 文本特征类型 (janus/qwen)
""")


def main():
    parser = argparse.ArgumentParser(
        description='CMCE Ablation Experiment Script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['full', 'num_anchors', 'max_iterations', 'deformable',
                                 'dropout', 'residual_scale', 'text_change', 'num_scales',
                                 'loss_weights', 'no_text_refine', 'no_visual_refine', 'no_coevolution'])
    parser.add_argument('--value', type=str, default='')
    parser.add_argument('--batch', type=str, default=None,
                        choices=list(ABLATION_GROUPS.keys()),
                        help='Run a batch of experiments')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--text-type', type=str, default='janus', choices=['janus', 'qwen'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force', action='store_true', help='Force re-run completed experiments')
    parser.add_argument('--list', action='store_true', help='Show help and options')

    args = parser.parse_args()

    if args.list:
        print_help()
        return

    if args.batch:
        # 批量运行模式
        run_batch_ablation(args)
    elif args.ablation:
        # 单个实验模式
        result = run_single_ablation(args, args.ablation, args.value)
        print(f"\nResult: Test IoU = {result['test_iou']:.4f}")
    else:
        print("Error: Please specify --ablation or --batch")
        print("Use --list for help")


if __name__ == "__main__":
    main()
