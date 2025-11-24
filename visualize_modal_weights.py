"""
模态权重可视化脚本
用于分析和可视化MulT改进模型的模态权重分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def extract_modal_weights(model_path, data_loader, device='cuda'):
    """
    从训练好的模型中提取模态权重
    
    Args:
        model_path: 模型文件路径
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        list: 每个样本的模态权重 [(weight_l, weight_a, weight_v), ...]
    """
    from src.models_improved import MULTModelImproved
    from src.utils import load_model
    
    # 加载模型（需要hyp_params，这里简化处理）
    # 实际使用时需要传入正确的hyp_params
    # model = load_model(hyp_params, name=model_name)
    
    # 这里提供一个示例函数，实际使用时需要根据具体情况调整
    weights_list = []
    
    # 注意：这个函数需要模型已经加载，并且可以访问modal_weight_net
    # 实际实现时需要在模型forward过程中记录权重
    
    return weights_list


def visualize_modal_weight_distribution(weights_list, save_path='modal_weight_distribution.png'):
    """
    可视化模态权重分布
    
    Args:
        weights_list: 权重列表，每个元素是 (weight_l, weight_a, weight_v)
        save_path: 保存路径
    """
    if len(weights_list) == 0:
        print("No weights to visualize")
        return
    
    weights_array = np.array(weights_list)  # (N, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    modal_names = ['Text', 'Audio', 'Vision']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (name, color) in enumerate(zip(modal_names, colors)):
        ax = axes[idx]
        weights = weights_array[:, idx]
        
        # 绘制直方图
        ax.hist(weights, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{name} Weight', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{name} Weight Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        ax.axvline(mean_weight, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_weight:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Modal weight distribution saved to {save_path}")
    plt.close()


def visualize_modal_weight_heatmap(weights_list, save_path='modal_weight_heatmap.png'):
    """
    可视化模态权重的热力图（展示不同样本的权重分布）
    
    Args:
        weights_list: 权重列表
        save_path: 保存路径
    """
    if len(weights_list) == 0:
        print("No weights to visualize")
        return
    
    weights_array = np.array(weights_list)  # (N, 3)
    
    # 对样本进行排序（按文本权重）
    sorted_indices = np.argsort(weights_array[:, 0])
    sorted_weights = weights_array[sorted_indices]
    
    # 只显示前100个样本（如果太多）
    if len(sorted_weights) > 100:
        sorted_weights = sorted_weights[:100]
    
    plt.figure(figsize=(10, max(6, len(sorted_weights) * 0.1)))
    sns.heatmap(sorted_weights.T, 
                xticklabels=False,
                yticklabels=['Text', 'Audio', 'Vision'],
                cmap='YlOrRd',
                cbar_kws={'label': 'Weight'})
    plt.title('Modal Weight Heatmap (Sorted by Text Weight)', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Modality', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Modal weight heatmap saved to {save_path}")
    plt.close()


def visualize_modal_weight_scatter(weights_list, save_path='modal_weight_scatter.png'):
    """
    可视化模态权重的散点图（展示模态之间的相关性）
    
    Args:
        weights_list: 权重列表
        save_path: 保存路径
    """
    if len(weights_list) == 0:
        print("No weights to visualize")
        return
    
    weights_array = np.array(weights_list)  # (N, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pairs = [(0, 1, 'Text', 'Audio'), (0, 2, 'Text', 'Vision'), (1, 2, 'Audio', 'Vision')]
    
    for idx, (i, j, name_i, name_j) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(weights_array[:, i], weights_array[:, j], alpha=0.5, s=20)
        ax.set_xlabel(f'{name_i} Weight', fontsize=12)
        ax.set_ylabel(f'{name_j} Weight', fontsize=12)
        ax.set_title(f'{name_i} vs {name_j}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加对角线
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Equal weights')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Modal weight scatter plot saved to {save_path}")
    plt.close()


def create_model_architecture_diagram(save_path='model_architecture.png'):
    """
    创建模型架构对比图（用于实验报告）
    
    Args:
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['Baseline', 'MulT Original', 'MulT Improved']
    descriptions = [
        'Simple MLP Fusion\n• Linear projection\n• Average pooling\n• Concatenation\n• MLP',
        'Transformer-based\n• Cross-modal attention\n• Self-attention\n• Concatenation\n• MLP',
        'MulT + Weight Fusion\n• Cross-modal attention\n• Self-attention\n• Modal weight learning\n• Weighted fusion'
    ]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    for idx, (model, desc, color) in enumerate(zip(models, descriptions, colors)):
        ax = axes[idx]
        ax.text(0.5, 0.5, f'{model}\n\n{desc}', 
               ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
               transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'{model} Architecture', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model architecture diagram saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize modal weights')
    parser.add_argument('--weights_file', type=str, default=None,
                       help='File containing modal weights (JSON format)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建模型架构图
    create_model_architecture_diagram(
        save_path=os.path.join(args.output_dir, 'model_architecture.png')
    )
    
    # 如果有权重文件，进行权重可视化
    if args.weights_file and os.path.exists(args.weights_file):
        with open(args.weights_file, 'r') as f:
            weights_list = json.load(f)
        
        visualize_modal_weight_distribution(
            weights_list,
            save_path=os.path.join(args.output_dir, 'modal_weight_distribution.png')
        )
        
        visualize_modal_weight_heatmap(
            weights_list,
            save_path=os.path.join(args.output_dir, 'modal_weight_heatmap.png')
        )
        
        visualize_modal_weight_scatter(
            weights_list,
            save_path=os.path.join(args.output_dir, 'modal_weight_scatter.png')
        )
    else:
        print("No weights file provided, only creating architecture diagram")

