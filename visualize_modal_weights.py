"""模态权重可视化脚本"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def extract_modal_weights_from_model(model, data_loader, device='cuda', max_samples=1000):
    model.eval()
    weights_list = []
    
    # 检查模型是否有模态权重网络
    if not hasattr(model, 'modal_weight_net'):
        print("Warning: Model does not have modal_weight_net. This function only works for MULTModelImproved.")
        return weights_list
    
    with torch.no_grad():
        sample_count = 0
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(data_loader):
            if sample_count >= max_samples:
                break
                
            sample_ind, text, audio, vision = batch_X
            
            # 移动到设备
            if device == 'cuda' and torch.cuda.is_available():
                text = text.cuda()
                audio = audio.cuda()
                vision = vision.cuda()
            
            # 获取模型中间特征（需要修改模型forward来返回权重）
            # 由于不能直接修改模型，我们需要hook来捕获权重
            modal_weights_batch = []
            
            def hook_fn(module, input, output):
                # 这个hook会在modal_weight_net被调用时捕获输出
                modal_weights_batch.append(output.cpu().numpy())
            
            # 注册hook到modal_weight_net
            hook = model.modal_weight_net.register_forward_hook(hook_fn)
            
            try:
                # 前向传播
                _, _ = model(text, audio, vision)
                
                # 由于modal_weight_net被调用了3次（对每个模态），我们需要重新设计
                # 更好的方法是修改模型来返回权重，或者直接在前向传播中提取
            finally:
                hook.remove()
            
            sample_count += text.size(0)
    
    return weights_list


def extract_modal_weights(model_path, dataset='mosi', data_path='data', split='test', 
                         device='cuda', max_samples=500):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.dataset import Multimodal_Datasets
    from torch.utils.data import DataLoader
    
    # 加载数据
    print(f"加载 {split} 数据...")
    data = Multimodal_Datasets(data_path, dataset, split, if_align=False)
    data_loader = DataLoader(data, batch_size=16, shuffle=False)
    
    # 加载模型
    print(f"从 {model_path} 加载模型...")
    if device == 'cuda' and torch.cuda.is_available():
        model = torch.load(model_path, map_location='cuda', weights_only=False)
    else:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        device = 'cpu'
    
    model.eval()  # 设为评估模式
    
    # 检查一下是不是改进模型（只有改进模型才有modal_weight_net）
    if not hasattr(model, 'modal_weight_net'):
        print("错误：这个模型没有modal_weight_net，只有改进模型才有这个功能")
        return []
    
    weights_list = []
    print("开始提取模态权重...")
    
    with torch.no_grad():
        sample_count = 0
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(data_loader):
            if sample_count >= max_samples:
                break
                
            sample_ind, text, audio, vision = batch_X
            
            if device == 'cuda' and torch.cuda.is_available():
                text = text.cuda()
                audio = audio.cuda()
                vision = vision.cuda()
            
            # 用return_weights=True来获取权重
            try:
                output, hidden, modal_weights = model(text, audio, vision, return_weights=True)
            except TypeError:
                # 如果模型不支持return_weights参数（可能是旧版本），就用hook方法
                # hook就是在前向传播的时候拦截一下，把权重拿出来
                captured_weights = []
                
                def weight_hook(module, input, output):
                    # 把权重存下来
                    if len(captured_weights) < 3:
                        captured_weights.append(output.detach())
                
                hook = model.modal_weight_net.register_forward_hook(weight_hook)
                
                try:
                    output, hidden = model(text, audio, vision)
                    # 手动算一下权重（如果hook没抓到softmax后的结果）
                    if len(captured_weights) == 3:
                        weight_l, weight_a, weight_v = captured_weights
                        modal_weights = torch.softmax(torch.cat([weight_l, weight_a, weight_v], dim=1), dim=1)
                    else:
                        # hook方法失败了，跳过这个batch
                        hook.remove()
                        continue
                finally:
                    hook.remove()
            
            if modal_weights is not None:
                # modal_weights是(batch_size, 3)的形状
                # 转成numpy存到列表里
                weights_np = modal_weights.cpu().numpy()
                for i in range(weights_np.shape[0]):
                    weights_list.append(weights_np[i].tolist())  # [文本权重, 音频权重, 视觉权重]
            
            batch_size = text.size(0)
            sample_count += batch_size
            
            if (i_batch + 1) % 10 == 0:
                print(f"  已经处理了 {sample_count} 个样本...")
    
    print(f"总共提取了 {sample_count} 个样本的权重")
    return weights_list


def visualize_modal_weight_distribution(weights_list, save_path='modal_weight_distribution.png'):
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
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file (.pt)')
    parser.add_argument('--dataset', type=str, default='mosi',
                       help='Dataset name (mosi, mosei_senti, iemocap)')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test',
                       help='Data split (train, valid, test)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum number of samples to extract weights from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建模型架构图
    create_model_architecture_diagram(
        save_path=os.path.join(args.output_dir, 'model_architecture.png')
    )
    
    weights_list = []
    
    # 如果提供了模型路径，从模型中提取权重
    if args.model_path and os.path.exists(args.model_path):
        print(f"Extracting modal weights from model: {args.model_path}")
        weights_list = extract_modal_weights(
            model_path=args.model_path,
            dataset=args.dataset,
            data_path=args.data_path,
            split=args.split,
            device=args.device,
            max_samples=args.max_samples
        )
        
        # 保存权重到JSON文件
        if len(weights_list) > 0:
            weights_file = os.path.join(args.output_dir, 'modal_weights.json')
            with open(weights_file, 'w') as f:
                json.dump(weights_list, f, indent=2)
            print(f"Modal weights saved to {weights_file}")
    
    # 如果有权重文件，从文件加载权重
    elif args.weights_file and os.path.exists(args.weights_file):
        print(f"Loading modal weights from file: {args.weights_file}")
        with open(args.weights_file, 'r') as f:
            weights_list = json.load(f)
    
    # 如果有权重数据，进行可视化
    if len(weights_list) > 0:
        print(f"Visualizing {len(weights_list)} modal weight samples...")
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
        print("Modal weight visualizations completed!")
    else:
        print("No weights extracted. Only architecture diagram created.")
        print("To extract weights, use: --model_path path/to/model.pt --dataset mosi")

