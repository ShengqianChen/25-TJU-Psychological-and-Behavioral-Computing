"""训练可视化脚本"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_log(log_path):
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_training_curves(log_paths, labels, save_path='training_curves.png', 
                         metrics=['train_loss', 'val_loss', 'test_loss']):
    plt.figure(figsize=(15, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (log_path, label) in enumerate(zip(log_paths, labels)):
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found, skipping...")
            continue
        
        history = load_training_log(log_path)
        epochs = history.get('epochs', [])
        
        if len(epochs) == 0:
            continue
        
        color = colors[idx % len(colors)]
        
        for metric in metrics:
            if metric in history and len(history[metric]) > 0:
                values = history[metric]
                if len(values) == len(epochs):
                    plt.plot(epochs, values, label=f'{label} - {metric}', 
                            color=color, linestyle='--' if 'train' in metric else '-',
                            alpha=0.7 if 'train' in metric else 1.0)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_metric_comparison(log_paths, labels, metric='val_loss', 
                           save_path='metric_comparison.png'):
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (log_path, label) in enumerate(zip(log_paths, labels)):
        if not os.path.exists(log_path):
            continue
        
        history = load_training_log(log_path)
        epochs = history.get('epochs', [])
        
        if metric in history and len(history[metric]) > 0:
            values = history[metric]
            if len(values) == len(epochs):
                plt.plot(epochs, values, label=label, 
                        color=colors[idx % len(colors)], linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metric comparison saved to {save_path}")
    plt.close()


def plot_final_metrics_comparison(log_paths, labels, metrics=['val_loss', 'test_loss', 'val_mae', 'test_mae'],
                                  save_path='final_metrics_comparison.png'):
    # 收集数据
    data = {label: [] for label in labels}
    metric_labels = []
    
    for log_path, label in zip(log_paths, labels):
        if not os.path.exists(log_path):
            data[label] = [None] * len(metrics)
            continue
        
        history = load_training_log(log_path)
        values = []
        for metric in metrics:
            if metric in history and len(history[metric]) > 0:
                values.append(history[metric][-1])  # 最后一个epoch的值
            else:
                values.append(None)
        data[label] = values
    
    # 过滤掉全为None的指标
    valid_metrics = []
    valid_data = {label: [] for label in labels}
    
    for idx, metric in enumerate(metrics):
        if any(data[label][idx] is not None for label in labels):
            valid_metrics.append(metric)
            for label in labels:
                valid_data[label].append(data[label][idx])
    
    if len(valid_metrics) == 0:
        print("No valid metrics to plot")
        return
    
    # 绘制柱状图
    x = np.arange(len(valid_metrics))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, label in enumerate(labels):
        values = [v if v is not None else 0 for v in valid_data[label]]
        ax.bar(x + idx * width, values, width, label=label, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Final Metrics Comparison', fontsize=14)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in valid_metrics], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Final metrics comparison saved to {save_path}")
    plt.close()


def plot_model_comparison_table(log_paths, labels, save_path='model_comparison_table.png'):
    import pandas as pd
    
    # 收集所有指标
    all_metrics = ['train_loss', 'val_loss', 'test_loss', 'val_mae', 'test_mae', 
                   'val_corr', 'test_corr', 'val_acc', 'test_acc']
    
    data = []
    for log_path, label in zip(log_paths, labels):
        if not os.path.exists(log_path):
            continue
        
        history = load_training_log(log_path)
        row = {'Model': label}
        
        for metric in all_metrics:
            if metric in history and len(history[metric]) > 0:
                row[metric.replace('_', ' ').title()] = f"{history[metric][-1]:.4f}"
            else:
                row[metric.replace('_', ' ').title()] = 'N/A'
        
        data.append(row)
    
    if len(data) == 0:
        print("No data to plot")
        return
    
    df = pd.DataFrame(data)
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(14, len(data) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Comparison Table', fontsize=14, pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison table saved to {save_path}")
    plt.close()


def visualize_all(log_dir='training_logs', output_dir='visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有日志文件
    log_files = list(Path(log_dir).glob('*.json'))
    
    if len(log_files) == 0:
        print(f"No training logs found in {log_dir}")
        return
    
    # 提取模型名称
    log_paths = [str(f) for f in log_files]
    labels = [f.stem for f in log_files]
    
    print(f"Found {len(log_paths)} training logs")
    
    # 绘制各种图表
    plot_training_curves(log_paths, labels, 
                        save_path=os.path.join(output_dir, 'training_curves.png'))
    
    plot_metric_comparison(log_paths, labels, metric='val_loss',
                          save_path=os.path.join(output_dir, 'val_loss_comparison.png'))
    
    plot_final_metrics_comparison(log_paths, labels,
                                  save_path=os.path.join(output_dir, 'final_metrics_comparison.png'))
    
    try:
        plot_model_comparison_table(log_paths, labels,
                                   save_path=os.path.join(output_dir, 'comparison_table.png'))
    except ImportError:
        print("pandas not available, skipping table visualization")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training logs')
    parser.add_argument('--log_dir', type=str, default='training_logs',
                       help='Directory containing training logs')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_all(args.log_dir, args.output_dir)

