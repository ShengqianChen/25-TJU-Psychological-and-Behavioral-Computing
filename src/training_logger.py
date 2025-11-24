"""
Training Logger Module
用于记录训练过程中的各种指标，便于后续可视化和分析
"""

import json
import os
from collections import defaultdict
import numpy as np


class TrainingLogger:
    """
    训练日志记录器
    记录每个epoch的训练loss、验证loss、测试loss等指标
    """
    
    def __init__(self, log_dir='training_logs', experiment_name='experiment'):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f'{experiment_name}.json')
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化记录字典
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_mae': [],      # 仅用于回归任务
            'val_mae': [],
            'test_mae': [],
            'train_corr': [],     # 仅用于回归任务
            'val_corr': [],
            'test_corr': [],
            'train_acc': [],      # 仅用于分类任务
            'val_acc': [],
            'test_acc': [],
            'learning_rate': [],
            'epochs': []
        }
        
        # 加载已有日志（如果存在）
        if os.path.exists(self.log_path):
            self.load()
    
    def log_epoch(self, epoch, train_loss, val_loss, test_loss, 
                  train_metrics=None, val_metrics=None, test_metrics=None, 
                  learning_rate=None):
        """
        记录一个epoch的训练结果
        
        Args:
            epoch: epoch编号
            train_loss: 训练loss
            val_loss: 验证loss
            test_loss: 测试loss
            train_metrics: 训练集其他指标（dict，如 {'mae': 0.8, 'corr': 0.7}）
            val_metrics: 验证集其他指标
            test_metrics: 测试集其他指标
            learning_rate: 当前学习率
        """
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['test_loss'].append(float(test_loss))
        
        if learning_rate is not None:
            self.history['learning_rate'].append(float(learning_rate))
        
        # 记录其他指标
        for metrics, prefix in [(train_metrics, 'train'), 
                                (val_metrics, 'val'), 
                                (test_metrics, 'test')]:
            if metrics:
                for key, value in metrics.items():
                    history_key = f'{prefix}_{key}'
                    if history_key in self.history:
                        self.history[history_key].append(float(value))
        
        # 自动保存
        self.save()
    
    def save(self):
        """保存日志到JSON文件"""
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self):
        """从JSON文件加载日志"""
        with open(self.log_path, 'r') as f:
            self.history = json.load(f)
    
    def get_best_epoch(self, metric='val_loss', mode='min'):
        """
        获取最佳epoch
        
        Args:
            metric: 指标名称（如 'val_loss', 'val_acc'）
            mode: 'min' 或 'max'
        
        Returns:
            (best_epoch, best_value)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return None, None
        
        values = self.history[metric]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        best_epoch = self.history['epochs'][best_idx]
        best_value = values[best_idx]
        
        return best_epoch, best_value
    
    def get_final_metrics(self):
        """获取最后一个epoch的所有指标"""
        if len(self.history['epochs']) == 0:
            return None
        
        final_metrics = {}
        for key, values in self.history.items():
            if len(values) > 0:
                final_metrics[key] = values[-1]
        
        return final_metrics

