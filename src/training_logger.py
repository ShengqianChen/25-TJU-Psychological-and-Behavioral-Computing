"""训练日志记录模块"""

import json
import os
from collections import defaultdict
import numpy as np


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir='training_logs', experiment_name='experiment'):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f'{experiment_name}.json')
        os.makedirs(log_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_mae': [],
            'val_mae': [],
            'test_mae': [],
            'train_corr': [],
            'val_corr': [],
            'test_corr': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'learning_rate': [],
            'epochs': []
        }
        
        if os.path.exists(self.log_path):
            self.load()
    
    def log_epoch(self, epoch, train_loss, val_loss, test_loss, 
                  train_metrics=None, val_metrics=None, test_metrics=None, 
                  learning_rate=None):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['test_loss'].append(float(test_loss))
        
        if learning_rate is not None:
            self.history['learning_rate'].append(float(learning_rate))
        
        for metrics, prefix in [(train_metrics, 'train'), 
                                (val_metrics, 'val'), 
                                (test_metrics, 'test')]:
            if metrics:
                for key, value in metrics.items():
                    history_key = f'{prefix}_{key}'
                    if history_key in self.history:
                        self.history[history_key].append(float(value))
        
        self.save()
    
    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self):
        with open(self.log_path, 'r') as f:
            self.history = json.load(f)
    
    def get_best_epoch(self, metric='val_loss', mode='min'):
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
        if len(self.history['epochs']) == 0:
            return None
        
        final_metrics = {}
        for key, values in self.history.items():
            if len(values) > 0:
                final_metrics[key] = values[-1]
        
        return final_metrics

