# MulT模型改进说明

## 改进内容：模态权重融合（Modal Weight Fusion）

### 改进动机

原始的MulT模型在融合三个模态（文本、音频、视觉）时，使用的是简单的特征拼接（concatenation）：
```python
last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
```

这种方式假设所有模态的重要性是相等的，但实际上：
- 不同样本中，不同模态的重要性可能不同
- 例如：有些样本中文本信息更重要，有些样本中音频信息更重要
- 简单拼接无法自适应地调整模态权重

### 改进方案

我们添加了一个**模态权重学习模块**，能够：
1. 根据每个样本的特征，自动学习每个模态的重要性权重
2. 使用学习到的权重对模态特征进行加权融合
3. 通过softmax确保权重归一化（权重和为1）

### 实现细节

#### 1. 模态权重网络（Modal Weight Network）

在模型初始化时（`__init__`方法），添加了一个小的MLP网络：

```python
self.modal_weight_net = nn.Sequential(
    nn.Linear(modal_dim, modal_dim // 2),  # 60 -> 30
    nn.ReLU(),
    nn.Dropout(self.embed_dropout),
    nn.Linear(modal_dim // 2, 1)            # 30 -> 1
)
```

这个网络接收每个模态的特征（维度：2*d = 60），输出一个标量权重。

#### 2. 权重计算与融合

在前向传播时（`forward`方法），替换了原来的简单拼接：

**原始方法：**
```python
if self.partial_mode == 3:
    last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
```

**改进方法：**
```python
if self.partial_mode == 3:
    # 计算每个模态的权重
    weight_l = self.modal_weight_net(last_h_l)  # (batch_size, 1)
    weight_a = self.modal_weight_net(last_h_a)  # (batch_size, 1)
    weight_v = self.modal_weight_net(last_h_v)  # (batch_size, 1)
    
    # Softmax归一化
    modal_weights = torch.softmax(torch.cat([weight_l, weight_a, weight_v], dim=1), dim=1)
    
    # 加权融合
    weighted_l = last_h_l * modal_weights[:, 0:1]
    weighted_a = last_h_a * modal_weights[:, 1:2]
    weighted_v = last_h_v * modal_weights[:, 2:3]
    
    # 拼接加权后的特征
    last_hs = torch.cat([weighted_l, weighted_a, weighted_v], dim=1)
```

### 改进优势

1. **自适应性**：能够根据输入样本自动调整模态权重
2. **可解释性**：可以观察不同样本中模态权重的分布
3. **简单有效**：只增加了少量参数（约1800个），但可能带来性能提升
4. **向后兼容**：当只使用单个模态时，行为与原始模型相同

### 代码位置

改进代码位于：`src/models.py`

- 第67-78行：模态权重网络的初始化
- 第151-168行：模态权重融合的前向传播

### 实验对比

建议进行以下对比实验：

1. **原始MulT**：使用原始模型
2. **改进MulT**：使用模态权重融合的改进模型

在相同的数据集和超参数下，对比两者的性能差异。

### 预期效果

- **性能提升**：在大多数情况下，改进模型应该能够达到与原始模型相当或更好的性能
- **模态权重分析**：可以分析不同数据集上各模态的平均权重，了解哪个模态更重要

### 使用说明

改进后的模型使用方法与原始模型完全相同，无需修改训练代码：

```bash
python main.py --dataset mosi --data_path data --name mosi_improved
```

模型会自动使用改进的模态权重融合机制（当使用所有三个模态时）。

### 注意事项

1. 模态权重融合只在**使用所有三个模态**时生效（`partial_mode == 3`）
2. 当只使用单个模态时（`--lonly`, `--aonly`, `--vonly`），行为与原始模型相同
3. 改进增加了少量参数，但不会显著增加训练时间

---

**改进作者**：基于MulT原始架构的改进  
**改进日期**：2024年

