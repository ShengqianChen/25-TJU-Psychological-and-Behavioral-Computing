# MulT多模态情感分析项目

基于Multimodal Transformer (MulT)的多模态情感分析项目，包含原始模型、改进模型和Baseline模型的完整实现。

## 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [数据集](#数据集)
- [模型说明](#模型说明)
- [快速开始](#快速开始)
- [训练指南](#训练指南)
- [模型对比实验](#模型对比实验)
- [训练可视化](#训练可视化)
- [文件结构](#文件结构)
- [常见问题](#常见问题)

---

## 项目概述

本项目实现了三个多模态情感分析模型：

1. **Baseline模型** - 简单的MLP融合baseline
2. **MulT原始模型** - 基于Transformer的多模态融合模型
3. **MulT改进模型** - 在MulT基础上添加模态权重融合机制

### 主要特性

- ✅ 支持MOSI、MOSEI、IEMOCAP三个数据集
- ✅ 自动训练日志记录
- ✅ 训练曲线可视化
- ✅ 模型性能对比分析
- ✅ 完整的实验报告支持

---

## 环境要求

### Python环境
- Python 3.6 或 3.7
- PyTorch >= 1.0.0
- CUDA 10.0 或更高版本（如果使用GPU）

### 依赖库

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib
# 可选：用于表格可视化
pip install pandas seaborn
```

---

## 数据集

### 数据集说明

项目支持三个多模态情感分析数据集：

1. **MOSI** (Multimodal Opinion Sentiment and Emotion Intensity)
   - 对齐版本: `mosi_data.pkl` (147MB)
   - 未对齐版本: `mosi_data_noalign.pkl` (326MB)
   - 任务: 情感强度回归

2. **MOSEI** (Multimodal Opinion Sentiment and Emotion Intensity)
   - 对齐版本: `mosei_senti_data.pkl` (3.5GB)
   - 未对齐版本: `mosei_senti_data_noalign.pkl` (12GB)
   - 任务: 情感强度回归（更大的数据集）

3. **IEMOCAP** (Interactive Emotional Dyadic Motion Capture)
   - 对齐版本: `iemocap_data.pkl` (279MB)
   - 未对齐版本: `iemocap_data_noalign.pkl` (1.8GB)
   - 任务: 情感分类（8类）

### 数据准备

数据文件应放在 `data/` 目录下。数据集已包含在项目中。

---

## 模型说明

### 1. Baseline模型

**文件**: `src/models_baseline.py`, `main_baseline.py`

**架构特点**:
- 简单的线性投影将各模态映射到共同维度
- 时序平均池化
- 特征拼接
- 多层MLP融合和预测

**特点**:
- 无注意力机制
- 无跨模态交互
- 最简单的多模态融合方法
- 训练速度快，参数量少

### 2. MulT原始模型

**文件**: `src/models.py`, `main.py`

**架构特点**:
- Transformer跨模态注意力机制
- 自注意力机制
- 残差连接
- 处理未对齐的多模态序列

**特点**:
- 使用Transformer捕获跨模态交互
- 能够处理时序未对齐的多模态数据
- 性能优于简单baseline

### 3. MulT改进模型（模态权重融合）

**文件**: `src/models_improved.py`, `main_improved.py`

**改进内容**: 模态权重融合机制

#### 改进动机

原始MulT模型在融合三个模态时使用简单的特征拼接，假设所有模态重要性相等。但实际上：
- 不同样本中，不同模态的重要性可能不同
- 例如：有些样本中文本信息更重要，有些样本中音频信息更重要
- 简单拼接无法自适应地调整模态权重

#### 改进方案

添加了**模态权重学习模块**：
1. 根据每个样本的特征，自动学习每个模态的重要性权重
2. 使用学习到的权重对模态特征进行加权融合
3. 通过softmax确保权重归一化（权重和为1）

#### 实现细节

**模态权重网络**:
```python
self.modal_weight_net = nn.Sequential(
    nn.Linear(modal_dim, modal_dim // 2),  # 60 -> 30
    nn.ReLU(),
    nn.Dropout(self.embed_dropout),
    nn.Linear(modal_dim // 2, 1)            # 30 -> 1
)
```

**加权融合**:
```python
# 计算每个模态的权重
weight_l = self.modal_weight_net(last_h_l)
weight_a = self.modal_weight_net(last_h_a)
weight_v = self.modal_weight_net(last_h_v)

# Softmax归一化
modal_weights = torch.softmax(torch.cat([weight_l, weight_a, weight_v], dim=1), dim=1)

# 加权融合
weighted_l = last_h_l * modal_weights[:, 0:1]
weighted_a = last_h_a * modal_weights[:, 1:2]
weighted_v = last_h_v * modal_weights[:, 2:3]

# 拼接加权后的特征
last_hs = torch.cat([weighted_l, weighted_a, weighted_v], dim=1)
```

#### 改进优势

1. **自适应性**: 能够根据输入样本自动调整模态权重
2. **可解释性**: 可以观察不同样本中模态权重的分布
3. **简单有效**: 只增加了少量参数（约1800个），但可能带来性能提升
4. **向后兼容**: 当只使用单个模态时，行为与原始模型相同

**注意**: 模态权重融合只在**使用所有三个模态**时生效（`partial_mode == 3`）

---

## 快速开始

### 1. 检查环境

```bash
cd /root/心理与行为计算/25-TJU-Psychological-and-Behavioral-Computing

# 检查Python版本
python --version

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 运行第一个实验

**最简单的开始 - MOSI数据集（小数据集，快速验证）:**

```bash
# Baseline模型
python main_baseline.py --dataset mosi --data_path data --name mosi_baseline --num_epochs 20

# MulT原始模型
python main.py --dataset mosi --data_path data --name mosi_mult_original --num_epochs 20

# MulT改进模型
python main_improved.py --dataset mosi --data_path data --name mosi_mult_improved --num_epochs 20
```

---

## 训练指南

### 基本训练命令

#### MOSI数据集

```bash
# 未对齐，所有模态
python main.py --dataset mosi --data_path data --name mosi_all_modalities

# 仅文本模态
python main.py --dataset mosi --data_path data --lonly --name mosi_text_only

# 仅音频模态
python main.py --dataset mosi --data_path data --aonly --name mosi_audio_only

# 仅视觉模态
python main.py --dataset mosi --data_path data --vonly --name mosi_visual_only
```

#### MOSEI数据集

```bash
# 未对齐，所有模态（默认配置）
python main.py \
    --dataset mosei_senti \
    --data_path data \
    --batch_size 24 \
    --num_epochs 40 \
    --name mosei_all_modalities

# 对齐版本对比
python main.py \
    --dataset mosei_senti \
    --data_path data \
    --aligned \
    --name mosei_aligned
```

#### IEMOCAP数据集

```bash
# IEMOCAP情感分类
python main.py \
    --dataset iemocap \
    --data_path data \
    --batch_size 24 \
    --num_epochs 40 \
    --name iemocap_classification
```

### 常用参数说明

#### 数据集相关
- `--dataset`: 选择数据集 (`mosei_senti`, `mosi`, `iemocap`)
- `--data_path`: 数据路径（默认: `data`）
- `--aligned`: 使用对齐的数据（默认: 未对齐）

#### 模态选择
- `--lonly`: 仅使用文本模态
- `--aonly`: 仅使用音频模态
- `--vonly`: 仅使用视觉模态
- 默认：使用所有三个模态

#### 训练参数
- `--batch_size`: 批次大小（默认: 24）
- `--num_epochs`: 训练轮数（默认: 40）
- `--lr`: 学习率（默认: 1e-3）
- `--clip`: 梯度裁剪值（默认: 0.8）

#### 模型架构
- `--nlevels`: Transformer层数（默认: 5）
- `--num_heads`: 注意力头数（默认: 5）
- `--attn_dropout`: 注意力dropout（默认: 0.1）

#### 其他
- `--name`: 实验名称（默认: `mult`）
- `--no_cuda`: 不使用CUDA（仅CPU训练）
- `--seed`: 随机种子（默认: 1111）

### 训练输出

训练过程中会显示：
- 每个epoch的训练损失
- 验证集损失
- 测试集损失
- 最佳模型会保存在 `pre_trained_models/` 目录

训练结束后会自动评估并显示：
- **MOSEI/MOSI**: MAE, 相关系数, 多分类准确率, F1分数
- **IEMOCAP**: 每个情感类别的F1分数和准确率

**注意**: 所有训练脚本已集成日志记录功能，训练日志会自动保存到 `training_logs/` 目录。

---

## 模型对比实验

### 三个模型对比

| 模型 | 复杂度 | 参数量 | 训练速度 | 预期性能 |
|------|--------|--------|----------|----------|
| **Baseline** | 低 | 少 | 快 | 低（baseline） |
| **MulT原始** | 中 | 中 | 中 | 中（较好） |
| **MulT改进** | 中高 | 中+ | 中 | 高（最好） |

### 对比实验方案

#### 实验1: MOSI数据集对比

```bash
cd /root/心理与行为计算/25-TJU-Psychological-and-Behavioral-Computing

# 1. Baseline模型
python main_baseline.py \
    --dataset mosi \
    --data_path data \
    --name mosi_baseline \
    --num_epochs 20 \
    --batch_size 16

# 2. MulT原始模型
python main.py \
    --dataset mosi \
    --data_path data \
    --name mosi_mult_original \
    --num_epochs 20 \
    --batch_size 16

# 3. MulT改进模型
python main_improved.py \
    --dataset mosi \
    --data_path data \
    --name mosi_mult_improved \
    --num_epochs 20 \
    --batch_size 16
```

#### 实验2: IEMOCAP数据集对比

```bash
# 1. Baseline模型
python main_baseline.py --dataset iemocap --data_path data --name iemocap_baseline --num_epochs 20

# 2. MulT原始模型
python main.py --dataset iemocap --data_path data --name iemocap_mult_original --num_epochs 20

# 3. MulT改进模型
python main_improved.py --dataset iemocap --data_path data --name iemocap_mult_improved --num_epochs 20
```

### 性能指标

#### MOSI/MOSEI（回归任务）
- **MAE** (Mean Absolute Error): 平均绝对误差（越小越好）
- **Correlation**: 相关系数（越大越好）
- **Acc-2/Acc-5/Acc-7**: 多分类准确率（越大越好）
- **F1-score**: F1分数（越大越好）

#### IEMOCAP（分类任务）
- 每个情感类别的F1分数
- 总体准确率

### 预期结果

**性能排序（预期）**:
1. **MulT改进模型** ≥ MulT原始模型 > Baseline模型
   - MulT改进模型应该达到最好或接近最好的性能
   - MulT原始模型应该明显优于Baseline
   - Baseline作为简单方法，性能应该较低

### 结果记录表格

建议创建如下表格记录结果：

| 模型 | 数据集 | MAE | Correlation | Acc-2 | Acc-5 | Acc-7 | F1 |
|------|--------|-----|------------|-------|-------|-------|-----|
| Baseline | MOSI | | | | | | |
| MulT原始 | MOSI | | | | | | |
| MulT改进 | MOSI | | | | | | |
| Baseline | IEMOCAP | | | | | | |
| MulT原始 | IEMOCAP | | | | | | |
| MulT改进 | IEMOCAP | | | | | | |

### 注意事项

1. **确保使用相同的随机种子**: 所有实验使用相同的 `--seed` 参数（默认1111）
2. **相同的超参数**: 除了模型本身，其他超参数保持一致
3. **训练时间**:
   - Baseline: 最快（约10-20分钟）
   - MulT原始: 中等（约30-60分钟）
   - MulT改进: 稍慢（约30-60分钟）
4. **GPU内存**: 如果遇到内存不足，可以减小 `--batch_size`

---

## 训练可视化

### 功能概述

项目提供了完整的训练日志记录和可视化功能：
1. **自动记录训练过程**: 每个epoch的loss、准确率等指标
2. **绘制训练曲线**: loss曲线、准确率曲线等
3. **模型对比可视化**: 对比不同模型的性能
4. **生成实验报告图表**: 可直接用于实验报告

### 自动日志记录

所有训练脚本（`main.py`, `main_improved.py`, `main_baseline.py`）已经集成了日志记录功能。

训练时，日志会自动保存到 `training_logs/` 目录：
```
training_logs/
├── mosi_baseline.json
├── mosi_mult_original.json
├── mosi_mult_improved.json
└── ...
```

每个JSON文件包含完整的训练历史：
```json
{
  "epochs": [1, 2, 3, ...],
  "train_loss": [0.8234, 0.7123, 0.6456, ...],
  "val_loss": [0.9123, 0.8234, 0.7567, ...],
  "test_loss": [0.9234, 0.8345, 0.7678, ...],
  "val_mae": [0.8234, 0.7123, 0.6456, ...],
  "test_mae": [0.8345, 0.7234, 0.6567, ...],
  ...
}
```

### 生成可视化图表

#### 方法1: 自动可视化所有日志

```bash
cd /root/心理与行为计算/25-TJU-Psychological-and-Behavioral-Computing

# 自动可视化所有训练日志
python visualize_training.py

# 或指定目录
python visualize_training.py --log_dir training_logs --output_dir visualizations
```

这会生成以下可视化文件：
- `visualizations/training_curves.png` - 训练曲线对比
- `visualizations/val_loss_comparison.png` - 验证loss对比
- `visualizations/final_metrics_comparison.png` - 最终指标对比
- `visualizations/comparison_table.png` - 模型对比表格

#### 方法2: 自定义可视化

```python
from visualize_training import *

# 指定要对比的模型
log_paths = [
    'training_logs/mosi_baseline.json',
    'training_logs/mosi_mult_original.json',
    'training_logs/mosi_mult_improved.json'
]
labels = ['Baseline', 'MulT Original', 'MulT Improved']

# 绘制训练曲线
plot_training_curves(log_paths, labels, save_path='my_training_curves.png')

# 对比特定指标
plot_metric_comparison(log_paths, labels, metric='val_loss', 
                      save_path='val_loss_comparison.png')
```

### 模态权重可视化（改进模型）

对于MulT改进模型，还可以可视化模态权重的分布：

```bash
# 创建模型架构对比图
python visualize_modal_weights.py --output_dir visualizations
```

这会生成：
- `model_architecture.png` - 三个模型的架构对比图

### 完整工作流程示例

```bash
# 1. 训练三个模型（会自动记录日志）
python main_baseline.py --dataset mosi --data_path data --name mosi_baseline --num_epochs 20
python main.py --dataset mosi --data_path data --name mosi_mult_original --num_epochs 20
python main_improved.py --dataset mosi --data_path data --name mosi_mult_improved --num_epochs 20

# 2. 生成可视化
python visualize_training.py
python visualize_modal_weights.py --output_dir visualizations

# 3. 查看生成的可视化文件
ls visualizations/
```

### 实验报告使用建议

#### 1. 训练过程分析
- **训练曲线图**: 展示模型收敛过程
- **Loss对比图**: 对比不同模型的loss下降速度
- **过拟合分析**: 观察训练loss和验证loss的差距

#### 2. 性能对比
- **最终指标对比柱状图**: 清晰展示各模型的最终性能
- **对比表格**: 详细列出所有指标

#### 3. 模型分析
- **收敛速度对比**: 哪个模型收敛更快
- **最佳epoch对比**: 不同模型在哪个epoch达到最佳性能

---

## 文件结构

```
25-TJU-Psychological-and-Behavioral-Computing/
├── src/
│   ├── models.py              # 原始MulT模型
│   ├── models_improved.py     # 改进MulT模型（模态权重融合）
│   ├── models_baseline.py     # Baseline模型
│   ├── train.py               # 原始训练函数
│   ├── train_with_logging.py  # 带日志记录的训练函数
│   ├── training_logger.py     # 训练日志记录器
│   ├── dataset.py             # 数据集加载
│   ├── eval_metrics.py        # 评估指标（计算MAE、相关系数、准确率等）
│   ├── utils.py               # 工具函数
│   └── README.md              # 原始说明（已整合到主README）
├── modules/
│   ├── transformer.py         # Transformer模块
│   ├── multihead_attention.py # 多头注意力机制
│   ├── position_embedding.py  # 位置编码
│   └── ...
├── main.py                    # 原始MulT训练脚本
├── main_improved.py           # 改进MulT训练脚本
├── main_baseline.py           # Baseline训练脚本
├── visualize_training.py      # 训练可视化脚本
├── visualize_modal_weights.py # 模态权重可视化脚本
├── data/                      # 数据集目录
│   ├── mosi_data.pkl          # MOSI对齐数据
│   ├── mosi_data_noalign.pkl  # MOSI未对齐数据
│   ├── mosei_senti_data.pkl   # MOSEI对齐数据
│   ├── mosei_senti_data_noalign.pkl # MOSEI未对齐数据
│   ├── iemocap_data.pkl       # IEMOCAP对齐数据
│   └── iemocap_data_noalign.pkl # IEMOCAP未对齐数据
├── training_logs/             # 训练日志目录（自动生成，JSON格式）
├── visualizations/            # 可视化输出目录（自动生成，PNG格式）
├── pre_trained_models/         # 保存的模型目录（自动生成）
└── README.md                  # 本文件（完整文档）
```

### 核心文件说明

**模型文件**:
- `src/models.py`: 原始MulT模型实现，包含跨模态Transformer注意力机制
- `src/models_improved.py`: 改进版MulT，添加了模态权重融合机制
- `src/models_baseline.py`: 简单baseline模型，用于性能对比

**训练脚本**:
- `main.py`: 使用原始MulT模型训练
- `main_improved.py`: 使用改进MulT模型训练
- `main_baseline.py`: 使用Baseline模型训练

**评估和可视化**:
- `src/eval_metrics.py`: 评估指标计算（MAE、相关系数、准确率等）
- `visualize_training.py`: 训练曲线和性能对比可视化
- `visualize_modal_weights.py`: 模态权重分布可视化

**数据说明**:
- 所有数据集应放在 `data/` 目录下
- 根据数据集类型，使用不同的评估指标（见 `eval_metrics.py`）

---

## 常见问题

### 环境问题

#### 如果遇到 "No module named" 错误
```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib
```

#### 如果遇到CUDA错误
```bash
# 使用CPU训练
python main.py --dataset mosi --data_path data --no_cuda
```

### 训练问题

#### 如果遇到内存错误
```bash
# 减小batch size
python main.py --dataset mosi --data_path data --batch_size 8

# 或使用batch chunking
python main.py --dataset mosi --data_path data --batch_chunk 2
```

#### 如果数据加载失败
- 检查 `data/` 目录下是否有对应的 `.pkl` 文件
- 确认文件名格式正确（`{dataset}_data_noalign.pkl` 或 `{dataset}_data.pkl`）

### 性能问题

#### GPU内存不足
- 减小 `--batch_size`（如改为16或8）
- 使用 `--batch_chunk` 参数将批次分块处理

#### 训练时间过长
- Baseline: 约30分钟-1小时
- MOSEI: 约2-4小时
- IEMOCAP: 约1-2小时

### 可视化问题

#### 如果可视化脚本报错
```bash
# 确保安装了必要的库
pip install matplotlib numpy
# 可选
pip install pandas seaborn
```

---

## 实验报告建议

### 1. 实验设置
- 数据集描述
- 模型架构（三个模型的对比）
- 超参数设置
- **改进方法说明**（模态权重融合的原理和实现）

### 2. 实验结果
- 不同数据集的性能对比
- 不同模态组合的对比
- 对齐 vs 未对齐的对比
- 超参数影响分析
- **改进前后性能对比**
- **模态权重分析**: 分析不同样本/数据集上各模态的权重分布

### 3. 分析与讨论
- 模型在多模态融合上的表现
- 不同模态的贡献
- 未对齐数据的处理效果
- **模态权重分析**: 讨论模态权重融合带来的改进
- **改进效果分析**: 讨论模态权重融合带来的改进

---

## 引用

如果使用本项目，请引用原始MulT论文：

```bibtex
@inproceedings{tsai2019MULT,
  title={Multimodal Transformer for Unaligned Multimodal Language Sequences},
  author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Liang, Paul Pu and Kolter, J. Zico and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month = {7},
  year={2019},
  address = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
}
```

---

**祝你实验顺利！** 🚀

