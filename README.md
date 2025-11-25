# 基于Multimodal Transformer的多模态情感分析

本项目是2025-2026第2学年天津大学智能与计算学部心理与行为计算课程大作业，项目实现了三个多模态情感分析模型：

1. **Baseline模型** - 仅将Transformer部分架构改为相同参数量MLP的Baseline
2. **MulT原始模型** - 基于Transformer的多模态融合模型
3. **MulT改进模型** - 在MulT基础上改进输出层从单层MLP为多层MLP

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

## 训练指南

### 参数说明

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

### 性能指标

#### MOSI/MOSEI（回归任务）
- **MAE** (Mean Absolute Error): 平均绝对误差（越小越好）
- **Correlation**: 相关系数（越大越好）
- **Acc-2/Acc-5/Acc-7**: 多分类准确率（越大越好）
- **F1-score**: F1分数（越大越好）

#### IEMOCAP（分类任务）
- 每个情感类别的F1分数
- 总体准确率


## 文件结构

```
25-TJU-Psychological-and-Behavioral-Computing/
├── src/
│   ├── models.py              # 原始MulT模型
│   ├── models_improved.py     # 改进MulT模型
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


