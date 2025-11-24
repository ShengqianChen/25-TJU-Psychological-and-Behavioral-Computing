# 模型使用说明

## 文件说明

### 模型文件
- **`src/models.py`**: 原始MulT模型（保持不变）
- **`src/models_improved.py`**: 改进版MulT模型（添加了模态权重融合）
- **`src/models_baseline.py`**: 简单Baseline模型（用于对比）

### 训练脚本
- **`main.py`**: 原始训练脚本（使用原始MulT模型）
- **`main_improved.py`**: 改进版训练脚本（使用改进MulT模型）
- **`main_baseline.py`**: Baseline训练脚本（使用简单baseline模型）

## 改进内容

改进版模型添加了**模态权重融合机制**：
- 原始模型：简单拼接三个模态的特征
- 改进模型：使用可学习的权重网络，自适应地调整每个模态的重要性

详细说明请参考：`IMPROVEMENT.md`

## 使用方法

### 使用Baseline模型（简单baseline）
```bash
cd /root/心理与行为计算/25-TJU-Psychological-and-Behavioral-Computing
python main_baseline.py --dataset mosi --data_path data --name mosi_baseline
```

### 使用原始MulT模型
```bash
cd /root/心理与行为计算/25-TJU-Psychological-and-Behavioral-Computing
python main.py --dataset mosi --data_path data --name mosi_original
```

### 使用改进MulT模型
```bash
cd /root/心理与行为计算/25-TJU-Psychological-and-Behavioral-Computing
python main_improved.py --dataset mosi --data_path data --name mosi_improved
```

### 对比实验
详细对比实验指南请参考：`COMPARISON_GUIDE.md`

## 参数说明

改进版训练脚本的参数与原始脚本完全相同，使用方法也相同。

**注意**：
- 改进模型默认使用 `--model MulTImproved`
- 模态权重融合只在**使用所有三个模态**时生效（不使用 `--lonly`, `--aonly`, `--vonly`）

## 对比实验

建议运行以下对比实验：

```bash
# 原始模型
python main.py --dataset mosi --data_path data --name mosi_original --num_epochs 20

# 改进模型
python main_improved.py --dataset mosi --data_path data --name mosi_improved --num_epochs 20
```

然后对比两者的性能指标。

## 文件结构

```
25-TJU-Psychological-and-Behavioral-Computing/
├── src/
│   ├── models.py              # 原始MulT模型（保持不变）
│   ├── models_improved.py     # 改进MulT模型（新增）
│   ├── models_baseline.py     # Baseline模型（新增）
│   └── train.py               # 训练函数（保持不变）
├── main.py                    # 原始训练脚本（保持不变）
├── main_improved.py           # 改进版训练脚本（新增）
├── main_baseline.py           # Baseline训练脚本（新增）
├── IMPROVEMENT.md             # 改进说明文档
├── COMPARISON_GUIDE.md        # 对比实验指南（新增）
└── README_IMPROVED.md         # 本文件
```

