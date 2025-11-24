# MulT 模型训练指南

## 环境要求

1. **Python**: 3.6 或 3.7
2. **PyTorch**: >= 1.0.0
3. **CUDA**: 10.0 或更高版本（如果使用GPU）
4. **其他依赖**:
   - numpy
   - scipy
   - scikit-learn
   - torchvision

## 安装依赖

```bash
pip install torch torchvision numpy scipy scikit-learn
```

## 数据准备

数据文件应该已经放在 `data/` 目录下：
- `mosei_senti_data_noalign.pkl` / `mosei_senti_data.pkl`
- `mosi_data_noalign.pkl` / `mosi_data.pkl`
- `iemocap_data_noalign.pkl` / `iemocap_data.pkl`

## 训练命令

### 1. MOSEI 数据集（默认，未对齐模式）

```bash
python main.py --dataset mosei_senti --data_path data
```

### 2. MOSEI 数据集（对齐模式）

```bash
python main.py --dataset mosei_senti --data_path data --aligned
```

### 3. MOSI 数据集（未对齐模式）

```bash
python main.py --dataset mosi --data_path data
```

### 4. IEMOCAP 数据集（未对齐模式）

```bash
python main.py --dataset iemocap --data_path data
```

## 常用参数说明

### 数据集相关
- `--dataset`: 选择数据集 (`mosei_senti`, `mosi`, `iemocap`)
- `--data_path`: 数据路径（默认: `data`）
- `--aligned`: 使用对齐的数据（默认: 未对齐）

### 模态选择
- `--lonly`: 仅使用文本模态
- `--aonly`: 仅使用音频模态
- `--vonly`: 仅使用视觉模态
- 默认：使用所有三个模态

### 训练参数
- `--batch_size`: 批次大小（默认: 24）
- `--num_epochs`: 训练轮数（默认: 40）
- `--lr`: 学习率（默认: 1e-3）
- `--clip`: 梯度裁剪值（默认: 0.8）

### 模型架构
- `--nlevels`: Transformer层数（默认: 5）
- `--num_heads`: 注意力头数（默认: 5）
- `--attn_dropout`: 注意力dropout（默认: 0.1）

### 其他
- `--name`: 实验名称（默认: `mult`）
- `--no_cuda`: 不使用CUDA（仅CPU训练）
- `--seed`: 随机种子（默认: 1111）

## 完整训练示例

### 示例1: MOSEI 完整训练（未对齐，所有模态）
```bash
python main.py \
    --dataset mosei_senti \
    --data_path data \
    --batch_size 24 \
    --num_epochs 40 \
    --lr 1e-3 \
    --nlevels 5 \
    --num_heads 5 \
    --name mosei_mult_unaligned
```

### 示例2: IEMOCAP 训练（未对齐，所有模态）
```bash
python main.py \
    --dataset iemocap \
    --data_path data \
    --batch_size 24 \
    --num_epochs 40 \
    --lr 1e-3 \
    --name iemocap_mult
```

### 示例3: 仅使用文本模态训练
```bash
python main.py \
    --dataset mosei_senti \
    --data_path data \
    --lonly \
    --name mosei_text_only
```

## 训练输出

训练过程中会显示：
- 每个epoch的训练损失
- 验证集损失
- 测试集损失
- 最佳模型会保存在 `pre_trained_models/` 目录

训练结束后会自动评估并显示：
- **MOSEI/MOSI**: MAE, 相关系数, 多分类准确率, F1分数
- **IEMOCAP**: 每个情感类别的F1分数和准确率

## 注意事项

1. **GPU内存**: 如果遇到GPU内存不足，可以减小 `--batch_size`
2. **训练时间**: 完整训练可能需要数小时，取决于数据集大小和硬件
3. **模型保存**: 最佳模型会自动保存，基于验证集损失
4. **数据缓存**: 首次运行会创建数据缓存文件（`.dt`文件），后续运行会更快

## 故障排除

### 如果遇到CUDA错误
- 使用 `--no_cuda` 进行CPU训练（会很慢）
- 检查CUDA版本是否兼容

### 如果遇到内存错误
- 减小 `--batch_size`
- 使用 `--batch_chunk` 参数将批次分块处理

### 如果数据加载失败
- 检查 `data/` 目录下是否有对应的 `.pkl` 文件
- 确认文件名格式正确（`{dataset}_data_noalign.pkl` 或 `{dataset}_data.pkl`）

