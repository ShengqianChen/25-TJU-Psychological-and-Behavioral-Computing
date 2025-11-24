# 模型对比实验指南

## 三个模型对比

本项目包含三个模型，用于性能对比：

### 1. Baseline模型（简单baseline）
- **文件**: `src/models_baseline.py`, `main_baseline.py`
- **架构**: 
  - 简单的线性投影将各模态映射到共同维度
  - 时序平均池化
  - 特征拼接
  - 多层MLP融合和预测
- **特点**: 
  - 无注意力机制
  - 无跨模态交互
  - 最简单的多模态融合方法

### 2. MulT原始模型
- **文件**: `src/models.py`, `main.py`
- **架构**:
  - Transformer跨模态注意力机制
  - 自注意力机制
  - 残差连接
- **特点**:
  - 使用Transformer捕获跨模态交互
  - 处理未对齐的多模态序列

### 3. MulT改进模型（模态权重融合）
- **文件**: `src/models_improved.py`, `main_improved.py`
- **架构**:
  - 在MulT基础上添加模态权重融合
  - 可学习的模态重要性权重
  - 加权特征融合
- **特点**:
  - 自适应调整模态权重
  - 在MulT基础上进一步改进

## 对比实验方案

### 实验1: MOSI数据集对比

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

### 实验2: IEMOCAP数据集对比

```bash
# 1. Baseline模型
python main_baseline.py \
    --dataset iemocap \
    --data_path data \
    --name iemocap_baseline \
    --num_epochs 20

# 2. MulT原始模型
python main.py \
    --dataset iemocap \
    --data_path data \
    --name iemocap_mult_original \
    --num_epochs 20

# 3. MulT改进模型
python main_improved.py \
    --dataset iemocap \
    --data_path data \
    --name iemocap_mult_improved \
    --num_epochs 20
```

### 实验3: 单模态对比（可选）

```bash
# 仅文本模态
python main_baseline.py --dataset mosi --data_path data --lonly --name mosi_baseline_text
python main.py --dataset mosi --data_path data --lonly --name mosi_mult_text
python main_improved.py --dataset mosi --data_path data --lonly --name mosi_improved_text
```

## 性能指标对比

### MOSI/MOSEI（回归任务）
- **MAE** (Mean Absolute Error): 平均绝对误差（越小越好）
- **Correlation**: 相关系数（越大越好）
- **Acc-2/Acc-5/Acc-7**: 多分类准确率（越大越好）
- **F1-score**: F1分数（越大越好）

### IEMOCAP（分类任务）
- 每个情感类别的F1分数
- 总体准确率

## 预期结果

### 性能排序（预期）
1. **MulT改进模型** ≥ MulT原始模型 > Baseline模型
   - MulT改进模型应该达到最好或接近最好的性能
   - MulT原始模型应该明显优于Baseline
   - Baseline作为简单方法，性能应该较低

### 分析要点

1. **Baseline vs MulT原始模型**
   - 展示Transformer和注意力机制的优势
   - 跨模态交互的重要性

2. **MulT原始 vs MulT改进**
   - 展示模态权重融合的改进效果
   - 分析不同数据集上模态权重分布

3. **计算效率对比**
   - Baseline: 最快，参数最少
   - MulT原始: 中等速度，参数较多
   - MulT改进: 稍慢，参数略多

## 结果记录表格

建议创建如下表格记录结果：

| 模型 | 数据集 | MAE | Correlation | Acc-2 | Acc-5 | Acc-7 | F1 |
|------|--------|-----|------------|-------|-------|-------|-----|
| Baseline | MOSI | | | | | | |
| MulT原始 | MOSI | | | | | | |
| MulT改进 | MOSI | | | | | | |
| Baseline | IEMOCAP | | | | | | |
| MulT原始 | IEMOCAP | | | | | | |
| MulT改进 | IEMOCAP | | | | | | |

## 快速对比脚本

创建一个批量运行脚本：

```bash
cat > run_comparison.sh << 'EOF'
#!/bin/bash

cd /root/心理与行为计算/25-TJU-Psychological-and-Behavioral-Computing

DATASET="mosi"
EPOCHS=20
BATCH_SIZE=16

echo "Running Baseline model..."
python main_baseline.py \
    --dataset $DATASET \
    --data_path data \
    --name ${DATASET}_baseline \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE

echo "Running MulT Original model..."
python main.py \
    --dataset $DATASET \
    --data_path data \
    --name ${DATASET}_mult_original \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE

echo "Running MulT Improved model..."
python main_improved.py \
    --dataset $DATASET \
    --data_path data \
    --name ${DATASET}_mult_improved \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE

echo "All experiments completed!"
EOF

chmod +x run_comparison.sh
```

## 注意事项

1. **确保使用相同的随机种子**：所有实验使用相同的 `--seed` 参数（默认1111）
2. **相同的超参数**：除了模型本身，其他超参数保持一致
3. **训练时间**：
   - Baseline: 最快（约10-20分钟）
   - MulT原始: 中等（约30-60分钟）
   - MulT改进: 稍慢（约30-60分钟）
4. **GPU内存**：如果遇到内存不足，可以减小 `--batch_size`

## 实验报告建议

### 1. 模型架构对比
- 描述三个模型的架构差异
- 说明各自的优势和特点

### 2. 实验结果
- 表格展示性能对比
- 可视化性能指标（柱状图等）

### 3. 分析与讨论
- **为什么MulT优于Baseline？**
  - Transformer注意力机制的作用
  - 跨模态交互的重要性
  
- **改进模型的效果如何？**
  - 模态权重融合带来的提升
  - 不同数据集上的表现差异

- **计算效率对比**
  - 参数量对比
  - 训练时间对比
  - 推理速度对比

---

**祝你实验顺利！** 🚀

