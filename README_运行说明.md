# SOC 光谱建模脚本（ATR-FTIR / LIBS）

本目录包含 4 个独立 Python 脚本，用于论文"Uneven calibration support drives prediction-range compression in soil organic carbon spectroscopy: Correction through targeted spectral generation"的数据分析。

## 1. 文件说明

| 文件 | 功能 |
|---|---|
| `01_missingness_analysis.py` | CNN 潜在空间局部稀疏度（local sparsity）与预测误差关系分析 |
| `02_SHAP_high_SOC_removal.py` | 高 SOC 训练样本逐步去除（0–50%）+ GradientSHAP + Q4 SHAP 联合 PCA |
| `03_CNN_model.py` | ATR-FTIR 和 LIBS 基础 1D-CNN 训练、测试、评价及潜在特征导出 |
| `04_CAVE_CNN_model.py` | SNV-KNN 域筛选、PCA-CVAE 高 SOC 定向生成及 CAVE-CNN 训练 |
| `requirements.txt` | Python 依赖 |

## 2. 输入数据要求

使用已经准备好的数据 `../data/prepared_data.pkl`。该文件应为字典结构，至少包含：

```python
total_data = {
    "SOC": {
        "y_train": ...,   # 内部训练集 SOC，一维数组
        "y_test": ...,    # 独立测试集 SOC，一维数组
        "y_ext": ...,     # 外部数据 SOC，一维数组（CAVE-CNN 必需）
    },
    "ATR_spec": {
        "X_train": ...,   # shape = (n_train, n_ATR_features)
        "X_test": ...,    # shape = (n_test, n_ATR_features)
        "X_ext": ...,     # shape = (n_ext, n_ATR_features)（CAVE-CNN 必需）
        # 可选：axis / spectral_axis / wavenumbers
    },
    "LIBS_spec": {
        "X_train": ...,
        "X_test": ...,
        "X_ext": ...,
        # 可选：axis / spectral_axis / wavelengths
    }
}
```

`X_train`、`X_test`、`X_ext` 均须为二维数值矩阵，SOC 标签须为一维数值数组；同一模态的光谱和 SOC 标签必须严格一一对应。脚本不会静默填补光谱或 SOC 中的 NaN/Inf。


## 3. 推荐运行顺序

### Step 1. 训练基础 CNN

```bash
python 03_CNN_model.py --data prepared_data.pkl --outdir outputs/CNN
```

默认同时训练 ATR 和 LIBS。结果包括：
- `outputs/CNN/ATR/`、`outputs/CNN/LIBS/`：模型权重、测试集预测、评价指标、训练历史和潜在特征；
- `outputs/CNN/atr_cnn.pkl`、`outputs/CNN/libs_cnn.pkl`：兼容原 Notebook 的预测结果文件。

如果需要对基础 CNN 也使用 SNV：

```bash
python 03_CNN_model.py --data prepared_data.pkl --outdir outputs/CNN_SNV --snv
```

### Step 2. 数据覆盖缺口 / 潜在空间局部稀疏度分析

```bash
python 01_missingness_analysis.py --cnn-dir outputs/CNN --outdir outputs/missingness
```

该脚本读取 Step 1 导出的 `latent_train`、`latent_test` 和测试集预测值，计算测试样本到训练样本的 Mahalanobis KNN 平均距离，并分析其与绝对预测误差的关系。

### Step 3. SHAP–高 SOC 样本去除试验

```bash
python 02_SHAP_high_SOC_removal.py --data prepared_data.pkl --outdir outputs/SHAP
```

每个光谱模态分别训练 6 个 CNN：完整训练集（0%）以及依次去除训练集中 SOC 最高的 10%、20%、30%、40%、50%。

输出包括：
- 每个 removal condition 的 CNN 权重和测试集预测；
- 每个 condition 的 SHAP 矩阵；
- 全测试集 mean(|SHAP|) 曲线；
- 测试集 Q4 的 SHAP 区域汇总（需要真实波数/波长轴）；
- 所有 removal condition 的 Q4 SHAP 联合 PCA 得分和载荷；
- 总体和 Q4 分层预测性能。

如基础 CNN 使用了 SNV，SHAP 分析必须使用同样的预处理：

```bash
python 02_SHAP_high_SOC_removal.py --data prepared_data.pkl --outdir outputs/SHAP_SNV --snv
```

若 `prepared_data.pkl` 没有包含实际波数/波长轴，可提供轴文件（`.npy` 或单列 CSV）：

```bash
python 02_SHAP_high_SOC_removal.py \
  --data prepared_data.pkl \
  --outdir outputs/SHAP \
  --atr-axis-file atr_axis.csv \
  --libs-axis-file libs_axis.csv
```

不提供真实光谱轴时，脚本仍会输出按通道索引计算的 SHAP 结果，但不会执行 ATR 官能团区间或 LIBS 元素谱线区间汇总。

### Step 4. CAVE-CNN

```bash
python 04_CAVE_CNN_model.py --data prepared_data.pkl --outdir outputs/CAVE
```

默认参数为 `layer2_soc5_bin10`：
- CVAE：2 层隐藏层，hidden_dim=64，500 epochs；
- PCA：累计解释方差 99%；
- 高 SOC 阈值：5 g/kg；
- SOC 等宽分箱：10；
- 每个高 SOC 分箱目标样本数：60；
- 外部光谱 KNN：k=5；
- CNN：Adam lr=0.001，最多 1000 epochs，patience=100。

如果需要执行 Notebook 中的完整参数网格（layer 1/2 × threshold 0/5/10/15/20 × bins 10/15/20/25/30），运行：

```bash
python 04_CAVE_CNN_model.py --data prepared_data.pkl --outdir outputs/CAVE_grid --grid-search
```

## 5. 依赖安装

```bash
pip install -r requirements.txt
```

PyTorch 如需 GPU 加速，建议根据本机 CUDA 版本从 PyTorch 官方安装页面选择对应安装命令。
