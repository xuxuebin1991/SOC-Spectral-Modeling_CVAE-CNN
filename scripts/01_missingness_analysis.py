#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_missingness_analysis.py
基于 CNN 潜在空间的局部样本稀疏度（local sparsity）/校准域覆盖不足分析。

输入：
03_CNN_model.py 输出的
outputs/CNN/ATR/atr_cnn.pkl
outputs/CNN/LIBS/libs_cnn.pkl

运行：
python 01_missingness_analysis.py --cnn-dir outputs/CNN --outdir outputs/missingness
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def load_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 CNN 结果文件：{path.resolve()}\n"
            "请先运行 03_CNN_model.py，或将 --cnn-dir 指向已有 CNN 结果目录。"
        )
    with path.open("rb") as f:
        obj = pickle.load(f)

    required = ["latent_train", "latent_test", "test", "test_pred"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(
            f"{path} 缺少字段 {missing}。需要使用 03_CNN_model.py 生成的完整结果文件，"
            "其中必须包含 latent_train / latent_test / test / test_pred。"
        )
    return obj


def calculate_local_sparsity(
    latent_train: np.ndarray,
    latent_test: np.ndarray,
    k: int = 5,
    max_pca_components: int = 50,
):
    """
    先将高维 CNN latent features 降至最多 50 个 PCA 维度，
    再以 Ledoit-Wolf shrinkage covariance 计算 Mahalanobis KNN 距离。
    """
    Xtr = np.asarray(latent_train, dtype=float)
    Xte = np.asarray(latent_test, dtype=float)

    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError("latent_train 和 latent_test 必须为二维矩阵。")
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError("训练集和测试集 latent feature 维度不一致。")
    if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
        raise ValueError("latent features 中存在 NaN/Inf。")
    if len(Xtr) < 3:
        raise ValueError("训练样本数过少，无法可靠估计潜在空间协方差。")

    n_components = min(max_pca_components, Xtr.shape[1], len(Xtr) - 1)
    if n_components < Xtr.shape[1]:
        pca = PCA(n_components=n_components, random_state=42)
        Ztr = pca.fit_transform(Xtr)
        Zte = pca.transform(Xte)
        explained = float(pca.explained_variance_ratio_.sum())
    else:
        pca = None
        Ztr, Zte = Xtr, Xte
        explained = np.nan

    cov_estimator = LedoitWolf().fit(Ztr)
    precision = cov_estimator.precision_

    k_eff = min(int(k), len(Ztr))
    nn = NearestNeighbors(
        n_neighbors=k_eff,
        metric="mahalanobis",
        metric_params={"VI": precision},
        algorithm="brute",
    )
    nn.fit(Ztr)
    distances, indices = nn.kneighbors(Zte)
    local_sparsity = distances.mean(axis=1)

    return local_sparsity, distances, indices, {
        "pca_components": int(n_components),
        "pca_explained_variance": explained,
        "k_neighbors": int(k_eff),
        "covariance_estimator": "LedoitWolf",
    }


def analyze_one(modality: str, result: dict, outdir: Path, k: int, max_pca_components: int):
    latent_train = np.asarray(result["latent_train"], dtype=float)
    latent_test = np.asarray(result["latent_test"], dtype=float)
    y_true = np.asarray(result["test"], dtype=float).ravel()
    y_pred = np.asarray(result["test_pred"], dtype=float).ravel()

    if len(y_true) != len(y_pred) or len(y_true) != len(latent_test):
        raise ValueError(f"{modality}: test labels, predictions, and latent features are not aligned.")

    local_sparsity, distances, indices, info = calculate_local_sparsity(
        latent_train, latent_test, k=k, max_pca_components=max_pca_components
    )
    abs_error = np.abs(y_pred - y_true)
    residual = y_pred - y_true

    mask = np.isfinite(local_sparsity) & np.isfinite(abs_error)
    x = local_sparsity[mask]
    y = abs_error[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        raise ValueError(f"{modality}: local sparsity 或绝对误差无变异，无法进行相关/回归分析。")

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    rho, p_spearman = spearmanr(x, y)

    outdir.mkdir(parents=True, exist_ok=True)
    sample_table = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "SOC_measured": y_true,
        "SOC_predicted": y_pred,
        "residual_pred_minus_measured": residual,
        "absolute_error": abs_error,
        "local_sparsity_mahalanobis_knn": local_sparsity,
        "nearest_train_mean_distance": distances.mean(axis=1),
    })
    sample_table.to_csv(outdir / f"{modality.lower()}_local_sparsity_samples.csv", index=False)

    summary = {
        "Modality": modality,
        "n_train": int(len(latent_train)),
        "n_test": int(len(latent_test)),
        **info,
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "Pearson_r": float(r_value),
        "linear_R2": float(r_value ** 2),
        "linear_p": float(p_value),
        "Spearman_rho": float(rho),
        "Spearman_p": float(p_spearman),
        "mean_local_sparsity": float(np.mean(local_sparsity)),
        "mean_absolute_error": float(np.mean(abs_error)),
    }
    pd.DataFrame([summary]).to_csv(outdir / f"{modality.lower()}_local_sparsity_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        local_sparsity, abs_error, c=abs_error, cmap="YlOrRd",
        alpha=0.75, edgecolors="white", linewidths=0.35, s=42
    )
    order = np.argsort(local_sparsity)
    ax.plot(
        local_sparsity[order],
        (slope * local_sparsity + intercept)[order],
        linestyle="--", linewidth=1.8, color="black"
    )
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    rho_text = "p < 0.001" if p_spearman < 0.001 else f"p = {p_spearman:.3f}"
    ax.text(
        0.04, 0.96,
        f"Slope = {slope:.3f}\nPearson r = {r_value:.3f}; R² = {r_value**2:.3f}\n"
        f"{p_text}\nSpearman ρ = {rho:.3f}; {rho_text}\n"
        f"PCA dimensions = {info['pca_components']}",
        transform=ax.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85}
    )
    fig.colorbar(sc, ax=ax, label="Absolute prediction error (g/kg)")
    ax.set_xlabel(f"Local sparsity in CNN latent space\nMean Mahalanobis distance to {info['k_neighbors']} nearest training samples")
    ax.set_ylabel("|Predicted SOC − measured SOC| (g/kg)")
    ax.set_title(f"{modality}: latent-space sparsity and prediction error")
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(outdir / f"{modality.lower()}_local_sparsity_vs_error.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{modality.lower()}_local_sparsity_vs_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{modality} local-sparsity analysis:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="CNN latent-space local sparsity analysis.")
    parser.add_argument("--cnn-dir", default="outputs/CNN")
    parser.add_argument("--outdir", default="outputs/missingness")
    parser.add_argument("--modalities", nargs="+", default=["ATR", "LIBS"], choices=["ATR", "LIBS"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-pca-components", type=int, default=50)
    args = parser.parse_args()

    cnn_dir = Path(args.cnn_dir)
    outdir = Path(args.outdir)
    summaries = []
    for modality in args.modalities:
        result_path = cnn_dir / modality / f"{modality.lower()}_cnn.pkl"
        result = load_result(result_path)
        summaries.append(analyze_one(
            modality, result, outdir, args.k, args.max_pca_components
        ))

    pd.DataFrame(summaries).to_csv(outdir / "local_sparsity_model_comparison.csv", index=False)
    print(f"\nFinished. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
