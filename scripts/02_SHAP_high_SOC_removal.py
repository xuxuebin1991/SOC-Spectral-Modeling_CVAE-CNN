#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_SHAP_high_SOC_removal.py
高 SOC 校准样本逐步去除试验 + CNN GradientSHAP 解释。

- 移除训练集中 SOC 最高的 10%、20%、30%、40%、50%；
- 0% removal（完整训练集）作为参照；
- 每个条件重新训练相同结构的 CNN；
- 所有条件使用相同独立测试集；
- SHAP 使用同一组原始训练光谱作为 background；
- 比较全测试集及 Q4（测试集 SOC 上四分位）SHAP 归因变化；
- 对所有 removal 条件的 Q4 SHAP 矩阵联合 PCA，保证投影空间一致。

运行：
python 02_SHAP_high_SOC_removal.py --data ../data/prepared_data.pkl --outdir outputs/SHAP
"""

from __future__ import annotations

import argparse
import pickle
import random
import warnings
from pathlib import Path

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


REMOVAL_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SoilSpectralCNN(nn.Module):
    """与 cnn.ipynb 中的 SoilSpectralCNN 保持一致。"""

    def __init__(self, input_length):
        super().__init__()
        if input_length < 16:
            raise ValueError("CNN 输入光谱长度必须至少为 16。")
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(24, 24, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(24, 48, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )
        with torch.no_grad():
            flattened_size = int(self.conv_block(torch.zeros(1, 1, input_length)).numel())
        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")
        x = self.conv_block(x)
        x = x.reshape(x.size(0), -1)
        return self.fc_block(x)


def load_total_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path.resolve()}")
    with path.open("rb") as f:
        data = pickle.load(f)
    for key in ["SOC", "ATR_spec", "LIBS_spec"]:
        if key not in data:
            raise KeyError(f"prepared_data.pkl 缺少顶层字段 {key}")
    for key in ["y_train", "y_test"]:
        if key not in data["SOC"]:
            raise KeyError(f"prepared_data.pkl 缺少 SOC.{key}")
    return data


def as_matrix(x, name):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} 应为二维矩阵，当前 shape={x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} 含 NaN/Inf，请先检查数据。")
    return x


def as_vector(y, name):
    y = np.asarray(y, dtype=np.float32).ravel()
    if not np.isfinite(y).all():
        raise ValueError(f"{name} 含 NaN/Inf。")
    return y


def snv_transform(X):
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def get_groups(groups_npy, n):
    if not groups_npy:
        return None
    groups = np.asarray(np.load(groups_npy, allow_pickle=True)).ravel()
    if len(groups) != n:
        raise ValueError(f"groups 文件长度 {len(groups)} 与训练样本数 {n} 不一致。")
    return groups


def split_fit_validation(n, val_size, seed, groups=None):
    idx = np.arange(n)
    return train_test_split(idx, test_size=val_size, random_state=seed, shuffle=True)


def make_loader(X, y, batch_size, shuffle, seed):
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def train_until_best_epoch(
    X_fit, y_fit, X_val, y_val, device,
    lr=1e-4, batch_size=32, epochs=500, patience=100, seed=42
):
    set_seed(seed)
    model = SoilSpectralCNN(X_fit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    fit_loader = make_loader(X_fit, y_fit, batch_size, True, seed)
    val_loader = make_loader(X_val, y_val, batch_size, False, seed)

    best_val = float("inf")
    best_epoch = 1
    history = []
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).reshape(-1)
            loss = loss_fn(pred, yb.reshape(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(float(loss_fn(model(xb).reshape(-1), yb.reshape(-1)).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch == 1 or epoch % 10 == 0:
            print(f"    Epoch {epoch:4d}/{epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping; selected epoch={best_epoch}")
                break

    return best_epoch, history


def train_full_data(X, y, epochs, device, lr=1e-4, batch_size=32, seed=42):
    set_seed(seed)
    model = SoilSpectralCNN(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = make_loader(X, y, batch_size, True, seed)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).reshape(-1)
            loss = loss_fn(pred, yb.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"    Final fit {epoch:4d}/{epochs} | loss={np.mean(losses):.5f}")
    return model


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    result = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
        result.append(model(xb).reshape(-1).cpu().numpy())
    return np.concatenate(result)


def calculate_metrics(y, pred):
    y = np.asarray(y).ravel()
    pred = np.asarray(pred).ravel()
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    r2 = float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))
    if len(y) > 1 and np.std(y) > 0:
        slope, intercept, r, p, _ = linregress(y, pred - y)
        slope_mp, intercept_mp, _, _, _ = linregress(y, pred)
    else:
        slope = intercept = r = p = slope_mp = intercept_mp = np.nan
    return {
        "R2": r2, "RMSE": rmse, "MAE": mae,
        "slope_measured_predicted": float(slope_mp),
        "intercept_measured_predicted": float(intercept_mp),
        "slope_residual_SOC": float(slope),
        "intercept_residual_SOC": float(intercept),
        "residual_r": float(r), "residual_p": float(p),
    }


def normalize_shap_shape(values, n_samples, n_features):
    """兼容不同 SHAP 版本的 regression 输出维度。"""
    if isinstance(values, list):
        values = values[0]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values)
    if arr.size != n_samples * n_features:
        arr = np.squeeze(arr)
    if arr.size != n_samples * n_features:
        raise ValueError(
            f"SHAP 输出无法整理为 ({n_samples}, {n_features})，实际 shape={arr.shape}"
        )
    return arr.reshape(n_samples, n_features)


def load_axis(data, modality, n_features, axis_file=None):
    physical_axis = False
    if axis_file:
        p = Path(axis_file)
        if p.suffix.lower() == ".npy":
            axis = np.asarray(np.load(p), dtype=float).ravel()
        else:
            tab = pd.read_csv(p, header=None)
            axis = pd.to_numeric(tab.iloc[:, 0], errors="coerce").dropna().to_numpy(float)
        physical_axis = True
    else:
        spec_key = "ATR_spec" if modality == "ATR" else "LIBS_spec"
        spec_data = data.get(spec_key, {})
        axis = None
        for key in ["axis", "spectral_axis", "wavenumbers", "wavelengths", "wave"]:
            if key in spec_data:
                axis = np.asarray(spec_data[key], dtype=float).ravel()
                physical_axis = True
                break
        if axis is None:
            warnings.warn(
                f"{modality}: prepared_data.pkl 未发现光谱轴字段，将使用 0...{n_features-1} 的通道索引。"
            )
            axis = np.arange(n_features, dtype=float)
    if len(axis) != n_features:
        raise ValueError(f"{modality} 光谱轴长度 {len(axis)} 与特征数 {n_features} 不一致。")
    return axis, physical_axis


def calculate_q4_region_shap(shap_by_condition, y_test, axis, modality, out_csv):
    """汇总 Q4 样本的有符号和绝对 SHAP。"""
    q4_mask = np.asarray(y_test) >= np.percentile(y_test, 75)
    if modality == "ATR":
        regions = {
            "Amide_I_II_COOH_C=C": (1550, 1750),
            "Amide_III_CH_CO": (1250, 1400),
            "CO_CN": (975, 1250),
            "Carbonates": (1400, 1550),
            "Quartz": (650, 975),
            "Clay_OH": (3500, 4000),
        }
        axis_label = "wavenumber"
    else:
        regions = {
            "C": (247.5, 249.5), "O": (791.6, 801.6), "H": (649.7, 664.3),
            "N": (497.9, 504.3), "K": (762.6, 772.8), "Ca": (314.3, 320.5),
            "Mg": (517.5, 519.0), "Si": (249.5, 255.2), "Fe": (272.9, 275.8),
        }
        axis_label = "wavelength"

    rows = []
    for condition, shap_values in shap_by_condition.items():
        q4_values = np.asarray(shap_values)[q4_mask]
        for region, (low, high) in regions.items():
            mask = (axis >= low) & (axis <= high)
            if not np.any(mask):
                continue
            vals = q4_values[:, mask]
            rows.append({
                "condition": condition,
                "SOC_group": "Q4",
                "region": region,
                "axis_type": axis_label,
                "axis_min": low,
                "axis_max": high,
                "mean_signed_SHAP": float(np.mean(vals)),
                "mean_absolute_SHAP": float(np.mean(np.abs(vals))),
                "n_test_Q4": int(q4_mask.sum()),
                "n_channels": int(mask.sum()),
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        print(f"{modality}: 光谱轴未覆盖预设区间，跳过区域 SHAP 汇总。")


def run_shap_for_modality(modality, data, outdir, groups, args, device):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("缺少 shap 包，请运行：pip install shap") from exc

    spec_key = "ATR_spec" if modality == "ATR" else "LIBS_spec"
    X_train = as_matrix(data[spec_key]["X_train"], f"{spec_key}.X_train")
    X_test = as_matrix(data[spec_key]["X_test"], f"{spec_key}.X_test")
    y_train = as_vector(data["SOC"]["y_train"], "SOC.y_train")
    y_test = as_vector(data["SOC"]["y_test"], "SOC.y_test")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(f"{modality}: 光谱和 SOC 标签长度不一致。")
    if args.snv:
        X_train = snv_transform(X_train)
        X_test = snv_transform(X_test)

    modality_dir = outdir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    requested_background = args.background_size
    if requested_background is None:
        requested_background = 100 if modality == "ATR" else 50
    n_bg = min(requested_background, len(X_train))
    shap_nsamples = args.shap_nsamples
    if shap_nsamples is None:
        shap_nsamples = 200 if modality == "ATR" else 50
    bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
    background = torch.as_tensor(X_train[bg_idx], dtype=torch.float32).unsqueeze(1).to(device)

    q4_threshold = float(np.percentile(y_test, 75))
    q4_mask = y_test >= q4_threshold

    shap_by_condition = {}
    metrics_rows = []
    removal_results = {}

    for ratio in REMOVAL_RATIOS:
        label = "0% (Full)" if ratio == 0 else f"{int(ratio * 100)}% Removed"
        print(f"\n{'=' * 65}\n{modality}: {label}\n{'=' * 65}")

        n_remove = int(np.ceil(len(y_train) * ratio))
        order_desc = np.argsort(y_train, kind="mergesort")[::-1]
        remove_idx = order_desc[:n_remove] if n_remove > 0 else np.array([], dtype=int)
        keep_mask = np.ones(len(y_train), dtype=bool)
        keep_mask[remove_idx] = False
        keep_idx = np.where(keep_mask)[0]

        X_remaining = X_train[keep_idx]
        y_remaining = y_train[keep_idx]
        groups_remaining = groups[keep_idx] if groups is not None else None

        if len(y_remaining) < 10:
            raise ValueError(f"{modality} {label}: 去除后训练样本过少，无法训练模型。")

        fit_idx, val_idx = split_fit_validation(
            len(y_remaining), args.val_size, args.seed, groups_remaining
        )
        best_epoch, history = train_until_best_epoch(
            X_remaining[fit_idx], y_remaining[fit_idx],
            X_remaining[val_idx], y_remaining[val_idx],
            device=device, lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, seed=args.seed
        )

        model = train_full_data(
            X_remaining, y_remaining, best_epoch, device,
            lr=args.lr, batch_size=args.batch_size, seed=args.seed
        )
        y_pred = predict(model, X_test, device)
        metrics = calculate_metrics(y_test, y_pred)

        checkpoint = modality_dir / f"best_{modality.lower()}_cnn_{int(ratio * 100)}pct_removed.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_length": int(X_train.shape[1]),
            "modality": modality,
            "removal_ratio": float(ratio),
            "best_epoch": int(best_epoch),
            "snv": bool(args.snv),
        }, checkpoint)

        model.eval()
        explainer = shap.GradientExplainer(model, background)
        test_tensor = torch.as_tensor(X_test, dtype=torch.float32).unsqueeze(1)
        shap_chunks = []
        for start in range(0, len(X_test), args.shap_batch_size):
            batch = test_tensor[start:start + args.shap_batch_size].to(device)
            values = explainer.shap_values(batch, nsamples=shap_nsamples)
            shap_chunks.append(normalize_shap_shape(values, len(batch), X_train.shape[1]))
        shap_values = np.vstack(shap_chunks)

        expected_value = float(model(background).detach().mean().cpu().item())
        shap_by_condition[label] = shap_values

        np.savez_compressed(
            modality_dir / f"{modality.lower()}_shap_{int(ratio * 100)}pct_removed.npz",
            shap_values=shap_values,
            expected_value=expected_value,
            y_test=y_test,
            y_pred=y_pred,
            background_indices=bg_idx,
            q4_threshold=q4_threshold,
        )

        result = {
            "test": y_test,
            "test_pred": y_pred,
            "metrics": metrics,
            "shap_vals": shap_values,
            "expected_value": expected_value,
            "best_epoch": int(best_epoch),
            "snv": bool(args.snv),
            "background_size": int(n_bg),
            "shap_nsamples": int(shap_nsamples),
            "remaining_train_indices": keep_idx,
            "removed_train_indices": remove_idx,
            "history": history,
            "model_path": str(checkpoint),
        }
        removal_results[label] = result
        metrics_rows.append({
            "Modality": modality,
            "Condition": label,
            "Removal_ratio": ratio,
            "n_train_remaining": len(y_remaining),
            "n_removed": len(remove_idx),
            "best_epoch": best_epoch,
            **metrics,
            "Q4_RMSE": float(np.sqrt(np.mean((y_pred[q4_mask] - y_test[q4_mask]) ** 2))),
            "Q4_bias_pred_minus_measured": float(np.mean(y_pred[q4_mask] - y_test[q4_mask])),
        })

        pd.DataFrame(history).to_csv(
            modality_dir / f"training_history_{int(ratio * 100)}pct_removed.csv", index=False
        )
        del explainer, model, test_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shap_result = {
        "X_test": X_test,
        "y_test": y_test,
        **{
            name: {
                "shap_vals": item["shap_vals"],
                "expected_value": item["expected_value"],
                "test_pred": item["test_pred"],
                "metrics": item["metrics"],
            }
            for name, item in removal_results.items()
        },
    }
    prefix = modality.lower()
    with (modality_dir / f"{prefix}_shap_res.pkl").open("wb") as f:
        pickle.dump(shap_result, f)
    with (modality_dir / f"{prefix}_cnn_removal_performance.pkl").open("wb") as f:
        pickle.dump({"raw_data": removal_results}, f)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(modality_dir / f"{prefix}_removal_performance.csv", index=False)

    axis_file = args.atr_axis_file if modality == "ATR" else args.libs_axis_file
    axis, physical_axis = load_axis(data, modality, X_train.shape[1], axis_file)
    axis_label = (
        "Wavenumber (cm$^{-1}$)" if modality == "ATR"
        else "Wavelength (nm)"
    ) if physical_axis else "Spectral channel index"

    global_rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition, vals in shap_by_condition.items():
        mean_abs = np.mean(np.abs(vals), axis=0)
        mean_signed = np.mean(vals, axis=0)
        for j in range(len(axis)):
            global_rows.append({
                "condition": condition,
                "spectral_axis": axis[j],
                "mean_abs_SHAP": mean_abs[j],
                "mean_signed_SHAP": mean_signed[j],
            })
        ax.plot(axis, mean_abs, linewidth=1.25, label=condition)
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Mean absolute SHAP value")
    ax.set_title(f"{modality}: global SHAP attribution across high-SOC removal")
    ax.legend(frameon=False, fontsize=8)
    if modality == "ATR" and physical_axis:
        ax.invert_xaxis()
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(modality_dir / f"{prefix}_global_SHAP_by_removal.pdf", bbox_inches="tight")
    fig.savefig(modality_dir / f"{prefix}_global_SHAP_by_removal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(global_rows).to_csv(modality_dir / f"{prefix}_global_SHAP_by_removal.csv", index=False)

    if physical_axis:
        calculate_q4_region_shap(
            shap_by_condition, y_test, axis, modality,
            modality_dir / f"{prefix}_Q4_region_SHAP.csv"
        )
    else:
        print(f"{modality}: 未提供实际波数/波长轴，跳过化学/元素区域 SHAP 汇总。")

    q4_stack = []
    condition_labels = []
    for condition, vals in shap_by_condition.items():
        q4_vals = vals[q4_mask]
        q4_stack.append(q4_vals)
        condition_labels.extend([condition] * len(q4_vals))
    X_q4 = np.vstack(q4_stack)
    X_scaled = StandardScaler().fit_transform(X_q4)
    pca = PCA(n_components=2, random_state=args.seed)
    scores = pca.fit_transform(X_scaled)
    scores_df = pd.DataFrame({
        "PC1": scores[:, 0], "PC2": scores[:, 1], "Condition": condition_labels,
    })
    scores_df.to_csv(modality_dir / f"{prefix}_Q4_SHAP_joint_PCA_scores.csv", index=False)
    pd.DataFrame({
        "spectral_axis": axis,
        "PC1_loading": pca.components_[0],
        "PC2_loading": pca.components_[1],
    }).to_csv(modality_dir / f"{prefix}_Q4_SHAP_joint_PCA_loadings.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for condition in [("0% (Full)" if r == 0 else f"{int(r*100)}% Removed") for r in REMOVAL_RATIOS]:
        mask = scores_df["Condition"].to_numpy() == condition
        axes[0].scatter(scores[mask, 0], scores[mask, 1], s=12, alpha=0.55, label=condition)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("Q4 SHAP scores")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.2, linestyle=":")

    axes[1].plot(axis, pca.components_[0], label="PC1 loading")
    axes[1].plot(axis, pca.components_[1], label="PC2 loading")
    axes[1].set_xlabel(axis_label)
    axes[1].set_ylabel("PCA loading")
    axes[1].set_title("Joint PCA loadings")
    axes[1].legend(frameon=False, fontsize=8)
    if modality == "ATR" and physical_axis:
        axes[1].invert_xaxis()
    axes[1].grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(modality_dir / f"{prefix}_Q4_SHAP_joint_PCA.pdf", bbox_inches="tight")
    fig.savefig(modality_dir / f"{prefix}_Q4_SHAP_joint_PCA.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return metrics_rows


def main():
    parser = argparse.ArgumentParser(description="High-SOC removal and SHAP analysis for CNN models.")
    parser.add_argument("--data", default="prepared_data.pkl")
    parser.add_argument("--outdir", default="outputs/SHAP_high_SOC_removal")
    parser.add_argument("--modalities", nargs="+", default=["ATR", "LIBS"], choices=["ATR", "LIBS"])
    parser.add_argument("--groups-npy", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--background-size", type=int, default=None,
                        help="SHAP background 数量；默认 ATR=100、LIBS=50（对应 Notebook）。")
    parser.add_argument("--snv", action="store_true",
                        help="对所有内部训练/测试光谱执行 SNV；需与基础 CNN 的 SNV 设置一致。")
    parser.add_argument("--shap-nsamples", type=int, default=None,
                        help="GradientExplainer nsamples；默认 ATR=200、LIBS=50（对应 Notebook）。")
    parser.add_argument("--shap-batch-size", type=int, default=32)
    parser.add_argument("--atr-axis-file", default=None)
    parser.add_argument("--libs-axis-file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_total_data(args.data)
    y_train = as_vector(data["SOC"]["y_train"], "SOC.y_train")
    groups = get_groups(args.groups_npy, len(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for modality in args.modalities:
        all_rows.extend(run_shap_for_modality(modality, data, outdir, groups, args, device))

    pd.DataFrame(all_rows).to_csv(outdir / "all_modalities_removal_metrics.csv", index=False)
    print(f"\nFinished. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
