#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_CNN_model.py
基于 prepared_data.pkl 的 ATR-FTIR / LIBS 一维 CNN-SOC 回归。


运行示例：
python 03_CNN_model.py --data prepared_data.pkl --outdir outputs/CNN
python 03_CNN_model.py --data prepared_data.pkl --outdir outputs/CNN --snv
"""

from __future__ import annotations

import argparse
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SoilSpectralCNN(nn.Module):
    """与 cnn.ipynb CELL 0 一致的 1D-CNN 结构。"""

    def __init__(self, input_length: int):
        super().__init__()
        if input_length < 16:
            raise ValueError(f"光谱长度必须至少为 16，当前为 {input_length}")

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
            dummy = torch.zeros(1, 1, input_length)
            flattened_size = int(self.conv_block(dummy).numel())

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(200, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"模型输入应为二维或三维张量，当前 shape={tuple(x.shape)}")
        x = self.conv_block(x)
        x = x.reshape(x.size(0), -1)
        return self.fc_block(x)


def load_total_data(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path.resolve()}")
    with path.open("rb") as f:
        data = pickle.load(f)

    required = [
        ("SOC", "y_train"), ("SOC", "y_test"),
        ("ATR_spec", "X_train"), ("ATR_spec", "X_test"),
        ("LIBS_spec", "X_train"), ("LIBS_spec", "X_test"),
    ]
    missing = [f"{a}.{b}" for a, b in required
               if a not in data or b not in data[a]]
    if missing:
        raise KeyError("prepared_data.pkl 缺少字段：" + ", ".join(missing))
    return data


def as_2d_float(X, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} 必须是二维矩阵 (n_samples, n_features)，当前 shape={X.shape}")
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"{name} 含有 {n_bad} 个 NaN/Inf。请先检查光谱预处理与缺失值，"
            "本脚本不会静默填补光谱缺失值。"
        )
    return X


def as_1d_float(y, name: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).ravel()
    if not np.isfinite(y).all():
        raise ValueError(f"{name} 含有 NaN/Inf，请先处理 SOC 标签。")
    return y


def snv_transform(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def get_groups(groups_npy: str | None, n_train: int):
    if not groups_npy:
        return None
    groups = np.asarray(np.load(groups_npy, allow_pickle=True)).ravel()
    if len(groups) != n_train:
        raise ValueError(
            f"--groups-npy 长度为 {len(groups)}，但训练集样本数为 {n_train}。"
        )
    return groups


def split_train_validation(n: int, val_size: float, seed: int, groups=None):
    idx = np.arange(n)
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        fit_idx, val_idx = next(splitter.split(idx, groups=groups))
        return fit_idx, val_idx, "GroupShuffleSplit"
    fit_idx, val_idx = train_test_split(
        idx, test_size=val_size, random_state=seed, shuffle=True
    )
    warnings.warn(
        "未提供 site/group ID：validation 使用样本级随机划分。"
        "若同一采样点存在配对深度样本，建议提供 --groups-npy，避免同一 site 跨入 fit/validation。",
        RuntimeWarning,
    )
    return fit_idx, val_idx, "train_test_split"


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if len(y) != len(p):
        raise ValueError("y_true 与 y_pred 长度不一致。")

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p))

    mean_y, mean_p = float(np.mean(y)), float(np.mean(p))
    var_y, var_p = float(np.var(y)), float(np.var(p))
    cov_yp = float(np.mean((y - mean_y) * (p - mean_p)))
    denom = var_y + var_p + (mean_y - mean_p) ** 2
    ccc = float(2 * cov_yp / denom) if denom > 0 else np.nan

    if len(y) >= 2 and np.std(y) > 0:
        slope, intercept, r_res, p_res, _ = linregress(y, p - y)
        slope_mp, intercept_mp, _, _, _ = linregress(y, p)
    else:
        slope = intercept = r_res = p_res = slope_mp = intercept_mp = np.nan

    return {
        "R2": r2, "RMSE": rmse, "MAE": mae, "CCC": ccc,
        "slope_measured_predicted": float(slope_mp),
        "intercept_measured_predicted": float(intercept_mp),
        "slope_residual_SOC": float(slope),
        "intercept_residual_SOC": float(intercept),
        "residual_r": float(r_res), "residual_p": float(p_res),
    }


def make_loader(X, y, batch_size=32, shuffle=False, seed=42):
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_early_stopping(
    X_fit, y_fit, X_val, y_val, device,
    lr=1e-4, batch_size=32, epochs=500, patience=100, seed=42, verbose=True
):
    """只用内部 validation 选择 epoch；不触碰独立 test set。"""
    set_seed(seed)
    model = SoilSpectralCNN(X_fit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    fit_loader = make_loader(X_fit, y_fit, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False, seed=seed)

    best_val = float("inf")
    best_epoch = 1
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).reshape(-1)
            loss = criterion(pred, yb.reshape(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).reshape(-1)
                val_losses.append(float(criterion(pred, yb.reshape(-1)).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"Epoch {epoch:4d}/{epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, val loss={best_val:.5f}")
                break

    if best_state is None:
        raise RuntimeError("模型训练失败：没有得到有效的 validation checkpoint。")
    return best_epoch, best_state, history


def train_fixed_epochs(X, y, epochs, device, lr=1e-4, batch_size=32, seed=42, verbose=True):
    """确定 epoch 后，使用完整训练集重新拟合最终模型。"""
    set_seed(seed)
    model = SoilSpectralCNN(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = make_loader(X, y, batch_size, shuffle=True, seed=seed)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).reshape(-1)
            loss = criterion(pred, yb.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            print(f"Final refit epoch {epoch:4d}/{epochs} | loss={np.mean(losses):.5f}")
    return model


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    outputs = []
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    for start in range(0, len(X_tensor), batch_size):
        xb = X_tensor[start:start + batch_size].to(device)
        outputs.append(model(xb).reshape(-1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


@torch.no_grad()
def extract_latent(model, X, device, batch_size=128):
    """提取 conv_block 输出并展平，作为 CNN latent representation。"""
    model.eval()
    outputs = []
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    for start in range(0, len(X_tensor), batch_size):
        xb = X_tensor[start:start + batch_size].to(device)
        if xb.ndim == 2:
            xb = xb.unsqueeze(1)
        feat = model.conv_block(xb)
        outputs.append(feat.reshape(feat.size(0), -1).cpu().numpy())
    return np.vstack(outputs)


def main():
    parser = argparse.ArgumentParser(description="Train baseline 1D-CNN models for soil SOC prediction.")
    parser.add_argument("--data", default="../data/prepared_data.pkl")
    parser.add_argument("--outdir", default="../outputs/CNN")
    parser.add_argument("--modalities", nargs="+", default=["ATR", "LIBS"], choices=["ATR", "LIBS"])
    parser.add_argument("--groups-npy", default=None,
                        help="可选：训练集 site/group ID 的 .npy 文件，长度必须等于 y_train。")
    parser.add_argument("--snv", action="store_true",
                        help="对训练/测试光谱统一执行 SNV；默认不执行，以复现原 CNN Notebook。")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_total_data(args.data)
    y_train = as_1d_float(data["SOC"]["y_train"], "SOC.y_train")
    y_test = as_1d_float(data["SOC"]["y_test"], "SOC.y_test")
    groups = get_groups(args.groups_npy, len(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for modality in args.modalities:
        print(f"\n{'=' * 70}\nTraining {modality} CNN\n{'=' * 70}")
        spec_key = "ATR_spec" if modality == "ATR" else "LIBS_spec"
        X_train = as_2d_float(data[spec_key]["X_train"], f"{spec_key}.X_train")
        X_test = as_2d_float(data[spec_key]["X_test"], f"{spec_key}.X_test")
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError(f"{modality}: 光谱与 SOC 标签样本数不一致。")

        if args.snv:
            X_train = snv_transform(X_train)
            X_test = snv_transform(X_test)

        fit_idx, val_idx, split_method = split_train_validation(
            len(y_train), args.val_size, args.seed, groups
        )
        best_epoch, _, history = train_early_stopping(
            X_train[fit_idx], y_train[fit_idx],
            X_train[val_idx], y_train[val_idx],
            device=device, lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, seed=args.seed
        )
        print(f"{modality}: validation split={split_method}; selected epoch={best_epoch}")

        model = train_fixed_epochs(
            X_train, y_train, best_epoch, device,
            lr=args.lr, batch_size=args.batch_size, seed=args.seed
        )
        y_pred = predict(model, X_test, device)
        metrics = regression_metrics(y_test, y_pred)
        print(f"{modality} test metrics: {metrics}")

        latent_train = extract_latent(model, X_train, device)
        latent_test = extract_latent(model, X_test, device)

        model_dir = outdir / modality
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"best_{modality.lower()}_cnn.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_length": int(X_train.shape[1]),
            "modality": modality,
            "snv": bool(args.snv),
            "best_epoch": int(best_epoch),
            "model_name": "SoilSpectralCNN",
        }, model_path)

        result = {
            "test": y_test,
            "test_pred": y_pred,
            "metrics": metrics,
            "latent_train": latent_train,
            "latent_test": latent_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_test_pred": y_pred,
            "best_epoch": int(best_epoch),
            "validation_split": split_method,
            "snv": bool(args.snv),
            "history": history,
            "model_path": str(model_path),
        }
        pkl_path = model_dir / f"{modality.lower()}_cnn.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(result, f)

        latent_result = {
            "latent_train": latent_train,
            "y_train": y_train,
            "y_train_pred": predict(model, X_train, device),
            "latent_test": latent_test,
            "y_test": y_test,
            "y_test_pred": y_pred,
        }
        with (model_dir / f"{modality.lower()}_cnn_latents.pkl").open("wb") as f:
            pickle.dump(latent_result, f)

        pd.DataFrame({
            "SOC_measured": y_test,
            "SOC_predicted": y_pred,
            "residual_pred_minus_measured": y_pred - y_test,
            "absolute_error": np.abs(y_pred - y_test),
        }).to_csv(model_dir / f"{modality.lower()}_cnn_test_predictions.csv", index=False)

        pd.DataFrame([metrics]).to_csv(model_dir / f"{modality.lower()}_cnn_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(model_dir / f"{modality.lower()}_cnn_training_history.csv", index=False)

        with (outdir / f"{modality.lower()}_cnn.pkl").open("wb") as f:
            pickle.dump({"test": y_test, "test_pred": y_pred, "metrics": metrics}, f)

        all_metrics.append({"Modality": modality, **metrics, "best_epoch": best_epoch})

    pd.DataFrame(all_metrics).to_csv(outdir / "CNN_model_comparison.csv", index=False)
    print(f"\nFinished. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
