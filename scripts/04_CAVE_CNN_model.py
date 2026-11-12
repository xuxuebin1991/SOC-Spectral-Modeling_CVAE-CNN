#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_CAVE_CNN_model.py
基于 CVAE 的条件光谱生成与 CAVE-CNN 土壤 SOC 回归。

核心流程（依据 cnn.ipynb 的 CAVE 部分整理）：
1. 读取内部训练集、独立测试集和外部光谱数据；
2. 内部训练子集与外部样本之间执行 SNV + KNN 光谱域筛选；
3. 对筛选后的训练池执行 SNV、PCA（累计解释方差 99%）和标准化；
4. 以 SOC 为条件变量训练 CVAE（1 层或 2 层隐藏层）；
5. 将 SOC 划分为等宽区间，仅对阈值以上的高 SOC 区间补充虚拟样本；
6. 将生成光谱与内部训练光谱合并，训练 SoilSpectralCNN；
7. 独立测试集只用于最终评价，不参与筛选、生成、训练或参数选择。

运行：
python 04_CAVE_CNN_model.py --data ../data/prepared_data.pkl --outdir outputs/CAVE
python 04_CAVE_CNN_model.py --data ../data/prepared_data.pkl --outdir outputs/CAVE --grid-search

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
import torch.nn.functional as F
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_total_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path.resolve()}")
    with path.open("rb") as f:
        data = pickle.load(f)

    required = [
        ("SOC", "y_train"), ("SOC", "y_test"), ("SOC", "y_ext"),
        ("ATR_spec", "X_train"), ("ATR_spec", "X_test"), ("ATR_spec", "X_ext"),
        ("LIBS_spec", "X_train"), ("LIBS_spec", "X_test"), ("LIBS_spec", "X_ext"),
    ]
    missing = [f"{a}.{b}" for a, b in required
               if a not in data or b not in data[a]]
    if missing:
        raise KeyError("prepared_data.pkl 缺少字段：" + ", ".join(missing))
    return data


def as_matrix(x, name):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} 必须为二维矩阵，当前 shape={x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} 中含有 NaN/Inf，请先检查光谱数据。")
    return x


def as_vector(y, name):
    y = np.asarray(y, dtype=np.float32).ravel()
    if not np.isfinite(y).all():
        raise ValueError(f"{name} 中含有 NaN/Inf。")
    return y


def snv_transform(X):
    """逐样本 SNV 变换。"""
    X = np.asarray(X, dtype=np.float32)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def get_groups(groups_npy, n_train):
    if not groups_npy:
        return None
    groups = np.asarray(np.load(groups_npy, allow_pickle=True)).ravel()
    if len(groups) != n_train:
        raise ValueError(
            f"--groups-npy 长度 {len(groups)} 与 y_train 长度 {n_train} 不一致。"
        )
    return groups


def split_train_validation(n, val_size, seed, groups=None):
    idx = np.arange(n)
    fit_idx, val_idx = train_test_split(
        idx, test_size=val_size, random_state=seed, shuffle=True
    )
    return fit_idx, val_idx, "train_test_split"


class SoilSpectralCNN(nn.Module):
    """与 cnn.ipynb CELL 0 相同的 1D-CNN 架构。"""

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


def regression_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p))

    my, mp = float(np.mean(y)), float(np.mean(p))
    vy, vp = float(np.var(y)), float(np.var(p))
    cov = float(np.mean((y - my) * (p - mp)))
    denom = vy + vp + (my - mp) ** 2
    ccc = float(2 * cov / denom) if denom > 0 else np.nan

    if len(y) > 1 and np.std(y) > 0:
        slope_mp, intercept_mp, _, _, _ = linregress(y, p)
        slope_res, intercept_res, r_res, p_res, _ = linregress(y, p - y)
    else:
        slope_mp = intercept_mp = slope_res = intercept_res = r_res = p_res = np.nan

    return {
        "R2": r2, "RMSE": rmse, "MAE": mae, "CCC": ccc,
        "slope_measured_predicted": float(slope_mp),
        "intercept_measured_predicted": float(intercept_mp),
        "slope_residual_SOC": float(slope_res),
        "intercept_residual_SOC": float(intercept_res),
        "residual_r": float(r_res), "residual_p": float(p_res),
    }


class ConditionalVAE1(nn.Module):
    """一层隐藏层 CVAE（对应原 Notebook layer=1）。"""

    def __init__(self, pc_dim, hidden_dim=64):
        super().__init__()
        self.enc_hidden = nn.Linear(pc_dim + 1, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, pc_dim)
        self.fc_logvar = nn.Linear(hidden_dim, pc_dim)
        self.dec_hidden = nn.Linear(pc_dim + 1, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, pc_dim)

    def forward(self, x_pcs, x_soc):
        enc_input = torch.cat([x_pcs, x_soc], dim=1)
        h = torch.relu(self.enc_hidden(enc_input))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, x_soc)
        return recon, mu, logvar

    def decode(self, z, x_soc):
        dec_input = torch.cat([z, x_soc], dim=1)
        h = torch.relu(self.dec_hidden(dec_input))
        return self.dec_out(h)


class ConditionalVAE2(nn.Module):
    """两层隐藏层 CVAE（对应原 Notebook layer=2）。"""

    def __init__(self, pc_dim, hidden_dim=64):
        super().__init__()
        self.enc_layer1 = nn.Linear(pc_dim + 1, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, pc_dim)
        self.fc_logvar = nn.Linear(hidden_dim, pc_dim)
        self.dec_layer1 = nn.Linear(pc_dim + 1, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, pc_dim)

    def forward(self, x_pcs, x_soc):
        enc_input = torch.cat([x_pcs, x_soc], dim=1)
        h1 = torch.relu(self.enc_layer1(enc_input))
        h2 = torch.relu(self.enc_layer2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, x_soc)
        return recon, mu, logvar

    def decode(self, z, x_soc):
        dec_input = torch.cat([z, x_soc], dim=1)
        h1 = torch.relu(self.dec_layer1(dec_input))
        h2 = torch.relu(self.dec_layer2(h1))
        return self.dec_out(h2)


class CVAE_Pipeline:
    def __init__(
        self, n_components=0.99, layer=2, hidden_dim=64, epochs=500,
        target_per_bin=60, high_soc_threshold=5, random_state=42,
        use_distance_filter=True, filter_mode="ratio", ext_ratio=1.5,
        threshold_multiplier=1.5, n_neighbors=5, num_bins=10, beta_kl=0.1,
    ):
        self.n_components = n_components
        self.layer = int(layer)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.target_per_bin = int(target_per_bin)
        self.high_soc_threshold = float(high_soc_threshold)
        self.random_state = int(random_state)
        self.use_distance_filter = bool(use_distance_filter)
        self.filter_mode = filter_mode
        self.ext_ratio = float(ext_ratio)
        self.threshold_multiplier = float(threshold_multiplier)
        self.n_neighbors = int(n_neighbors)
        self.num_bins = int(num_bins)
        self.beta_kl = float(beta_kl)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pca = None
        self.scaler_pcs = None
        self.scaler_soc = None
        self.cvae_model = None
        self.pc_dim = None
        self.filter_info = {}

    def _filter_external_by_distance(self, X_train, X_ext, y_ext):
        """在 SNV 光谱空间按 KNN 距离筛选外部样本。"""
        if len(X_ext) == 0:
            return X_ext, y_ext, {"n_external_before": 0, "n_external_after": 0}

        X_train_snv = snv_transform(X_train)
        X_ext_snv = snv_transform(X_ext)
        k = min(self.n_neighbors, len(X_train))

        nn_ext = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn_ext.fit(X_train_snv)
        ext_dist, _ = nn_ext.kneighbors(X_ext_snv)
        mean_ext_dist = ext_dist.mean(axis=1)

        if self.filter_mode == "ratio":
            target_n = min(len(X_ext), int(len(X_train) * self.ext_ratio))
            order = np.argsort(mean_ext_dist)
            keep_idx = order[:target_n]
            cutoff = float(mean_ext_dist[keep_idx].max()) if len(keep_idx) else np.nan

        elif self.filter_mode == "threshold":
            k_train = min(k + 1, len(X_train))
            nn_train = NearestNeighbors(n_neighbors=k_train, metric="euclidean")
            nn_train.fit(X_train_snv)
            train_dist, _ = nn_train.kneighbors(X_train_snv)
            if train_dist.shape[1] > 1:
                train_dist = train_dist[:, 1:]
            mean_train_dist = train_dist.mean(axis=1)
            base_distance = float(np.mean(mean_train_dist))
            cutoff = base_distance * self.threshold_multiplier
            keep_idx = np.where(mean_ext_dist <= cutoff)[0]
        else:
            raise ValueError("filter_mode 必须为 'ratio' 或 'threshold'。")

        X_filtered = X_ext[keep_idx]
        y_filtered = y_ext[keep_idx]
        self.filter_info = {
            "filter_mode": self.filter_mode,
            "n_neighbors": int(k),
            "n_external_before": int(len(X_ext)),
            "n_external_after": int(len(X_filtered)),
            "n_external_removed": int(len(X_ext) - len(X_filtered)),
            "external_mean_knn_distance": float(np.mean(mean_ext_dist)) if len(mean_ext_dist) else np.nan,
            "distance_cutoff": float(cutoff) if np.isfinite(cutoff) else np.nan,
            "ext_ratio": self.ext_ratio if self.filter_mode == "ratio" else np.nan,
            "threshold_multiplier": self.threshold_multiplier if self.filter_mode == "threshold" else np.nan,
        }
        print(
            f"External-domain filtering ({self.filter_mode}): "
            f"{len(X_ext)} -> {len(X_filtered)} samples"
        )
        return X_filtered, y_filtered, self.filter_info

    def fit_and_generate(self, X_train, y_train, X_ext, y_ext):
        X_train = as_matrix(X_train, "CAVE X_train")
        X_ext = as_matrix(X_ext, "CAVE X_ext")
        y_train = as_vector(y_train, "CAVE y_train")
        y_ext = as_vector(y_ext, "CAVE y_ext")

        if len(X_train) != len(y_train) or len(X_ext) != len(y_ext):
            raise ValueError("CAVE 光谱与 SOC 标签样本数不一致。")
        if X_train.shape[1] != X_ext.shape[1]:
            raise ValueError("内部与外部光谱的特征维度不一致。")
        if len(X_train) < 5:
            raise ValueError("CVAE 训练样本数过少。")

        if self.use_distance_filter:
            X_ext_f, y_ext_f, filter_info = self._filter_external_by_distance(
                X_train, X_ext, y_ext
            )
        else:
            X_ext_f, y_ext_f = X_ext, y_ext
            filter_info = {
                "filter_mode": "none",
                "n_external_before": int(len(X_ext)),
                "n_external_after": int(len(X_ext)),
                "n_external_removed": 0,
            }

        # CVAE 在 SNV 光谱域中学习，合成光谱也因此处于 SNV 域。
        X_pool_snv = snv_transform(np.vstack([X_train, X_ext_f]))
        y_pool = np.concatenate([y_train, y_ext_f])

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pcs = self.pca.fit_transform(X_pool_snv)
        self.pc_dim = int(pcs.shape[1])
        self.scaler_pcs = StandardScaler()
        pcs_scaled = self.scaler_pcs.fit_transform(pcs)
        self.scaler_soc = StandardScaler()
        soc_scaled = self.scaler_soc.fit_transform(y_pool.reshape(-1, 1)).astype(np.float32)

        if self.layer == 1:
            self.cvae_model = ConditionalVAE1(self.pc_dim, self.hidden_dim).to(self.device)
        elif self.layer == 2:
            self.cvae_model = ConditionalVAE2(self.pc_dim, self.hidden_dim).to(self.device)
        else:
            raise ValueError("layer 只能为 1 或 2。")

        X_tensor = torch.as_tensor(pcs_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(soc_scaled, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.cvae_model.parameters(), lr=1e-3)

        self.cvae_model.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            recon, mu, logvar = self.cvae_model(X_tensor, y_tensor)
            recon_loss = F.mse_loss(recon, X_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X_tensor.size(0)
            loss = recon_loss + self.beta_kl * kl_loss
            loss.backward()
            optimizer.step()

            if epoch == 1 or epoch % 100 == 0 or epoch == self.epochs:
                print(
                    f"  CVAE epoch {epoch:4d}/{self.epochs} | "
                    f"total={loss.item():.5f} | recon={recon_loss.item():.5f} | KL={kl_loss.item():.5f}"
                )

        bins = np.linspace(float(y_train.min()), float(y_train.max()), self.num_bins + 1)
        counts, _ = np.histogram(y_train, bins=bins)
        target_soc = []
        bin_records = []

        for i in range(self.num_bins):
            lower, upper = float(bins[i]), float(bins[i + 1])
            if lower < self.high_soc_threshold:
                continue
            need = max(0, self.target_per_bin - int(counts[i]))
            if need > 0:
                target_soc.extend(
                    np.random.default_rng(self.random_state + i).uniform(lower, upper, need).tolist()
                )
            bin_records.append({
                "bin": i + 1,
                "soc_lower": lower,
                "soc_upper": upper,
                "original_count": int(counts[i]),
                "generated_count": int(need),
            })

        if len(target_soc) == 0:
            warnings.warn(
                "当前阈值和分箱设置未产生虚拟样本；请检查 high_soc_threshold、num_bins、"
                "target_per_bin 及训练集 SOC 范围。",
                RuntimeWarning,
            )
            synthetic_spectra = np.empty((0, X_train.shape[1]), dtype=np.float32)
            y_synthetic = np.empty((0,), dtype=np.float32)
        else:
            self.cvae_model.eval()
            target_soc = np.asarray(target_soc, dtype=np.float32)
            target_soc_scaled = self.scaler_soc.transform(target_soc.reshape(-1, 1)).astype(np.float32)
            z = torch.randn(len(target_soc), self.pc_dim, device=self.device)
            target_tensor = torch.as_tensor(target_soc_scaled, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                synthetic_pcs_scaled = self.cvae_model.decode(z, target_tensor).cpu().numpy()

            synthetic_pcs = self.scaler_pcs.inverse_transform(synthetic_pcs_scaled)
            synthetic_spectra = self.pca.inverse_transform(synthetic_pcs).astype(np.float32)
            y_synthetic = target_soc

        metadata = {
            "layer": self.layer,
            "hidden_dim": self.hidden_dim,
            "cvae_epochs": self.epochs,
            "pc_dim": self.pc_dim,
            "pca_explained_variance": float(self.pca.explained_variance_ratio_.sum()),
            "high_soc_threshold": self.high_soc_threshold,
            "num_bins": self.num_bins,
            "target_per_bin": self.target_per_bin,
            "n_synthetic": int(len(y_synthetic)),
            "bin_records": bin_records,
            "filter_info": filter_info,
            "cnn_input_preprocessing": "SNV",
        }
        return synthetic_spectra, y_synthetic, metadata


def make_loader(X, y, batch_size, shuffle, seed):
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def train_cnn_early_stopping(
    X_fit, y_fit, X_val, y_val, device,
    lr=1e-3, batch_size=32, epochs=1000, patience=100, seed=42
):
    set_seed(seed)
    model = SoilSpectralCNN(X_fit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    fit_loader = make_loader(X_fit, y_fit, batch_size, True, seed)
    val_loader = make_loader(X_val, y_val, batch_size, False, seed)

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

        tr_loss, va_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if epoch == 1 or epoch % 10 == 0:
            print(f"  CNN epoch {epoch:4d}/{epochs} | train={tr_loss:.5f} | val={va_loss:.5f}")

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  CNN early stopping; selected epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("CNN early stopping 未获得有效模型。")
    model.load_state_dict(best_state)
    return model, best_epoch, history, best_val


def train_cnn_fixed_epochs(X, y, epochs, device, lr=1e-3, batch_size=32, seed=42):
    """以 validation 选择的 epoch 数，在完整内部训练集上重训。"""
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
            print(f"  Final CNN refit {epoch:4d}/{epochs} | loss={np.mean(losses):.5f}")
    return model


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    pred_list = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
        pred_list.append(model(xb).reshape(-1).cpu().numpy())
    return np.concatenate(pred_list)


def evaluate_config(
    X_core, y_core, X_val, y_val, X_ext, y_ext,
    config, args, device, seed
):
    """"""
    pipeline = CVAE_Pipeline(
        n_components=0.99,
        layer=config["layer"],
        hidden_dim=64,
        epochs=args.cvae_epochs,
        target_per_bin=args.target_per_bin,
        high_soc_threshold=config["threshold"],
        random_state=seed,
        use_distance_filter=not args.no_distance_filter,
        filter_mode=args.filter_mode,
        ext_ratio=args.ext_ratio,
        threshold_multiplier=args.threshold_multiplier,
        n_neighbors=args.n_neighbors,
        num_bins=config["bins"],
        beta_kl=0.1,
    )
    X_syn, y_syn, cave_meta = pipeline.fit_and_generate(X_core, y_core, X_ext, y_ext)

    X_core_cnn = snv_transform(X_core)
    X_val_cnn = snv_transform(X_val)
    if len(X_syn):
        X_fit_aug = np.vstack([X_core_cnn, X_syn])
        y_fit_aug = np.concatenate([y_core, y_syn])
    else:
        X_fit_aug, y_fit_aug = X_core_cnn, y_core

    model, best_epoch, history, best_val_loss = train_cnn_early_stopping(
        X_fit_aug, y_fit_aug, X_val_cnn, y_val,
        device=device, lr=args.cnn_lr, batch_size=args.batch_size,
        epochs=args.cnn_epochs, patience=args.patience, seed=seed
    )
    val_pred = predict(model, X_val_cnn, device)
    val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))

    return {
        "config": config,
        "val_RMSE": val_rmse,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "val_pred": val_pred,
        "history": history,
        "pipeline": pipeline,
        "X_syn": X_syn,
        "y_syn": y_syn,
        "cave_meta": cave_meta,
    }


def run_one_modality(modality, data, groups, args, outdir, device):
    spec_key = "ATR_spec" if modality == "ATR" else "LIBS_spec"
    X_train = as_matrix(data[spec_key]["X_train"], f"{spec_key}.X_train")
    X_test = as_matrix(data[spec_key]["X_test"], f"{spec_key}.X_test")
    X_ext = as_matrix(data[spec_key]["X_ext"], f"{spec_key}.X_ext")
    y_train = as_vector(data["SOC"]["y_train"], "SOC.y_train")
    y_test = as_vector(data["SOC"]["y_test"], "SOC.y_test")
    y_ext = as_vector(data["SOC"]["y_ext"], "SOC.y_ext")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test) or len(X_ext) != len(y_ext):
        raise ValueError(f"{modality}: 光谱矩阵与 SOC 标签样本数不一致。")
    if X_train.shape[1] != X_test.shape[1] or X_train.shape[1] != X_ext.shape[1]:
        raise ValueError(f"{modality}: train/test/external 光谱特征维度不一致。")

    modality_dir = outdir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    fit_idx, val_idx, split_method = split_train_validation(
        len(y_train), args.val_size, args.seed, groups
    )
    X_core, y_core = X_train[fit_idx], y_train[fit_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    if args.grid_search:
        configs = [
            {"layer": layer, "threshold": threshold, "bins": bins}
            for layer in [1, 2]
            for threshold in [0, 5, 10, 15, 20]
            for bins in [10, 15, 20, 25, 30]
        ]
    else:
        configs = [{
            "layer": args.layer,
            "threshold": args.soc_threshold,
            "bins": args.num_bins,
        }]

    search_rows = []
    best = None
    for i, config in enumerate(configs, start=1):
        print(
            f"\n{modality} CAVE configuration {i}/{len(configs)}: "
            f"layer={config['layer']}, threshold={config['threshold']}, bins={config['bins']}"
        )
        set_seed(args.seed)
        evaluated = evaluate_config(
            X_core, y_core, X_val, y_val, X_ext, y_ext,
            config, args, device, args.seed
        )
        search_rows.append({
            **config,
            "validation_RMSE": evaluated["val_RMSE"],
            "best_epoch": evaluated["best_epoch"],
            "n_synthetic_validation": len(evaluated["y_syn"]),
            "n_external_retained": evaluated["cave_meta"]["filter_info"].get("n_external_after", len(X_ext)),
            "validation_split": split_method,
        })
        if best is None or evaluated["val_RMSE"] < best["val_RMSE"]:
            best = evaluated
        del evaluated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    search_df = pd.DataFrame(search_rows).sort_values("validation_RMSE")
    search_df.to_csv(modality_dir / f"{modality.lower()}_CAVE_parameter_screening.csv", index=False)
    print(f"\n{modality}: selected config by internal validation = {best['config']}")
    print(f"{modality}: validation RMSE = {best['val_RMSE']:.4f}")

    config = best["config"]
    final_pipeline = CVAE_Pipeline(
        n_components=0.99,
        layer=config["layer"],
        hidden_dim=64,
        epochs=args.cvae_epochs,
        target_per_bin=args.target_per_bin,
        high_soc_threshold=config["threshold"],
        random_state=args.seed,
        use_distance_filter=not args.no_distance_filter,
        filter_mode=args.filter_mode,
        ext_ratio=args.ext_ratio,
        threshold_multiplier=args.threshold_multiplier,
        n_neighbors=args.n_neighbors,
        num_bins=config["bins"],
        beta_kl=0.1,
    )
    X_syn, y_syn, cave_meta = final_pipeline.fit_and_generate(
        X_train, y_train, X_ext, y_ext
    )

    X_train_snv = snv_transform(X_train)
    X_test_snv = snv_transform(X_test)
    if len(X_syn):
        X_train_cave = np.vstack([X_train_snv, X_syn])
        y_train_cave = np.concatenate([y_train, y_syn])
    else:
        X_train_cave = X_train_snv
        y_train_cave = y_train

    best_epoch = int(best["best_epoch"])
    final_model = train_cnn_fixed_epochs(
        X_train_cave, y_train_cave, best_epoch, device,
        lr=args.cnn_lr, batch_size=args.batch_size, seed=args.seed
    )
    y_pred = predict(final_model, X_test_snv, device)
    metrics = regression_metrics(y_test, y_pred)

    model_path = modality_dir / f"best_{modality.lower()}_cave_cnn.pth"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "input_length": int(X_train.shape[1]),
        "modality": modality,
        "model_name": "CAVE-CNN",
        "cnn_input_preprocessing": "SNV",
        "cvae_config": config,
        "best_epoch": best_epoch,
    }, model_path)

    pd.DataFrame({
        "SOC_measured": y_test,
        "SOC_predicted": y_pred,
        "residual_pred_minus_measured": y_pred - y_test,
        "absolute_error": np.abs(y_pred - y_test),
    }).to_csv(modality_dir / f"{modality.lower()}_CAVE_CNN_test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(modality_dir / f"{modality.lower()}_CAVE_CNN_metrics.csv", index=False)

    if len(y_syn):
        pd.DataFrame({
            "synthetic_SOC": y_syn,
            "synthetic_sample_index": np.arange(len(y_syn)),
        }).to_csv(modality_dir / f"{modality.lower()}_CAVE_synthetic_SOC.csv", index=False)
        np.save(modality_dir / f"{modality.lower()}_CAVE_synthetic_spectra_snv.npy", X_syn)

    pd.DataFrame(cave_meta["bin_records"]).to_csv(
        modality_dir / f"{modality.lower()}_CAVE_bin_generation.csv", index=False
    )
    pd.DataFrame([{
        **cave_meta,
        "bin_records": str(cave_meta["bin_records"]),
        "filter_info": str(cave_meta["filter_info"]),
        "validation_RMSE": best["val_RMSE"],
        "validation_split": split_method,
        "best_epoch": best_epoch,
    }]).to_csv(modality_dir / f"{modality.lower()}_CAVE_metadata.csv", index=False)

    result = {
        "test": y_test,
        "test_pred": y_pred,
        "test_performance": metrics,
        "CAVE_set": {
            "X_train_cave": X_syn,
            "y_train_cave": y_syn,
            "X_train_augmented_for_CNN": X_train_cave,
            "y_train_augmented_for_CNN": y_train_cave,
        },
        "validation": {
            "validation_RMSE": best["val_RMSE"],
            "best_epoch": best_epoch,
            "split_method": split_method,
        },
        "config": config,
        "metadata": cave_meta,
        "model_path": str(model_path),
    }
    with (modality_dir / f"{modality.lower()}_CAVE_CNN_result.pkl").open("wb") as f:
        pickle.dump(result, f)

    print(f"\n{modality} CAVE-CNN test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return result, search_rows


def main():
    parser = argparse.ArgumentParser(description="CVAE-based CAVE-CNN for soil SOC prediction.")
    parser.add_argument("--data", default="prepared_data.pkl")
    parser.add_argument("--outdir", default="outputs/CAVE_CNN")
    parser.add_argument("--modalities", nargs="+", default=["ATR", "LIBS"], choices=["ATR", "LIBS"])
    parser.add_argument("--groups-npy", default=None)
    parser.add_argument("--layer", type=int, default=2, choices=[1, 2])
    parser.add_argument("--soc-threshold", type=float, default=5)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--target-per-bin", type=int, default=60)
    parser.add_argument("--cvae-epochs", type=int, default=500)
    parser.add_argument("--cnn-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--filter-mode", choices=["ratio", "threshold"], default="ratio")
    parser.add_argument("--ext-ratio", type=float, default=1.5)
    parser.add_argument("--threshold-multiplier", type=float, default=1.5)
    parser.add_argument("--no-distance-filter", action="store_true")
    parser.add_argument("--grid-search", action="store_true",
                        help="运行 Notebook 中的 2×5×5 参数网格；计算量较大。")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_bins < 2 or args.target_per_bin < 1:
        raise ValueError("num-bins 至少为 2，target-per-bin 至少为 1。")
    set_seed(args.seed)
    data = load_total_data(args.data)
    y_train = as_vector(data["SOC"]["y_train"], "SOC.y_train")
    groups = get_groups(args.groups_npy, len(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    all_search_rows = []

    for modality in args.modalities:
        print(f"\n{'#' * 75}\nRunning CAVE-CNN for {modality}\n{'#' * 75}")
        result, search_rows = run_one_modality(
            modality, data, groups, args, outdir, device
        )
        config = result["config"]
        result_key = f"layer{config['layer']}_soc{config['threshold']}_bin{config['bins']}"
        all_results.setdefault(result_key, {})[modality] = {
            "CAVE_set": result["CAVE_set"],
            "test_pred": {
                "test": result["test"],
                "test_pred": result["test_pred"],
            },
            "test_performance": result["test_performance"],
            "validation": result["validation"],
            "config": result["config"],
            "metadata": result["metadata"],
            "model_path": result["model_path"],
        }
        all_search_rows.extend([{"Modality": modality, **row} for row in search_rows])

    with (outdir / "cave_cnn_results.pkl").open("wb") as f:
        pickle.dump(all_results, f)
    pd.DataFrame(all_search_rows).to_csv(outdir / "CAVE_CNN_parameter_screening_all.csv", index=False)

    print(f"\nFinished. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
