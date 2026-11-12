# SOC Spectral Modeling Scripts (ATR-FTIR / LIBS)

This directory contains four standalone Python scripts for "Uneven calibration support drives prediction-range compression in soil organic carbon spectroscopy: Correction through targeted spectral generation".

## 1. File Overview

| File | Description |
|---|---|
| `01_missingness_analysis.py` | Analyzes the relationship between local sparsity in the CNN latent space and prediction error. |
| `02_SHAP_high_SOC_removal.py` | Sequentially removes high-SOC training samples (0–50%), performs GradientSHAP analysis, and conducts joint PCA of Q4 SHAP values. |
| `03_CNN_model.py` | Trains and evaluates baseline 1D-CNN models for ATR-FTIR and LIBS, and exports latent features. |
| `04_CAVE_CNN_model.py` | Performs SNV-KNN spectral-domain filtering, PCA-CVAE-based targeted generation of high-SOC spectra, and CAVE-CNN training. |
| `requirements.txt` | Lists the required Python packages. |

## 2. Input Data Requirements

The scripts use the prepared dataset `../data/prepared_data.pkl`. This file should be a Python dictionary containing at least the following fields:

```python
total_data = {
    "SOC": {
        "y_train": ...,   # SOC values for the internal training set (1D array)
        "y_test": ...,    # SOC values for the independent test set (1D array)
        "y_ext": ...,     # SOC values for the external dataset (1D array; required for CAVE-CNN)
    },
    "ATR_spec": {
        "X_train": ...,   # shape = (n_train, n_ATR_features)
        "X_test": ...,    # shape = (n_test, n_ATR_features)
        "X_ext": ...,     # shape = (n_ext, n_ATR_features; required for CAVE-CNN)
        # Optional: axis / spectral_axis / wavenumbers
    },
    "LIBS_spec": {
        "X_train": ...,
        "X_test": ...,
        "X_ext": ...,
        # Optional: axis / spectral_axis / wavelengths
    }
}
```

`X_train`, `X_test`, and `X_ext` must be two-dimensional numerical matrices, whereas the SOC labels must be one-dimensional numerical arrays. Within each spectral modality, the spectra and SOC labels must correspond to one another sample by sample. The scripts do not silently impute missing or infinite values (NaN/Inf) in the spectra or SOC labels.

## 3. Recommended Execution Order

### Step 1. Train the Baseline CNN

```bash
python 03_CNN_model.py --data prepared_data.pkl --outdir outputs/CNN
```

By default, the script trains models for both ATR-FTIR and LIBS. The outputs include:

- `outputs/CNN/ATR/` and `outputs/CNN/LIBS/`: model weights, test-set predictions, evaluation metrics, training histories, and latent features.
- `outputs/CNN/atr_cnn.pkl` and `outputs/CNN/libs_cnn.pkl`: prediction-result files compatible with the original notebook workflow.

To apply SNV preprocessing to the baseline CNN as well, run:

```bash
python 03_CNN_model.py --data prepared_data.pkl --outdir outputs/CNN_SNV --snv
```

### Step 2. Analyze Data-Coverage Gaps / Local Sparsity in the Latent Space

```bash
python 01_missingness_analysis.py --cnn-dir outputs/CNN --outdir outputs/missingness
```

This script reads `latent_train`, `latent_test`, and the test-set predictions exported in Step 1. It calculates the mean Mahalanobis K-nearest-neighbor (KNN) distance from each test sample to the training samples, and then analyzes the relationship between this distance and the absolute prediction error.

### Step 3. SHAP Analysis and High-SOC Training-Sample Removal Experiment

```bash
python 02_SHAP_high_SOC_removal.py --data prepared_data.pkl --outdir outputs/SHAP
```

For each spectral modality, six CNN models are trained: one using the complete training set (0% removal), and five using training sets from which the samples with the highest SOC values have been sequentially removed at proportions of 10%, 20%, 30%, 40%, and 50%.

The outputs include:

- CNN model weights and test-set predictions for each removal condition;
- SHAP-value matrices for each condition;
- Mean absolute SHAP (`mean(|SHAP|)`) profiles across the full test set;
- Region-level SHAP summaries for Q4 samples in the test set (requires the actual wavenumber or wavelength axis);
- Joint PCA scores and loadings for Q4 SHAP values across all removal conditions;
- Overall and Q4-stratified prediction performance.

If SNV preprocessing was used for the baseline CNN, the same preprocessing must also be used in the SHAP analysis:

```bash
python 02_SHAP_high_SOC_removal.py --data prepared_data.pkl --outdir outputs/SHAP_SNV --snv
```

If `prepared_data.pkl` does not contain the actual wavenumber or wavelength axis, an axis file can be supplied as a `.npy` file or a single-column CSV file:

```bash
python 02_SHAP_high_SOC_removal.py \
  --data prepared_data.pkl \
  --outdir outputs/SHAP \
  --atr-axis-file atr_axis.csv \
  --libs-axis-file libs_axis.csv
```

Without the actual spectral axis, the script will still calculate and export SHAP results using spectral-channel indices. However, it will not perform region-level aggregation for ATR-FTIR functional-group bands or LIBS elemental emission lines.

### Step 4. CAVE-CNN

```bash
python 04_CAVE_CNN_model.py --data prepared_data.pkl --outdir outputs/CAVE
```

The default configuration is `layer2_soc5_bin10`:

- CVAE: two hidden layers, `hidden_dim=64`, and 500 training epochs;
- PCA: 99% cumulative explained variance;
- High-SOC threshold: 5 g/kg;
- SOC binning: 10 equal-width bins;
- Target number of samples per high-SOC bin: 60;
- External spectral KNN filtering: `k=5`;
- CNN: Adam optimizer with `lr=0.001`, a maximum of 1,000 epochs, and `patience=100`.

To run the full parameter grid used in the notebook (layer 1/2 × threshold 0/5/10/15/20 × bins 10/15/20/25/30), run:

```bash
python 04_CAVE_CNN_model.py --data prepared_data.pkl --outdir outputs/CAVE_grid --grid-search
```

The grid search can be computationally intensive. The script selects the parameter configuration based on the internal validation RMSE; the independent test set is not used for parameter selection.

## 5. Installing Dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration with PyTorch, install the build compatible with the CUDA version on your machine. Select the appropriate installation command from the official PyTorch installation page.

---

**Path note:** The dataset is described above as `../data/prepared_data.pkl`, whereas the example commands use `prepared_data.pkl`. If the file is actually stored in the `../data/` directory relative to your working directory, replace `--data prepared_data.pkl` in the commands with `--data ../data/prepared_data.pkl`.
