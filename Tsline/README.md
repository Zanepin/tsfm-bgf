# Tsline Benchmark Models

This directory contains the deep learning and traditional statistical baseline models for continuous glucose monitoring (CGM) forecasting. The codebase uses a unified, object-oriented Wrapper architecture for consistency, clarity, and ease of reproducibility.

## Experiment Scenarios & Supported Models

The benchmark suite includes 8 models categorized into Deep Learning and Traditional Statistical methods. 

| Model | Zero-Shot Forecasting | Full Fine-Tuning | Generalization Evaluation | Directory |
|---|:---:|:---:|:---:|---|
| **Deep Learning Models** | | | | |
| PatchTST | NA | ✅ | ✅ | `PatchTST/` |
| TFT | NA | ✅ | ✅ | `TFT/` |
| LSTM | NA | ✅ | ✅ | `LSTM/` |
| N-BEATS | NA | ✅ | ✅ | `Nbeats/` |
| N-HiTS | NA | ✅ | ✅ | `Nhits/` |
| WaveNet | NA | ✅ | ✅ | `Wavenet/` |
| **Statistical Models** | | | | |
| Auto ARIMA | NA | ✅ | NA | `ARIMA/` |
| Auto ETS | NA | ✅ | NA | `ETS/` |

> **Note:** The `lg` dataset introduces a high glycemic variability (GV) forecasting scenario where test subjects have no overlapping historical data in the training set. Because traditional statistical models mathematically require intra-subject history to fit their smoothing parameters, they are incompatible with this GV split. They are evaluated exclusively on `op` and `re` datasets.

## Environment & Dependencies

This benchmark suite relies on the following key packages to implement the baseline models and evaluation logic:
- `torch` & `pytorch-lightning`: Core deep learning engines natively powering models like LSTM and PatchTST.
- `darts`: Used for unified time series processing and architectures such as TFT, N-BEATS, N-HiTS, AutoARIMA, and AutoETS.
- `gluonts`: Provides advanced probabilistic forecasting estimators used by WaveNet and PatchTST.
- `pandas` & `numpy`: Standard data manipulation and tensor processing.
- `scikit-learn`: Standard metric calculations (MAE, RMSE, R2).

## Architecture & Configuration

All models inherit from a shared interface `BaseModelWrapper` located in `core/base_wrapper.py`. This ensures that every model uses the exact same data splitting paradigm and the same medical evaluation metrics.

### Configuration & Hyperparameters

The optimal hyperparameters and specific configuration settings chosen for each baseline model are transparently encapsulated directly within their respective wrapper scripts. You can find the exact `BASE_CONFIG` dictionary (including learning rates, hidden sizes, patch lengths, etc.) at the bottom of each `[model]_wrapper.py` file under the `if __name__ == "__main__":` block.

## Directory Structure

```text
Tsline/
├── README.md
├── core/
│   └── base_wrapper.py        # Unified training/evaluation interface
├── utils/
│   └── metrics.py             # Shared Clarke Error Grid and standard metrics
├── PatchTST/                  # Model-specific directories with their wrappers
├── TFT/
├── LSTM/
├── Nbeats/
├── Nhits/
├── Wavenet/
├── ARIMA/
└── ETS/
```

## Running Evaluations

To run any benchmark on the datasets (`op`, `re`, or `lg`), simply execute its corresponding wrapper script. The scripts are built to automatically loop over the defined datasets and prediction horizons, train the model, evaluate it, and export a unified CSV report.

**Example: Running PatchTST**
```bash
cd Tsline
python PatchTST/patchtst_wrapper.py
```

## Evaluation Metrics

Each script will output a summarized CSV file (e.g., `patchtst_experiment_summary_all.csv`) containing the following identical columns evaluated independently per subject:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² (%)** (Coefficient of Determination)
- **CEA Zone A+B (%)** — Clarke Error Grid Analysis clinically acceptable zones
- **CEA Zone C+D+E (%)** — Clarke Error Grid Analysis clinically dangerous zones
- **Median Runtime (s)** — Inference time per subject
