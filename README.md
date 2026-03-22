
# A Comprehensive Benchmark of Time-Series Foundation Models for Blood Glucose Forecasting

This repository contains the official source code for the paper *"A comprehensive benchmark of time-series foundation models for blood glucose forecasting"*. We provide a framework to evaluate ten Time-Series Foundation Models (TSFMs) across zero-shot, fine-tuned, and high glycemic variability scenarios. This codebase facilitates research on applying TSFMs to address the cold-start problem in personalized medical time-series forecasting.

## Evaluation Overview

All models are evaluated under three scenarios: **zero-shot forecasting**, **full fine-tuning**, and **generalization evaluation** on a high glycemic variability (GV) dataset. The table below summarizes model participation:

| Model | Zero-Shot Forecasting | Full Fine-Tuning | Generalization Evaluation |
|---|:---:|:---:|:---:|
| **Time-series foundation models** | | | |
| Chronos-2 | ✅ | ✅ | ✅ |
| Chronos-bolt-base | ✅ | ✅ | ✅ |
| FlowState-r1 | ✅ | NA | NA |
| Moirai-1.1-R-base | ✅ | ✅ | ✅ |
| Sundial-base | ✅ | NA | NA |
| TabPFN-v2 | ✅ | NA | NA |
| TimeMoE-200m | ✅ | ✅ | ✅ |
| TimesFM-2.5 | ✅ | NA | NA |
| Tirex | ✅ | NA | NA |
| ToTo-base-1.0 | ✅ | NA | NA |
| **Deep learning models** | | | |
| LSTM | NA | ✅ | ✅ |
| N-BEATS | NA | ✅ | ✅ |
| N-HiTS | NA | ✅ | ✅ |
| PatchTST | NA | ✅ | ✅ |
| TFT | NA | ✅ | ✅ |
| WaveNet | NA | ✅ | ✅ |
| **Automated statistical models** | | | |
| Auto ARIMA | NA | ✅ | NA |
| Auto ETS | NA | ✅ | NA |

✅ = applicable; NA = not applicable due to training, API, or protocol constraints.

## Repository Structure

```
tsfm-bgf/
├── TSFMs/          # Time-series foundation models (10 models)
├── Tsline/         # Deep learning & statistical baseline models (8 models)
└── README.md
```

### [`TSFMs/`](TSFMs/)

Contains evaluation and fine-tuning scripts for 10 TSFMs. Each model is organized into `zero-shot/` and `fine-tuned/` (if applicable) subdirectories. Models retain their original inference pipelines — no unified wrapper is imposed. See [`TSFMs/README.md`](TSFMs/README.md) for upstream repo versions and per-model usage.

### [`Tsline/`](Tsline/)

Contains 8 baseline models (6 deep learning + 2 statistical) built on a unified wrapper architecture. All models share a consistent `BaseModelWrapper` interface for training and evaluation. Model hyperparameters can be found in the `BASE_CONFIG` dictionary within each wrapper script. See [`Tsline/README.md`](Tsline/README.md) for details.

## Getting Started

### Data Preparation

This benchmark uses publicly available continuous glucose monitoring (CGM) datasets. To reproduce the experiments:

1. Obtain the raw CGM data from the public dataset sources referenced in the paper.
2. Segment each subject's recordings into continuous CSV files named `{subjectID}_{fileID}.csv`. Each CSV must contain a `GlucoseValue` column.
3. Organize the files into the following directory structure:

```
glucose_data/
├── op_split2/          # OP dataset
│   ├── Train/
│   ├── Val/
│   └── Test/
├── re_split2/          # RE dataset
│   ├── Train/
│   ├── Val/
│   └── Test/
└── lg_split2/          # LG (high GV) dataset
    ├── Train/
    ├── Val/
    └── Test/
```

### Running Evaluations

Navigate to the target model's directory and run the corresponding evaluation script directly:

```bash
# Example: TSFMs zero-shot evaluation
python TSFMs/Sundial/zero-shot/sundial_eval.py

# Example: Tsline baseline training & evaluation
python Tsline/PatchTST/patchtst_wrapper.py
```

Refer to each subdirectory's `README.md` for model-specific instructions, CLI arguments, and fine-tuning workflows.

