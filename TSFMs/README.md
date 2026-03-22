# TSFMs Benchmark Models

This directory contains the evaluation and fine-tuning scripts for 10 time-series foundation models (TSFMs) applied to continuous glucose monitoring (CGM) forecasting. Unlike the baseline models in `Tsline/`, each TSFM retains its own unique inference pipeline to ensure native representation.

## Experiment Scenarios & Supported Models

Each foundation model participates in one or more of the following scenarios natively or via fine-tuning:

| Model | Zero-Shot Forecasting | Full Fine-Tuning | Generalization (lg) | Directory |
|---|:---:|:---:|:---:|---|
| Chronos-2 | ✅ | ✅ | ✅ | `Chronos/` |
| Chronos-bolt-base | ✅ | ✅ | ✅ | `Chronos/` |
| FlowState-r1 | ✅ | — | — | `FlowState/` |
| Moirai-1.1-R-base | ✅ | ✅ | ✅ | `Moirai/` |
| Sundial-base | ✅ | — | — | `Sundial/` |
| TabPFN-v2 | ✅ | — | — | `TabPFN/` |
| TimeMoE-200m | ✅ | ✅ | ✅ | `TimeMoE/` |
| TimesFM-2.5 | ✅ | — | — | `TimesFM/` |
| Tirex | ✅ | — | — | `Tirex/` |
| ToTo-base-1.0 | ✅ | — | — | `ToTo/` |

> **Note:** The `lg` dataset is a custom high glycemic variability (GV) generalization set where test subjects have no overlap with the training set. Only models with fine-tuning capability participate in this scenario.

## Environment & Dependencies

This suite evaluates diverse foundation models, which natively rely on the following shared ecosystems to function:
- `torch`: Core deep learning engine powering the inference and native training pipelines.
- `transformers` & `huggingface-hub`: Essential for loading models published on Hugging Face (e.g., Chronos, TimeMoE, ToTo).
- `pandas` & `numpy`: Standard data manipulation and local dataset preprocessing.
- `scikit-learn`: Standard array and regression metric calculations (MAE, RMSE, R2).

*(Note: Certain models require their exclusive upstream frameworks such as `uni2ts` for Moirai or Google's `timesfm` package. Please refer directly to the cloned upstream repositories for their specific `requirements.txt` / `pyproject.toml` setups).*

## Architecture & Configuration

Unlike the `Tsline/` baseline counterparts, we do **not** enforce a rigid, unified wrapper class here. Each model is executed using its native loading/inference paradigm to guarantee reproducibility and optimal hardware utilization. 

However, evaluation logic has been unified:
- **`utils/clarke_error_grid.py`**: A shared module utilized collectively by all evaluation scripts to guarantee the standardized extraction of the Clarke Error Grid Analysis zones and core statistical metrics.

## Directory Structure

```text
TSFMs/
├── README.md
├── utils/                            # Shared utilities and evaluation metrics
│   └── clarke_error_grid.py 
├── Chronos/                          # Chronos-2 & Chronos-bolt (Amazon)
│   ├── zero-shot/
│   │   ├── chronos2_eval.py
│   │   └── chronos_bolt_eval.py
│   └── fine-tuned/
│       ├── chronos-bolt/             # Bolt fine-tuning (uses chronos-forecasting v1)
│       └── chronos-2/                # Chronos-2 fine-tuning (uses chronos-forecasting v2)
├── FlowState/                        # FlowState-r1 (IBM Granite)
├── Moirai/                           # Moirai 1.1 & 2.0 (Salesforce)
│   ├── zero-shot/
│   └── fine-tuned/
│       ├── process_data.py           # Data preparation
│       ├── run_finetune.sh           # Parameterized training script
│       ├── run_eval.sh               # Parameterized evaluation script
│       └── eval_clarke.py            # Evaluation core logic
├── Sundial/                          # Sundial-base-128m (THU)
├── TabPFN/                           # TabPFN-v2
├── TimeMoE/                          # TimeMoE-200M (Maple)
├── TimesFM/                          # TimesFM-2.5 (Google)
├── Tirex/                            # Tirex (NX-AI)
└── ToTo/                             # ToTo-base-1.0
```

## Upstream Repositories

Several models require their upstream source code to be cloned locally for fine-tuning or zero-shot inference. The table below records the exact versions used in our experiments:

| Model | Repository | Commit Hash |
|---|---|---|
| Chronos-bolt (fine-tuning) | `amazon-science/chronos-forecasting` | `6a9c8da` (v1.5.2) |
| Chronos-2 (fine-tuning) | `amazon-science/chronos-forecasting` | `1f099eb` |
| FlowState | `ibm-granite/granite-tsfm` | `bdc36c3` |
| Moirai (zero-shot) | `SalesforceAIResearch/uni2ts` | `b4ad1f4` |
| Moirai (fine-tuned) | `SalesforceAIResearch/uni2ts` | `8d2a08d` |
| TimeMoE | `Time-MoE/Time-MoE` | `67460eb` |
| TimesFM | `google-research/timesfm` | `6bd8044` |
| Tirex | `NX-AI/tirex` | `295ced7` |

## Running Evaluations

### Zero-Shot Models
```bash
# Example: Sundial
python TSFMs/Sundial/zero-shot/sundial_eval.py
```

### Fine-Tuned Models
Fine-tuned models typically require cloning the upstream repo, preparing the dataset, scaling/training via shell scripts, and finally evaluating checkpoints.

**Example: Moirai 1.1**
```bash
cd TSFMs/Moirai/fine-tuned
# Train OP dataset for 6-step horizon
./run_finetune.sh op 6
# Evaluate checkpoint
./run_eval.sh path/to/ckpt.ckpt glucose_op_test_full.csv 6 137.5710 53.2954
```

## Evaluation Metrics

All evaluation scripts natively calculate or utilize `utils.clarke_error_grid` to output identical metrics per test subject:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² (%)** (Coefficient of Determination)
- **CEA Zone A+B (%)** — Clarke Error Grid Analysis clinically acceptable zones
- **CEA Zone C+D+E (%)** — Clarke Error Grid Analysis clinically dangerous zones
- **Median Runtime (s)** — Inference time per subject
