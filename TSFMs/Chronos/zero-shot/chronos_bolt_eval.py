import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid, calculate_clarke_metrics


# ==========================================
# Chronos-Bolt Imports
# ==========================================
try:
    from chronos import ChronosBoltPipeline
except ImportError:
    # If import fails, try fallback or notify the user
    try:
        from chronos import BaseChronosPipeline as ChronosBoltPipeline
    except ImportError:
        raise ImportError(
            "Please ensure the latest version of Chronos is installed: pip install 'chronos[pytorch] @ git+https://github.com/amazon-science/chronos-forecasting.git'")

# Ignore warnings
warnings.filterwarnings('ignore')


# ==========================================
# PART 1: CEA Standard Analysis Module
# ==========================================
# ==========================================
# PART 2: Configuration and Data Loading
# ==========================================

DATASETS = {
    "Re": "/mnt/d/glucose_data/internal/re_split2/test",
    "Op": "/mnt/d/glucose_data/internal/op_split2/test"
}

# Experimental parameters
INPUT_LEN = 48  # Context length
STRIDE = 12  # Sliding stride
HORIZONS = [6, 12]  # Prediction horizons
BATCH_SIZE = 64  # Bolt is relatively fast, so batch size can be larger
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Using Bolt Base model
CHRONOS_MODEL_PATH = "amazon/chronos-bolt-base"


def load_and_group_by_subject(data_path):
    """Load CSVs and group by Subject ID"""
    subject_map = {}
    files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"Loading {len(files)} files from {data_path}...")

    for f in tqdm(files, desc="Parsing Files"):
        filename = os.path.basename(f)
        try:
            # Logic for extracting Subject ID based on dataset naming convention
            if 're_split' in data_path or 're_' in filename.lower():
                sub_id = filename.rsplit('_', 1)[0]
            else:
                sub_id = filename.split('_')[0]

            df = pd.read_csv(f)
            target_col = 'GlucoseValue' if 'GlucoseValue' in df.columns else 'target'
            if target_col not in df.columns: continue

            vals = df[target_col].values.astype(np.float32)

            if len(vals) <= INPUT_LEN: continue

            if sub_id not in subject_map:
                subject_map[sub_id] = []
            subject_map[sub_id].append(vals)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return subject_map


def generate_windows(sessions, input_len, output_len, stride):
    """Generate sliding window samples"""
    X_list, Y_list = [], []
    for seq in sessions:
        seq_len = len(seq)
        if seq_len <= input_len + output_len: continue
        for i in range(0, seq_len - input_len - output_len + 1, stride):
            x = seq[i: i + input_len]
            y = seq[i + input_len: i + input_len + output_len]
            X_list.append(x)
            Y_list.append(y)
    return np.array(X_list), np.array(Y_list)


# ==========================================
# PART 3: Chronos-Bolt Inference Wrapper
# ==========================================

def run_chronos_bolt_inference_batched(pipeline, X_np, horizon, batch_size=64):
    """
    Perform batch inference for Chronos-Bolt
    X_np: (N_samples, INPUT_LEN)
    """
    n_samples = len(X_np)
    all_preds = []

    # Convert to Tensor; Bolt recommends direct Tensor input for speed
    # Note: We don't manually move to GPU here; it's handled internally by the pipeline device mapping
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    # Get the index of the 0.5 (median) quantile in the quantiles list
    # Bolt default quantiles = [0.1, 0.2, ..., 0.9]
    try:
        model_quantiles = pipeline.quantiles
        if 0.5 in model_quantiles:
            median_idx = model_quantiles.index(0.5)
        else:
            # If an exact 0.5 quantile match is not found, take the middle position (Fallback)
            median_idx = len(model_quantiles) // 2
            print(f"Warning: Exact 0.5 quantile not found in {model_quantiles}. Using index {median_idx}.")
    except Exception:
        # If attribute retrieval fails, default to the 4th index (e.g., for 0.1, 0.2, 0.3, 0.4, 0.5)
        median_idx = 4

    for i in range(0, n_samples, batch_size):
        # 1. Prepare Batch
        batch_context = X_tensor[i: i + batch_size]
        current_bs = len(batch_context)

        # 2. Predict
        try:
            # Based on source code: predict(self, inputs, prediction_length, limit_prediction_length)
            # Note: We do not pass quantile_levels here because it returns all pre-trained quantiles
            forecast = pipeline.predict(
                inputs=batch_context,
                prediction_length=horizon,
                limit_prediction_length=False
            )

            # 3. Parse Results
            # Source shows forecast is a Tensor: (batch_size, num_quantiles, prediction_length)

            # Extract median layer -> (Batch, Horizon)
            batch_preds = forecast[:, median_idx, :].cpu().numpy()

            all_preds.append(batch_preds)

        except Exception as e:
            print(f"Batch inference failed at index {i}: {e}")
            import traceback
            traceback.print_exc()
            # Fill with NaN on failure
            all_preds.append(np.full((current_bs, horizon), np.nan))

    return np.concatenate(all_preds, axis=0)


# ==========================================
# PART 4: Main Execution Flow
# ==========================================

def main():
    print(f"Running Chronos-Bolt Evaluation on {DEVICE}")
    print(f"Model: {CHRONOS_MODEL_PATH}")

    # 1. Load Chronos-Bolt Pipeline
    try:
        # Using torch_dtype is standard for HuggingFace AutoConfig setups
        pipeline = ChronosBoltPipeline.from_pretrained(
            CHRONOS_MODEL_PATH,
            device_map=DEVICE,
            dtype=torch.bfloat16,
        )
        print("Chronos-Bolt pipeline loaded successfully.")
    except Exception as e:
        print(f"Failed to load Chronos-Bolt: {e}")
        return

    final_results = []

    # 2. Iterate through datasets
    for ds_name, ds_path in DATASETS.items():
        print(f"\n##########################################")
        print(f" PROCESSING DATASET: {ds_name}")
        print(f"##########################################")

        subjects_data = load_and_group_by_subject(ds_path)
        subject_ids = list(subjects_data.keys())

        if not subject_ids:
            print(f"No data found in {ds_path}, skipping.")
            continue

        # 3. Iterate through prediction horizons
        for horizon in HORIZONS:
            horizon_min = horizon * 5

            metrics_storage = {
                "MAE": [], "RMSE": [], "R2": [],
                "CEA_AB": [], "CEA_CDE": [], "Runtime": []
            }

            # 4. Iterate through subjects
            for sub_id in tqdm(subject_ids, desc=f"Evaluating {ds_name} (H={horizon_min}m)"):
                sessions = subjects_data[sub_id]

                # Generate samples
                X_np, Y_true = generate_windows(sessions, INPUT_LEN, horizon, STRIDE)
                if len(X_np) == 0: continue

                # Inference
                t_start = time.perf_counter()
                try:
                    Y_pred = run_chronos_bolt_inference_batched(
                        pipeline,
                        X_np,
                        horizon,
                        batch_size=BATCH_SIZE
                    )
                except Exception as e:
                    print(f"Inference error on {sub_id}: {e}")
                    continue
                t_end = time.perf_counter()

                # Calculate metrics
                flat_true = Y_true.flatten()
                flat_pred = Y_pred.flatten()

                # Handle potential NaNs
                if np.isnan(flat_pred).any():
                    flat_pred = np.nan_to_num(flat_pred, nan=np.nanmean(flat_true))

                mae = mean_absolute_error(flat_true, flat_pred)
                rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
                r2 = r2_score(flat_true, flat_pred) * 100
                cea_res = calculate_clarke_metrics(flat_true, flat_pred)

                metrics_storage["MAE"].append(mae)
                metrics_storage["RMSE"].append(rmse)
                metrics_storage["R2"].append(r2)
                metrics_storage["CEA_AB"].append(cea_res['AB_percentage'])
                metrics_storage["CEA_CDE"].append(cea_res['CDE_percentage'])
                metrics_storage["Runtime"].append(t_end - t_start)

            # Summarize results
            def fmt(val_list):
                if not val_list: return "N/A"
                return f"{np.mean(val_list):.2f} ± {np.std(val_list):.2f}"

            median_runtime = np.median(metrics_storage['Runtime']) if metrics_storage['Runtime'] else 0.0

            summary_row = {
                "Dataset": ds_name,
                "Horizon (min)": horizon_min,
                "MAE": fmt(metrics_storage["MAE"]),
                "RMSE": fmt(metrics_storage["RMSE"]),
                "R2 (%)": fmt(metrics_storage["R2"]),
                "CEA Zone A+B (%)": fmt(metrics_storage["CEA_AB"]),
                "CEA Zone C+D+E (%)": fmt(metrics_storage["CEA_CDE"]),
                "Media Runtime (sec)": f"{median_runtime:.2f}"
            }
            final_results.append(summary_row)

    # 5. Output results
    if final_results:
        df_res = pd.DataFrame(final_results)
        cols = ["Dataset", "Horizon (min)", "MAE", "RMSE", "R2 (%)", "CEA Zone A+B (%)", "CEA Zone C+D+E (%)",
                "Media Runtime (sec)"]
        df_res = df_res[cols]

        print("\n\n================ CHRONOS-BOLT ZERO-SHOT RESULTS ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        df_res.to_csv("chronos_bolt_glucose_evaluation.csv", index=False)
        print("\nResults saved to chronos_bolt_glucose_evaluation.csv")


if __name__ == "__main__":
    main()