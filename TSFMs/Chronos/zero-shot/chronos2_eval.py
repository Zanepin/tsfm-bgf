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
# Chronos Imports
# ==========================================
try:
    from chronos import Chronos2Pipeline
except ImportError:
    raise ImportError(
        "Please ensure Chronos is installed: pip install 'chronos[pytorch] @ git+https://github.com/amazon-science/chronos-forecasting.git'")

# Ignore warnings
warnings.filterwarnings('ignore')


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
BATCH_SIZE = 128  # Chronos Batch Size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHRONOS_MODEL_PATH = "amazon/chronos-2"  # or "amazon/chronos-2"


def load_and_group_by_subject(data_path):
    """Load CSV files and group them by Subject ID"""
    subject_map = {}
    files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"Loading {len(files)} files from {data_path}...")

    for f in tqdm(files, desc="Parsing Files"):
        filename = os.path.basename(f)
        try:
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
# PART 3: Chronos-2 Inference Wrapper
# ==========================================

def run_chronos2_inference_batched(pipeline, X_np, horizon, batch_size=32):
    """
    Perform batch inference for Chronos-2
    """
    n_samples = len(X_np)
    all_preds = []

    # Pre-generate a generic time index
    dummy_timestamps = pd.date_range(start="2020-01-01 00:00:00", periods=INPUT_LEN, freq="5min")

    for i in range(0, n_samples, batch_size):
        # 1. Prepare batch data
        batch_x = X_np[i: i + batch_size]  # (B, 48)
        current_bs = len(batch_x)

        # 2. Construct Long-format DataFrame
        timestamps_col = np.tile(dummy_timestamps, current_bs)
        ids_col = np.repeat([f"w_{k}" for k in range(current_bs)], INPUT_LEN)
        targets_col = batch_x.flatten()

        batch_df = pd.DataFrame({
            "id": ids_col,
            "timestamp": timestamps_col,
            "target": targets_col
        })

        # 3. Predict
        try:
            forecast_df = pipeline.predict_df(
                batch_df,
                prediction_length=horizon,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )

            # Determine quantile column name (usually string '0.5' rather than float 0.5)
            val_col = '0.5'
            if val_col not in forecast_df.columns:
                # Defensive programming: if not '0.5', try finding 0.5 or take the last column
                if 0.5 in forecast_df.columns:
                    val_col = 0.5
                else:
                    # Assume there is only one prediction column, take it directly
                    cols = [c for c in forecast_df.columns if c not in ['id', 'timestamp']]
                    if cols: val_col = cols[0]

            # 4. Parse results
            # Pivot table: Index=id, Columns=Timestamp (Sequence), Values=Predicted
            pivot_df = forecast_df.pivot(index="id", columns="timestamp", values=val_col)

            # Reorder indices to maintain original batch order
            ordered_ids = [f"w_{k}" for k in range(current_bs)]
            sorted_preds = pivot_df.reindex(ordered_ids).values  # (B, Horizon)

            all_preds.append(sorted_preds)

        except Exception as e:
            print(f"Batch inference failed: {e}")
            # Fill with NaN if batch inference fails
            all_preds.append(np.full((current_bs, horizon), np.nan))

    return np.concatenate(all_preds, axis=0)


# ==========================================
# PART 4: Main Execution Flow
# ==========================================

def main():
    print(f"Running Chronos-2 Evaluation on {DEVICE}")
    print(f"Model: {CHRONOS_MODEL_PATH}")

    # 1. Load Chronos-2 Pipeline
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL_PATH,
            device_map=DEVICE,
            dtype=torch.bfloat16,
        )
        print("Chronos pipeline loaded successfully.")
    except Exception as e:
        print(f"Failed to load Chronos: {e}")
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
                    Y_pred = run_chronos2_inference_batched(
                        pipeline,
                        X_np,
                        horizon,
                        batch_size=BATCH_SIZE
                    )
                except Exception as e:
                    print(f"Inference error on {sub_id}: {e}")
                    # import traceback
                    # traceback.print_exc()
                    continue
                t_end = time.perf_counter()

                # Calculate metrics
                flat_true = Y_true.flatten()
                flat_pred = Y_pred.flatten()

                # Handle NaNs
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

    # 5. Output
    if final_results:
        df_res = pd.DataFrame(final_results)
        cols = ["Dataset", "Horizon (min)", "MAE", "RMSE", "R2 (%)", "CEA Zone A+B (%)", "CEA Zone C+D+E (%)",
                "Media Runtime (sec)"]
        df_res = df_res[cols]

        print("\n\n================ CHRONOS-2 ZERO-SHOT RESULTS ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        df_res.to_csv("chronos2_glucose_evaluation.csv", index=False)
        print("\nResults saved to chronos2_glucose_evaluation.csv")


if __name__ == "__main__":
    main()