import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from tirex import load_model
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid, calculate_clarke_metrics

# Ignore warnings
warnings.filterwarnings('ignore')

# ==========================================
# PART 2: Configuration and Data Loading
# ==========================================

DATASETS = {
    "Op": "/mnt/d/glucose_data/internal/op_split2/Test",
    "Re": "/mnt/d/glucose_data/internal/re_split2/Test"
}

INPUT_LEN = 48  # 4 hours
STRIDE = 12  # 1 hour
HORIZONS = [6, 12]  # 30 min, 60 min
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_group_by_subject(data_path):
    """Load CSV files and group by Subject ID."""
    subject_map = {}
    # Get all CSV files in the directory
    files = glob.glob(os.path.join(data_path, "*.csv"))

    print(f"Loading {len(files)} files from {data_path}...")

    for f in tqdm(files, desc="Parsing Files"):
        filename = os.path.basename(f)
        try:
            # Extract Subject ID (filename format: id_suffix.csv)
            sub_id = filename.split('_')[0]

            df = pd.read_csv(f)
            if 'GlucoseValue' not in df.columns: continue

            # if 'full_time' in df.columns:
            #     df['full_time'] = pd.to_datetime(df['full_time'])
            #     df = df.sort_values('full_time')

            # Get glucose values
            vals = df['GlucoseValue'].values.astype(np.float32)

            if len(vals) <= INPUT_LEN: continue

            if sub_id not in subject_map:
                subject_map[sub_id] = []
            subject_map[sub_id].append(vals)

        except Exception as e:
            print(f"Skipping {f}: {e}")

    return subject_map


def generate_windows(sessions, input_len, output_len, stride):
    """
    Generate sliding windows for all sessions of a subject.
    """
    X_list, Y_list = [], []
    for seq in sessions:
        seq_len = len(seq)
        if seq_len <= input_len + output_len: continue

        # Sliding window
        for i in range(0, seq_len - input_len - output_len + 1, stride):
            x = seq[i: i + input_len]
            y = seq[i + input_len: i + input_len + output_len]
            X_list.append(x)
            Y_list.append(y)

    return np.array(X_list), np.array(Y_list)


# ==========================================
# PART 3: Main
# ==========================================

def main():
    print(f"Running evaluation on device: {DEVICE}")

    # 1. Load model
    print("Loading TiRex model...")
    try:
        model = load_model("NX-AI/TiRex", backend="torch", device=DEVICE, compile=True)
    except:
        print("Model compilation fallback...")
        model = load_model("NX-AI/TiRex", backend="torch", device=DEVICE, compile=False)

    final_results = []

    # 2. Iterate through datasets
    for ds_name, ds_path in DATASETS.items():
        print(f"\n##########################################")
        print(f" PROCESSING DATASET: {ds_name}")
        print(f"##########################################")

        # Load current dataset
        subjects_data = load_and_group_by_subject(ds_path)
        subject_ids = list(subjects_data.keys())
        print(f"Dataset '{ds_name}' has {len(subject_ids)} subjects.")

        if len(subject_ids) == 0:
            print(f"Warning: No valid data found for {ds_name}, skipping.")
            continue

        # 3. Iterate through horizons
        for horizon in HORIZONS:
            horizon_min = horizon * 5

            # Store metrics
            metrics_storage = {
                "MAE": [], "RMSE": [], "R2": [],
                "CEA_AB": [], "CEA_CDE": [], "Runtime": []
            }

            # Iterate through subjects
            for sub_id in tqdm(subject_ids, desc=f"Evaluating {ds_name} (H={horizon_min}m)"):
                sessions = subjects_data[sub_id]

                # Generate samples
                X_np, Y_true = generate_windows(sessions, INPUT_LEN, horizon, STRIDE)
                if len(X_np) == 0: continue

                # Inference
                t_start = time.perf_counter()
                try:
                    ctx_tensor = torch.tensor(X_np, dtype=torch.float32)

                    # Zero-shot inference
                    _, preds_mean = model.forecast(
                        context=ctx_tensor,
                        prediction_length=horizon,
                        batch_size=BATCH_SIZE,
                        output_type="numpy"
                    )
                    Y_pred = preds_mean
                except Exception as e:
                    print(f"Error on {sub_id}: {e}")
                    continue
                t_end = time.perf_counter()

                # Calculate metrics
                flat_true = Y_true.flatten()
                flat_pred = Y_pred.flatten()

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

    # 4. Output
    if final_results:
        df_res = pd.DataFrame(final_results)

        # Adjust column order, put Dataset in the first column
        cols = ["Dataset", "Horizon (min)", "MAE", "RMSE", "R2 (%)", "CEA Zone A+B (%)", "CEA Zone C+D+E (%)",
                "Media Runtime (sec)"]
        df_res = df_res[cols]

        print("\n\n================ FINAL RESULTS ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        # Save
        csv_name = "tirex_glucose_evaluation.csv"
        df_res.to_csv(csv_name, index=False)
        print(f"\nResults saved to: {csv_name}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
