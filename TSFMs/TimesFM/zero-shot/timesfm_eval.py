import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import timesfm
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid, calculate_clarke_metrics


# ==========================================
# PART 2: Configuration and Data Loading (Remain Unchanged)
# ==========================================

# Optimize matrix multiplication precision, compatible with Ampere architecture GPUs
torch.set_float32_matmul_precision("high")

DATASETS = {
    "Op": "/mnt/d/glucose_data/internal/op_split2/Test",
    "Re": "/mnt/d/glucose_data/internal/re_split2/Test"
}

# Experimental parameters
INPUT_LEN = 48  # 4 hours (Context Length)
STRIDE = 12     # 1 hour
HORIZONS = [6, 12]  # 30 min, 60 min
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TimesFM Model ID
TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"

def load_and_group_by_subject(data_path):
    """Load CSVs and group by Subject ID"""
    subject_map = {}
    files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"Loading {len(files)} files from {data_path}...")

    for f in tqdm(files, desc="Parsing Files"):
        filename = os.path.basename(f)
        try:
            sub_id = filename.split('_')[0]
            df = pd.read_csv(f)
            if 'GlucoseValue' not in df.columns: continue
            vals = df['GlucoseValue'].values.astype(np.float32)
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
# PART 4: Main Flow
# ==========================================

def main():
    print(f"Running TimesFM-2.5 evaluation on device: {DEVICE}")

    final_results = []

    # 2. Iterate through datasets
    for ds_name, ds_path in DATASETS.items():
        print(f"\n##########################################")
        print(f" PROCESSING DATASET: {ds_name}")
        print(f"##########################################")

        subjects_data = load_and_group_by_subject(ds_path)
        subject_ids = list(subjects_data.keys())

        if not subject_ids:
            print("No data found, skipping.")
            continue

        # 3. Iterate through prediction horizons
        for horizon in HORIZONS:
            print(f"\nCompiling model for horizon: {horizon}")
            horizon_min = horizon * 5

            # Load TimesFM model
            print(f"Loading {TIMESFM_CHECKPOINT}...")
            try:
                model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_CHECKPOINT, torch_compile=True)

                model.compile(
                    timesfm.ForecastConfig(
                        max_context=INPUT_LEN,
                        max_horizon=horizon,  # Select current horizon
                        normalize_inputs=True,
                        use_continuous_quantile_head=True,
                        force_flip_invariance=True,
                        infer_is_positive=True,
                        fix_quantile_crossing=True,
                        per_core_batch_size=1024,
                    )
                )
                print("TimesFM compiled successfully.")

            except Exception as e:
                print(f"Failed to load/compile TimesFM model: {e}")
                return

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
                    # Call model forecast method directly for inference
                    point_forecast, quantile_forecast = model.forecast(
                        inputs=[row.tolist() for row in X_np],  # Convert input to list format
                        horizon=horizon
                    )
                except Exception as e:
                    print(f"Inference error on {sub_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                t_end = time.perf_counter()

                # --- Debug Section ---
                if sub_id == subject_ids[0]:
                    flat_t = Y_true.flatten()
                    flat_p = point_forecast.flatten()  # Get forecast results
                    print(f"\n[Debug {sub_id}] Mean True: {np.mean(flat_t):.2f}, Mean Pred: {np.mean(flat_p):.2f}")
                # ---------------------

                # Calculate metrics (Maintain original logic)
                flat_true = Y_true.flatten()
                flat_pred = point_forecast.flatten()

                # Prevent negative predictions (Glucose levels cannot be negative)
                flat_pred = np.maximum(flat_pred, 0)

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
        cols = ["Dataset", "Horizon (min)", "MAE", "RMSE", "R2 (%)", "CEA Zone A+B (%)", "CEA Zone C+D+E (%)", "Media Runtime (sec)"]
        df_res = df_res[cols]

        print("\n\n================ TIMESFM-2.5 RESULTS ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        df_res.to_csv("timesfm2p5_horizon_glucose_evaluation.csv", index=False)
        print("\nResults saved to timesfm2p5_horizon_glucose_evaluation.csv")

if __name__ == "__main__":
    main()