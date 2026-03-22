import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid, calculate_clarke_metrics


# ==========================================
# TabPFN Imports
# ==========================================
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
except ImportError:
    raise ImportError("请确保已安装 TabPFN: pip install tabpfn-client")

# Ignore warnings
warnings.filterwarnings('ignore')


# ==========================================
# PART 2: Configuration and data loading
# ==========================================

DATASETS = {
    "Op": "/mnt/d/glucose_data/internal/op_split2/Test",
    "Re": "/mnt/d/glucose_data/internal/re_split2/Test"
}

# Experimental parameters
INPUT_LEN = 48  # 4 hours
STRIDE = 12  # 1 hour
HORIZONS = [6, 12]  # 30 min, 60 min
MAX_SAMPLES_PER_SUBJECT = 100  # TabPFN inference is limited


def load_and_group_by_subject(data_path):
    """Load CSV and group by Subject ID"""
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
# PART 3: TabPFN inference wrapper
# ==========================================

def run_tabpfn_inference_loop(regressor, X_np, horizon):
    """
    Use TabPFN for per-sample inference (In-Context Learning)
    Treat each sliding window as an independent regression task:
    Fit: Time [0..47] -> Glucose History
    Predict: Time [48..48+H] -> Future Glucose
    """
    preds = []

    # Build time features (all samples share the same time step index)
    # Train Features: 0, 1, ..., 47
    train_indices = np.arange(X_np.shape[1]).reshape(-1, 1)
    # Test Features: 48, ..., 48+H-1
    test_indices = np.arange(X_np.shape[1], X_np.shape[1] + horizon).reshape(-1, 1)

    # Per-sample inference
    # Note: TabPFN needs to fit each independent context, unlike RNN/Transformer which can perform batch inference directly
    for i in tqdm(range(len(X_np)), desc="TabPFN Inference", leave=False):
        history_vals = X_np[i]  # Shape (48,)

        # 1. Construct context
        X_train = train_indices
        y_train = history_vals

        # 2. Fit (In-Context Learning)
        # TabPFN clears previous state and predicts based on current fit data
        regressor.fit(X_train, y_train)

        # 3. Predict
        # TabPFN outputs Mean by default (regression task)
        y_pred = regressor.predict(test_indices)

        preds.append(y_pred)

    return np.array(preds)


# ==========================================
# PART 4: Main
# ==========================================

def main():
    print("Initializing TabPFN V2 Regressor...")
    # Use TabPFN v2 version
    try:
        regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        print("TabPFN v2 loaded successfully.")
    except Exception as e:
        print(f"Error loading specific version, falling back to default constructor: {e}")
        regressor = TabPFNRegressor()  # Fallback

    final_results = []

    # 1. Iterate through datasets
    for ds_name, ds_path in DATASETS.items():
        print(f"\n##########################################")
        print(f" PROCESSING DATASET: {ds_name}")
        print(f"##########################################")

        subjects_data = load_and_group_by_subject(ds_path)
        subject_ids = list(subjects_data.keys())

        if not subject_ids:
            print("No data found, skipping.")
            continue

        # 2. Iterate through prediction horizons
        for horizon in HORIZONS:
            horizon_min = horizon * 5

            metrics_storage = {
                "MAE": [], "RMSE": [], "R2": [],
                "CEA_AB": [], "CEA_CDE": [], "Runtime": []
            }

            # 3. Iterate through subjects
            for sub_id in tqdm(subject_ids, desc=f"Evaluating {ds_name} (H={horizon_min}m)"):
                sessions = subjects_data[sub_id]

                # Generate samples
                X_np, Y_true = generate_windows(sessions, INPUT_LEN, horizon, STRIDE)
                total_samples = len(X_np)

                if total_samples == 0: continue

                # Record the actual number of samples used for inference
                if total_samples > MAX_SAMPLES_PER_SUBJECT:
                    indices = np.random.choice(total_samples, MAX_SAMPLES_PER_SUBJECT, replace=False)
                    X_input = X_np[indices]
                    Y_eval = Y_true[indices]
                    processed_count = MAX_SAMPLES_PER_SUBJECT
                else:
                    X_input = X_np
                    Y_eval = Y_true
                    processed_count = total_samples

                # Inference
                t_start = time.perf_counter()
                try:
                    Y_pred = run_tabpfn_inference_loop(
                        regressor,
                        X_input,
                        horizon
                    )
                except Exception as e:
                    print(f"Inference error on {sub_id}: {e}")
                    continue
                t_end = time.perf_counter()

                measured_time = t_end - t_start
                estimated_full_runtime = measured_time * (total_samples / processed_count)

                # Calculate metrics
                flat_true = Y_eval.flatten()
                flat_pred = Y_pred.flatten()

                # Simple outlier handling
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

                metrics_storage["Runtime"].append(estimated_full_runtime)

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
        cols = ["Dataset", "Horizon (min)", "MAE", "RMSE", "R2 (%)", "CEA Zone A+B (%)", "CEA Zone C+D+E (%)",
                "Media Runtime (sec)"]
        df_res = df_res[cols]

        print("\n\n================ TABPFN V2 ZERO-SHOT RESULTS ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        df_res.to_csv("tabpfn_v2_glucose_evaluation.csv", index=False)
        print("\nResults saved to tabpfn_v2_glucose_evaluation.csv")


if __name__ == "__main__":
    main()