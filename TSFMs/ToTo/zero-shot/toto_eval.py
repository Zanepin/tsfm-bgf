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


# Toto Imports
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

# Ignore warnings
warnings.filterwarnings('ignore')

# ==========================================
# PART 2: Configuration and Data Loading
# ==========================================

DATASETS = {
    "Op": "/mnt/d/glucose_data/internal/op_split2/Test",
    "Re": "/mnt/d/glucose_data/internal/re_split2/Test"
}

# Experimental parameters
INPUT_LEN = 48  # 4 hours
STRIDE = 12  # 1 hour
HORIZONS = [6, 12]  # 30 min, 60 min
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    """Generate sliding windows"""
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
# PART 3: Toto Inference Wrapper (Add RevIN)
# ==========================================

def run_toto_inference_batched(forecaster, X_np, horizon, device, batch_size=64):
    """
    Manual batch inference for Toto with Instance Normalization (RevIN)
    """
    n_samples = len(X_np)
    all_preds = []

    X_tensor_all = torch.tensor(X_np, dtype=torch.float32).to(device)
    TIME_INTERVAL = 300.0

    for i in range(0, n_samples, batch_size):
        # 1. Prepare Mini-batch
        batch_input = X_tensor_all[i: i + batch_size]  # (B, T_in)
        current_bs = batch_input.shape[0]

        # -------------------------------------------------------
        # RevIN Logic Step 1: Normalize Input
        # Calculate mean and variance (on time dimension dim=1)
        # keepdim=True makes the shape (B, 1), which is convenient for broadcasting
        # -------------------------------------------------------
        batch_mean = torch.mean(batch_input, dim=1, keepdim=True)
        batch_std = torch.std(batch_input, dim=1, keepdim=True) + 1e-5  # Add epsilon to prevent division by zero

        # Perform normalization
        batch_input_norm = (batch_input - batch_mean) / batch_std
        # -------------------------------------------------------

        # 2. Construct Toto required metadata (use normalized data as input)
        time_int_secs = torch.full((current_bs,), TIME_INTERVAL, device=device, dtype=torch.float32)
        # Note: use batch_input_norm
        ts_secs = torch.zeros_like(batch_input_norm, device=device, dtype=torch.float32)
        pad_mask = torch.full_like(batch_input_norm, True, dtype=torch.bool, device=device)
        id_msk = torch.zeros_like(batch_input_norm, device=device, dtype=torch.float32)

        # 3. Encapsulate
        inputs = MaskedTimeseries(
            series=batch_input_norm,  # <-- Pass normalized data
            padding_mask=pad_mask,
            id_mask=id_msk,
            timestamp_seconds=ts_secs,
            time_interval_seconds=time_int_secs,
        )

        # 4. Predict
        with torch.no_grad():
            forecast = forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=100,
                samples_per_batch=100
            )

            # 5. Get results
            # Toto's output is usually (Horizon, Batch) or (Batch, Horizon)
            batch_preds = forecast.median.cpu().numpy()

            # -------------------------------------------------------
            # RevIN Logic Step 2: Denormalize Output
            # -------------------------------------------------------

            # Convert statistics to CPU Numpy for calculation
            mean_np = batch_mean.cpu().numpy()  # (B, 1)
            std_np = batch_std.cpu().numpy()  # (B, 1)

            # Align shapes: For convenient broadcasting calculation (pred * std + mean), we unify to (Batch, Horizon)
            if batch_preds.shape[0] == horizon and batch_preds.shape[1] == current_bs:
                # Current is (Horizon, Batch), first transpose to (Batch, Horizon)
                batch_preds = batch_preds.T

            # At this time, batch_preds is (Batch, Horizon)
            # mean_np/std_np is (Batch, 1)
            # Broadcasting mechanism will automatically handle
            batch_preds = batch_preds * std_np + mean_np

            # Restore to (Horizon, Batch) to match the original code's splicing logic
            batch_preds = batch_preds.T
            # -------------------------------------------------------

            all_preds.append(batch_preds)

    # 6. Concatenate
    # all_preds has shape [(Horizon, 64), (Horizon, 64), ..., (Horizon, 48)]
    # We concatenate on axis=1 (Batch direction), result -> (Horizon, Total_Samples)
    merged = np.concatenate(all_preds, axis=1)

    # 7. Transpose back to (Total_Samples, Horizon) for subsequent metric calculation
    return merged.T


# ==========================================
# PART 4: Main
# ==========================================

def main():
    print(f"Running Toto evaluation on device: {DEVICE}")

    # 1. Load Toto model
    print("Loading Toto-Open-Base-1.0...")
    try:
        toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
        toto.to(DEVICE)
        try:
            toto.compile()
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Compilation skipped: {e}")

        forecaster = TotoForecaster(toto.model)

    except Exception as e:
        print(f"Failed to load Toto model: {e}")
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
            print("No data found, skipping.")
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
                    # Call Toto wrapper function
                    Y_pred = run_toto_inference_batched(
                        forecaster,
                        X_np,
                        horizon,
                        DEVICE,
                        batch_size=BATCH_SIZE
                    )
                except Exception as e:
                    print(f"Inference error on {sub_id}: {e}")
                    continue
                t_end = time.perf_counter()

                # Calculate metrics
                flat_true = Y_true.flatten()
                flat_pred = Y_pred.flatten()

                # Data check
                if len(flat_true) != len(flat_pred):
                    print(f"Size mismatch: True {len(flat_true)} vs Pred {len(flat_pred)}")
                    continue

                if sub_id == subject_ids[0]:  # Only print for the first subject
                    print(f"\n--- Debug: Subject {sub_id} ---")
                    print(f"True Values (First 10): {flat_true[:10]}")
                    print(f"Pred Values (First 10): {flat_pred[:10]}")
                    print(f"True Mean: {np.mean(flat_true):.2f}, Pred Mean: {np.mean(flat_pred):.2f}")
                    print("-------------------------------")

                mae = mean_absolute_error(flat_true, flat_pred)
                rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
                r2 = r2_score(flat_true, flat_pred) * 100
                cea_res = calculate_clarke_metrics(flat_true, flat_pred)

                metrics_storage["MAE"].append(mae)
                metrics_storage["RMSE"].append(rmse)
                metrics_storage["R2"].append(r2)
                metrics_storage["CEA_AB"].append(cea_res['AB_percentage'])
                metrics_storage["CEA_CDE"].append(cea_res['CDE_percentage'])
                metrics_storage["Runtime"].append((t_end - t_start))

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

        print("\n\n================ TOTO RESULTS (WITH RevIN) ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        df_res.to_csv("toto_glucose_evaluation_revin.csv", index=False)
        print("\nResults saved to toto_glucose_evaluation_revin.csv")


if __name__ == "__main__":
    main()
