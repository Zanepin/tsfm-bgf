import os
import glob
import time
import warnings
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid, calculate_clarke_metrics


# ==========================================
# FlowState Imports
# ==========================================
from tsfm_public import FlowStateForPrediction

# Ignore common warnings
warnings.filterwarnings('ignore')
# Ignore specific tsfm configuration warnings (for cleaner output)
logging.getLogger("tsfm_public.models.flowstate.configuration_flowstate").setLevel(logging.ERROR)


# ==========================================
# PART 2: Configuration and Data Loading
# ==========================================

DATASETS = {
    "Op": "/mnt/d/glucose_data/internal/op_split2/Test",
    "Re": "/mnt/d/glucose_data/internal/re_split2/Test"
}

# Experimental parameters
INPUT_LEN = 48  # 4 hours (assumption: 5min intervals)
STRIDE = 12  # 1 hour
HORIZONS = [6, 12]  # 30 min, 60 min

# Batch Size Updated
BATCH_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Configuration
MODEL_NAME = "ibm-granite/granite-timeseries-flowstate-r1"


def load_and_group_by_subject(data_path):
    """Load CSV files and group by Subject ID"""
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
            y_list.append(y)
    return np.array(X_list), np.array(Y_list)


# ==========================================
# PART 3: FlowState Inference Wrapper
# ==========================================

def run_flowstate_inference_batched(model, X_np, horizon, device, batch_size=64):
    """
    Perform Batch Inference for FlowState
    X_np: (N_samples, Input_Len)
    """
    n_samples = len(X_np)
    all_preds = []

    # Convert to Tensor and add feature dimension
    # FlowState input shape: (Batch, Time, Features) -> (Batch, 48, 1)
    X_tensor_all = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1).to(device)

    # Use DataLoader to process batches
    dataset = torch.utils.data.TensorDataset(X_tensor_all)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0]  # (B, T, 1)

            # FlowState Forward Pass
            # scale_factor=1.0 is default. Adjust if specific seasonality scaling is known.
            # prediction_length determines the output step length
            forecast = model(
                past_values=batch_x,
                prediction_length=horizon,
                batch_first=True,
                scale_factor=0.25
            )

            # Get prediction results
            # quantile_outputs Shape: (Batch, Quantiles, Horizon, Features)
            # Default quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # Index 4 corresponds to 0.5 (Median)

            # Extract Median (Median/0.5 quantile)
            # Shape change: (B, 9, H, 1) -> (B, H, 1) -> (B, H)
            batch_preds = forecast.quantile_outputs[:, 4, :, 0]

            all_preds.append(batch_preds.cpu().numpy())

    # Concatenate results
    return np.concatenate(all_preds, axis=0)


# ==========================================
# PART 4: Main Flow
# ==========================================

def main():
    print(f"Running FlowState ({MODEL_NAME}) evaluation on device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")

    # 1. Load model
    print(f"Loading {MODEL_NAME}...")
    try:
        model = FlowStateForPrediction.from_pretrained(MODEL_NAME).to(DEVICE)
        print("FlowState model loaded successfully.")
    except Exception as e:
        print(f"Failed to load FlowState model: {e}")
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
            horizon_min = horizon * 5  # Assumption: 5 min data

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
                    # FlowState supports dynamic prediction_length, just pass in horizon directly
                    Y_pred = run_flowstate_inference_batched(
                        model,
                        X_np,
                        horizon=horizon,
                        device=DEVICE,
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

        print("\n\n================ FLOWSTATE RESULTS ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        output_filename = "flowstate_glucose_evaluation.csv"
        df_res.to_csv(output_filename, index=False)
        print(f"\nResults saved to {output_filename}")


if __name__ == "__main__":
    main()