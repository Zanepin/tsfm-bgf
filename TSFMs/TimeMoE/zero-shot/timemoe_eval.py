import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForCausalLM
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

# Experimental parameters
INPUT_LEN = 48  # 4 hours
STRIDE = 12  # 1 hour
HORIZONS = [6, 12]  # 30 min, 60 min
BATCH_SIZE = 64  # Batch size for inference loop
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Time-MoE Model Parameters
# Options: Maple728/TimeMoE-50M, Maple728/TimeMoE-200M, Maple728/TimeMoE-400M, Maple728/TimeMoE-800M
# Recommend using 200M or 50M for fast testing; 800M may perform better but is slower.
TIMEMOE_MODEL_ID = "Maple728/TimeMoE-200M"


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
# PART 3: Time-MoE Inference Wrapper
# ==========================================

def run_timemoe_inference_batched(model, X_np, prediction_length, device, batch_size=64):
    """
    Perform Batch Inference for Time-MoE
    Includes Instance Normalization (RevIN logic)
    """
    n_samples = len(X_np)
    all_preds = []

    # 1. Convert to Tensor (B, T)
    # Time-MoE is a Causal LM, so the input is directly a 1D time series
    X_tensor_all = torch.tensor(X_np, dtype=torch.float32).to(device)

    for i in range(0, n_samples, batch_size):
        batch_input = X_tensor_all[i: i + batch_size]

        # 2. Instance Normalization (Standardize per series)
        # mean: (B, 1), std: (B, 1)
        seq_mean = batch_input.mean(dim=-1, keepdim=True)
        seq_std = batch_input.std(dim=-1, keepdim=True)

        # Prevent division by zero
        seq_std = torch.where(seq_std < 1e-5, torch.tensor(1e-5, device=device), seq_std)

        normed_batch_input = (batch_input - seq_mean) / seq_std

        # 3. Predict
        with torch.no_grad():
            # Time-MoE's generate method accepts a normalized float tensor
            # Returns: (B, Context + Horizon)
            output_sequences = model.generate(
                normed_batch_input,
                max_new_tokens=prediction_length,
                pad_token_id=None  # Prevent warnings
            )

            # Intercept the prediction part (the last prediction_length points)
            normed_preds = output_sequences[:, -prediction_length:]

        # 4. Inverse Normalization
        batch_preds = normed_preds * seq_std + seq_mean

        all_preds.append(batch_preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# ==========================================
# PART 4: Main
# ==========================================

def main():
    print(f"Running Time-MoE evaluation on device: {DEVICE}")

    # 1. Load Model
    print(f"Loading {TIMEMOE_MODEL_ID}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            TIMEMOE_MODEL_ID,
            device_map=DEVICE,
            trust_remote_code=True
        )
        model.eval()  # Ensure evaluation mode
        print("Time-MoE loaded successfully.")
    except Exception as e:
        print(f"Failed to load Time-MoE model: {e}")
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

            # Time-MoE Limit Check (Context + Pred < 4096)
            if INPUT_LEN + horizon > 4096:
                print(f"Warning: Sequence length {INPUT_LEN + horizon} exceeds model limit.")

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
                    # Use the new Time-MoE inference function
                    Y_pred = run_timemoe_inference_batched(
                        model,
                        X_np,
                        prediction_length=horizon,
                        device=DEVICE,
                        batch_size=BATCH_SIZE
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
                    flat_p = Y_pred.flatten()
                    print(f"\n[Debug {sub_id}] Mean True: {np.mean(flat_t):.2f}, Mean Pred: {np.mean(flat_p):.2f}")
                # -------------------------------

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

        print("\n\n================ TIME-MOE RESULTS ================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_res.to_string(index=False))

        df_res.to_csv("timemoe_glucose_evaluation.csv", index=False)
        print("\nResults saved to timemoe_glucose_evaluation.csv")


if __name__ == "__main__":
    main()