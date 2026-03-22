import os
import glob
import time
import warnings
import argparse
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid, calculate_clarke_metrics


# ==========================================
# Moirai Imports
# ==========================================
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # v1.1
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module  # v2.0

warnings.filterwarnings('ignore')

# ==========================================
# PART 2: Configuration and tool functions
# ==========================================

DATASETS = {
    "Op": "/mnt/d/glucose_data/internal/op_split2/Test",
    "Re": "/mnt/d/glucose_data/internal/re_split2/Test",
    "lg": "/mnt/d/glucose_data/internal/lg_split2/Test"
}

INPUT_LEN = 48  # 4 hours
STRIDE = 12  # 1 hour
HORIZONS = [6, 12]  # 30 min, 60 min
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_group_by_subject(data_path):
    subject_map = {}
    files = glob.glob(os.path.join(data_path, "*.csv"))
    for f in files:
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
        except Exception:
            pass
    return subject_map

def generate_windows(sessions, input_len, output_len, stride):
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
# PART 3: General inference core
# ==========================================

def run_moirai_inference_batched(forecaster, X_np, device, batch_size=64, num_samples=100):
    n_samples_data = len(X_np)
    all_preds = []

    X_tensor_all = torch.tensor(X_np, dtype=torch.float32).to(device)
    forecaster.to(device)
    forecaster.eval()

    for i in range(0, n_samples_data, batch_size):
        batch_input = X_tensor_all[i: i + batch_size] 
        current_bs = batch_input.shape[0]
        past_target = batch_input.unsqueeze(-1) # (B, T, 1)
        
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool, device=device)
        past_is_pad = torch.zeros(current_bs, past_target.shape[1], dtype=torch.bool, device=device)

        with torch.no_grad():
            output = forecaster(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad
            )
            
            if hasattr(output, 'sample'):
                # === Moirai 1.1 ===
                samples = output.sample((num_samples,))
                median_preds = samples.median(dim=0).values
            
            elif isinstance(output, torch.Tensor):
                # === Moirai 2.0 ===
                if output.dim() == 3: 
                    median_preds = output.median(dim=1).values
                elif output.dim() == 4:
                    median_preds = output.median(dim=1).values
                else:
                    if output.shape[1] == current_bs:
                        median_preds = output.median(dim=0).values
                    else:
                        median_preds = output.median(dim=1).values
            else:
                raise ValueError(f"Unknown output type: {type(output)}")

            batch_preds_np = median_preds.cpu().numpy()
            if batch_preds_np.ndim == 3:
                batch_preds_np = batch_preds_np.squeeze(-1)
            
            all_preds.append(batch_preds_np)

    return np.concatenate(all_preds, axis=0)


# ==========================================
# PART 4: Main workflow
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description="Moirai Glucose Evaluation")
    parser.add_argument('--version', type=str, default='2.0', choices=['1.1', '2.0'])
    parser.add_argument('--size', type=str, default='small', choices=['small', 'base'])
    args = parser.parse_args()
    
    if args.version == '2.0' and args.size == 'base':
        print("Notice: Switching to 'small' for Moirai 2.0 (base not available).")
        args.size = 'small'
    return args

def main():
    args = get_args()
    
    # Build model ID
    if args.version == '1.1':
        model_id = f"Salesforce/moirai-1.1-R-{args.size}"
    else:
        model_id = f"Salesforce/moirai-2.0-R-{args.size}"

    print(f"Running Evaluation for: {model_id} on {DEVICE}")

    # 1. Load model
    try:
        if args.version == '1.1':
            module = MoiraiModule.from_pretrained(model_id)
        else:
            module = Moirai2Module.from_pretrained(model_id)
        module.to(DEVICE)
        module.eval()
    except Exception as e:
        print(f"Failed to load model {model_id}: {e}")
        return

    final_results = []

    # 2. Iterate through datasets
    for ds_name, ds_path in DATASETS.items():
        print(f"\n--- DATASET: {ds_name} ---")
        subjects_data = load_and_group_by_subject(ds_path)
        subject_ids = list(subjects_data.keys())
        if not subject_ids: continue

        # 3. Iterate through prediction horizons
        for horizon in HORIZONS:
            horizon_min = horizon * 5
            
            # Only Moirai 1.1 has strict Patch alignment requirements
            # We assume minimum Patch Size = 8
            aligned_horizon = horizon
            
            if args.version == '1.1':
                min_patch_size = 8
                # If horizon is not divisible by 8, round up to the nearest multiple of 8
                if horizon % min_patch_size != 0:
                    aligned_horizon = math.ceil(horizon / min_patch_size) * min_patch_size
                    print(f"Info: Aligning horizon {horizon} -> {aligned_horizon} to match patch size {min_patch_size}")

            # Initialize Forecaster
            if args.version == '1.1':
                forecaster = MoiraiForecast(
                    module=module,
                    prediction_length=aligned_horizon,
                    context_length=INPUT_LEN,
                    patch_size=8,
                    num_samples=100,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            else:
                forecaster = Moirai2Forecast(
                    module=module,
                    prediction_length=horizon,
                    context_length=INPUT_LEN,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            
            metrics_storage = {"MAE": [], "RMSE": [], "R2": [], "CEA_AB": [], "CEA_CDE": [], "Runtime": []}
            
            pbar = tqdm(subject_ids, desc=f"Eval {ds_name} H={horizon_min}m")
            for sub_id in pbar:
                sessions = subjects_data[sub_id]
                # Generate data using the original horizon (6) because we want to compare the last 6 points of the true data
                X_np, Y_true = generate_windows(sessions, INPUT_LEN, horizon, STRIDE)
                if len(X_np) == 0: continue

                t_start = time.perf_counter()
                try:
                    # Inference
                    Y_pred = run_moirai_inference_batched(forecaster, X_np, DEVICE, batch_size=BATCH_SIZE)
                    
                    # If aligned_horizon (8) > horizon (6), the model will output 8 points
                    # We need to cut off the extra 2 points and keep only the first 6
                    if Y_pred.shape[1] > horizon:
                        Y_pred = Y_pred[:, :horizon]
                        
                except Exception as e:
                    print(f"Err {sub_id}: {e}")
                    continue
                t_end = time.perf_counter()

                # Metrics
                flat_true = Y_true.flatten()
                flat_pred = Y_pred.flatten()
                
                # check length
                if len(flat_true) != len(flat_pred):
                    # If it's still not right, it means the data generation logic is problematic, skip
                    continue

                metrics_storage["MAE"].append(mean_absolute_error(flat_true, flat_pred))
                metrics_storage["RMSE"].append(np.sqrt(mean_squared_error(flat_true, flat_pred)))
                metrics_storage["R2"].append(r2_score(flat_true, flat_pred) * 100)
                cea = calculate_clarke_metrics(flat_true, flat_pred)
                metrics_storage["CEA_AB"].append(cea['AB_percentage'])
                metrics_storage["CEA_CDE"].append(cea['CDE_percentage'])
                metrics_storage["Runtime"].append(t_end - t_start)

            def fmt(lst): return f"{np.mean(lst):.2f} ± {np.std(lst):.2f}" if lst else "N/A"
            res_row = {
                "Model": model_id.split('/')[-1],
                "Dataset": ds_name,
                "Horizon": f"{horizon_min} min",
                "MAE": fmt(metrics_storage["MAE"]),
                "RMSE": fmt(metrics_storage["RMSE"]),
                "R2": fmt(metrics_storage["R2"]),
                "CEA A+B": fmt(metrics_storage["CEA_AB"]),
                "Runtime": f"{np.median(metrics_storage['Runtime']):.2f}s"
            }
            final_results.append(res_row)

    if final_results:
        df_res = pd.DataFrame(final_results)
        print("\n\n================ RESULTS ================")
        pd.set_option('display.max_columns', None)
        print(df_res.to_string(index=False))
        csv_name = f"eval_{args.version}_{args.size}.csv"
        df_res.to_csv(csv_name, index=False)
        print(f"\nSaved to {csv_name}")

if __name__ == "__main__":
    main()