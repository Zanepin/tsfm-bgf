"""
Evaluate Moirai model using CSV files - V5 (Merged Zone A+B and C+D+E)
Features:
1. Integrated standard ClarkeErrorGrid class
2. Automatic unit detection
3. Flattened batch inference
4. Outputs only merged A+B and C+D+E metrics
"""
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import time
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid


# --- 2. Helper and Model Functions ---

def calculate_detailed_metrics(ref, pred):
    """
    Calculate all metrics, including unit conversion logic to adapt to the Clarke class.
    """
    ref = np.array(ref)
    pred = np.array(pred)
    
    # Basic metrics
    mae = np.mean(np.abs(ref - pred))
    rmse = np.sqrt(np.mean((ref - pred)**2))
    
    ss_res = np.sum((ref - pred)**2)
    ss_tot = np.sum((ref - np.mean(ref))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Clarke Error Grid Analysis (Unit Adaptation)
    # Convert mmol/L to mg/dL if the values are small
    if np.max(ref) < 50: 
        ref_clarke = ref * 18.0182
        pred_clarke = pred * 18.0182
    else:
        ref_clarke = ref
        pred_clarke = pred
        
    ceg = ClarkeErrorGrid()
    clarke_results = ceg.run(ref_clarke, pred_clarke)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        **clarke_results
    }

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uni2ts.model.moirai import MoiraiForecast

def load_model(checkpoint_path, context_length=48, patch_size=16, device='cuda'):
    print(f"Loading model from: {checkpoint_path}")
    
    import torch
    safe_globals = []
    try:
        from uni2ts.distribution.mixture import MixtureOutput
        from uni2ts.distribution.student_t import StudentTOutput
        from uni2ts.distribution.normal import NormalFixedScaleOutput
        from uni2ts.distribution.negative_binomial import NegativeBinomialOutput
        from uni2ts.distribution.log_normal import LogNormalOutput
        from uni2ts.distribution.laplace import LaplaceOutput
        from uni2ts.distribution.pareto import ParetoOutput
        from uni2ts.loss.packed.distribution import PackedNLLLoss
        from uni2ts.loss.packed.point import PackedMSELoss, PackedNRMSELoss
        from uni2ts.loss.packed.normalized import PointNormType, PackedNMSELoss, PackedNRMSELoss as PackedNRMSELossNorm
        
        safe_globals.extend([
            MixtureOutput, StudentTOutput, NormalFixedScaleOutput, 
            NegativeBinomialOutput, LogNormalOutput, LaplaceOutput, ParetoOutput,
            PackedNLLLoss, PackedMSELoss, PackedNRMSELoss, 
            PointNormType, PackedNMSELoss, PackedNRMSELossNorm
        ])
        torch.serialization.add_safe_globals(safe_globals)
    except ImportError:
        pass

    model = MoiraiForecast.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        patch_size=patch_size,
        context_length=context_length,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        map_location='cpu'
    )
    model.eval()
    model = model.to(device)
    return model

def extract_subject_id(item_id):
    return str(item_id).split('_')[0] if '_' in str(item_id) else str(item_id)

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.windows = windows
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return self.windows[idx]

def prepare_all_windows(df, id_col, target_col, context_length, prediction_length, stride):
    all_windows = []
    grouped = df.groupby(id_col)
    
    for item_id, group in tqdm(grouped, desc="Prep Data"):
        ts_data = group[target_col].values.astype(np.float32)
        total_len = len(ts_data)
        if total_len < context_length + prediction_length:
            continue
        
        subject_id = extract_subject_id(item_id)
        for i in range(context_length, total_len - prediction_length + 1, stride):
            context = ts_data[i-context_length : i]
            ground_truth = ts_data[i : i+prediction_length]
            all_windows.append({
                'context': context, 'ground_truth': ground_truth,
                'subject_id': subject_id
            })
    return all_windows

def collate_fn(batch):
    contexts = np.stack([item['context'] for item in batch])
    past_target = torch.from_numpy(contexts).float().unsqueeze(-1)
    metadata = [{'subject_id': item['subject_id'], 'ground_truth': item['ground_truth']} for item in batch]
    return past_target, metadata

# --- 3. Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Moirai Evaluation (V5 Merged Zones)")
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--prediction_length', type=int, default=6)
    parser.add_argument('--context_length', type=int, default=48)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--target_mean', type=float, default=None, help='RevIN Mean')
    parser.add_argument('--target_std', type=float, default=None, help='RevIN Std')
    
    args = parser.parse_args()

    # Load data
    print(f"Loading data...")
    df = pd.read_csv(args.csv_file)
    df = df.sort_values(['id', 'timestamp'])
    
    # Prepare windows
    windows_list = prepare_all_windows(
        df, 'id', 'glucose', 
        args.context_length, args.prediction_length, args.stride
    )
    
    dataset = TimeSeriesDataset(windows_list)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=0  # Set to 0 to eliminate ~180s process initialization delay
    )

    # Load model
    model = load_model(args.checkpoint, args.context_length, args.patch_size, args.device)

    # Store results
    subject_data = defaultdict(lambda: {'preds': [], 'truths': [], 'n_windows': 0})
    
    # Inference
    print(f"\nStarting Inference...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_past_target, batch_meta in tqdm(dataloader, desc="Infer"):
            batch_past_target = batch_past_target.to(args.device)
            bs, seq_len, _ = batch_past_target.shape
            
            # Construct input
            forecast_samples = model(
                past_target=batch_past_target,
                past_observed_target=torch.ones_like(batch_past_target, dtype=torch.bool),
                past_is_pad=torch.zeros((bs, seq_len), dtype=torch.bool, device=args.device),
                num_samples=100
            )
            
            # Take median
            if forecast_samples.ndim == 4:
                preds = torch.median(forecast_samples[:, :, :, 0], dim=1).values.cpu().numpy()
            else:
                preds = torch.median(forecast_samples, dim=1).values.cpu().numpy()
            
            # Collect data
            for i, meta in enumerate(batch_meta):
                sub_id = meta['subject_id']
                subject_data[sub_id]['preds'].append(preds[i])
                subject_data[sub_id]['truths'].append(meta['ground_truth'])
                subject_data[sub_id]['n_windows'] += 1

    total_inference_time = time.time() - start_time
    total_windows = len(windows_list)
    avg_time_per_window = total_inference_time / total_windows if total_windows > 0 else 0

    print(f"\nInference finished in {total_inference_time:.2f}s")

    # --- Calculate Metrics ---
    print("\n" + "-"*100)
    print(f"{'Subject':<10} {'MAE':<8} {'RMSE':<8} {'R2':<8} {'Zone A+B%':<12} {'Zone C+D+E%':<12} {'Runtime(s)':<10}")
    print("-" * 100)

    final_metrics = []
    subject_runtimes = []

    for sub_id in sorted(subject_data.keys()):
        data = subject_data[sub_id]
        
        all_preds = np.concatenate(data['preds'])
        all_truths = np.concatenate(data['truths'])
        
        # Inverse Normalization
        if args.target_mean is not None and args.target_std is not None:
            all_preds = all_preds * args.target_std + args.target_mean
            all_truths = all_truths * args.target_std + args.target_mean
        
        # Estimate Runtime
        sub_runtime = avg_time_per_window * data['n_windows']
        subject_runtimes.append(sub_runtime)
        
        # Calculate detailed metrics
        m = calculate_detailed_metrics(all_truths, all_preds)
        
        # Get metrics for A+B and C+D+E
        zone_ab = m.get('AB_percentage', np.nan)
        zone_cde = m.get('CDE_percentage', np.nan)
        
        m['Subject'] = sub_id
        m['Runtime'] = sub_runtime
        m['AB'] = zone_ab
        m['CDE'] = zone_cde
        
        final_metrics.append(m)
        
        print(f"{sub_id:<10} "
              f"{m['MAE']:>6.2f}   "
              f"{m['RMSE']:>6.2f}   "
              f"{m['R2']:>6.4f}   "
              f"{zone_ab:>9.2f}   "
              f"{zone_cde:>11.2f}   "
              f"{sub_runtime:>8.3f}")

    # --- Overall Summary ---
    if final_metrics:
        df_m = pd.DataFrame(final_metrics)
        runtimes = np.array(subject_runtimes)
        
        print("-" * 100)
        print("OVERALL PERFORMANCE SUMMARY")
        print("-" * 100)
        print(f"MAE:         {df_m['MAE'].mean():.2f} ± {df_m['MAE'].std():.2f}")
        print(f"RMSE:        {df_m['RMSE'].mean():.2f} ± {df_m['RMSE'].std():.2f}")
        print(f"R2:          {df_m['R2'].mean():.4f} ± {df_m['R2'].std():.4f}")
        print("-" * 40)
        print(f"Zone A+B:    {df_m['AB'].mean():.2f}% ± {df_m['AB'].std():.2f}")
        print(f"Zone C+D+E:  {df_m['CDE'].mean():.2f}% ± {df_m['CDE'].std():.2f}")
        print("-" * 40)
        print(f"Runtime Median (per Subject): {np.median(runtimes):.3f} s")
        print(f"Runtime Mean   (per Subject): {np.mean(runtimes):.3f} s")
        print("-" * 100)

if __name__ == "__main__":
    main()