import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.base_wrapper import BaseModelWrapper
from utils.metrics import calculate_all_metrics

# Darts imports
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.utils.callbacks import TFMProgressBar

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')

class TFTWrapper(BaseModelWrapper):
    TFT_QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_root = self.config.get("data_root", "/mnt/d/glucose_data/Internal")

    def load_dataset_as_series(self, folder_path: str, max_len: int = None) -> list:
        folder = Path(folder_path)
        if not folder.exists():
            return []

        csv_files = sorted(list(folder.glob("*.csv")))
        series_list = []

        for i, f in enumerate(csv_files):
            try:
                df = pd.read_csv(f)
                vals = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                if len(vals) < (self.config['input_len'] + 12): 
                    continue
                if max_len is not None and len(vals) > max_len:
                    vals = vals[:max_len]

                start_time = f"2024-01-{i % 28 + 1:02d} {(i * 2) % 24:02d}:00:00"
                timestamps = pd.date_range(start=start_time, periods=len(vals), freq=self.config['time_freq'])
                
                ts = TimeSeries.from_dataframe(
                    pd.DataFrame({'val': vals}, index=timestamps),
                    value_cols=['val'], freq=self.config['time_freq']
                ).astype(np.float32)
                
                series_list.append(ts)
            except Exception:
                pass
                
        return series_list

    def train(self, dataset_type: str, pred_len: int):
        model_name = f"tft_{dataset_type}_pred{pred_len}_native_cut"
        model_dir = os.path.join(current_dir, "saved_models", model_name)
        os.makedirs(model_dir, exist_ok=True)

        train_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Train")
        val_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Val")

        train_series = self.load_dataset_as_series(train_path, max_len=self.config['train_sample_len'])
        val_series = self.load_dataset_as_series(val_path, max_len=self.config['val_sample_len'])

        if not train_series: return None

        add_encoders = {'cyclic': {'future': ['hour', 'minute']}}
        
        pl_trainer_kwargs = {
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": self.config['max_epochs'],
            "enable_model_summary": True,
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
        
        from pytorch_lightning.callbacks import EarlyStopping
        pl_trainer_kwargs["callbacks"].append(
            EarlyStopping(monitor="val_loss", patience=self.config['early_stop_patience'], mode="min", verbose=True)
        )

        model = TFTModel(
            input_chunk_length=self.config['input_len'],
            output_chunk_length=pred_len,
            hidden_size=self.config['hidden_size'],
            lstm_layers=self.config['lstm_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            dropout=self.config['dropout'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['max_epochs'],
            add_relative_index=True,
            add_encoders=add_encoders,
            use_reversible_instance_norm=True,
            likelihood=QuantileRegression(quantiles=self.TFT_QUANTILES),
            optimizer_kwargs={"lr": self.config['learning_rate']},
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name=model_name,
            work_dir=model_dir,
            save_checkpoints=True,
            force_reset=True
        )

        model.fit(series=train_series, val_series=val_series, max_samples_per_ts=150, verbose=True)
        return TFTModel.load_from_checkpoint(model_name=model_name, work_dir=model_dir, best=True)

    def evaluate(self, model, dataset_type: str, pred_len: int) -> Dict[str, Any]:
        test_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Test")
        test_series_list = self.load_dataset_as_series(test_path)

        if not test_series_list: return {}

        metrics_storage = defaultdict(list)
        detailed_results = []
        subject_runtimes = []

        input_chunk = self.config['input_len']
        eval_stride = self.config['stride']

        for i, full_ts in enumerate(test_series_list):
            if len(full_ts) < input_chunk + pred_len: continue

            t_start = time.time()
            try:
                vals = full_ts.values().flatten()
                batch_inputs = []
                batch_targets = []
                n_points = len(full_ts)

                for start_idx in range(0, n_points - input_chunk - pred_len + 1, eval_stride):
                    in_end = start_idx + input_chunk
                    batch_inputs.append(full_ts[start_idx: in_end])
                    batch_targets.append(vals[in_end: in_end + pred_len])

                if not batch_inputs: continue

                preds = model.predict(n=pred_len, series=batch_inputs, num_samples=50, verbose=False)

                all_pred_vals = []
                all_true_vals = []

                for idx, p_ts in enumerate(preds):
                    p_val = p_ts.quantile(0.5).values().flatten()
                    t_val = batch_targets[idx]
                    all_pred_vals.extend(p_val)
                    all_true_vals.extend(t_val)
                
                runtime = time.time() - t_start
                subject_runtimes.append(runtime)

                metrics = calculate_all_metrics(all_true_vals, all_pred_vals)
                
                metrics_storage["MAE"].append(metrics["MAE"])
                metrics_storage["RMSE"].append(metrics["RMSE"])
                metrics_storage["R2"].append(metrics["R2"])
                metrics_storage["CEA_AB"].append(metrics["CEA_AB"])
                metrics_storage["CEA_CDE"].append(metrics["CEA_CDE"])

                detailed_results.append({
                    "Subject_ID": f"Sub_{i}",
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                    "CEA_AB": metrics["CEA_AB"]
                })

            except Exception as e:
                continue
                
        out_csv = os.path.join(current_dir, f"subject_metrics_{dataset_type}_pred{pred_len}.csv")
        pd.DataFrame(detailed_results).to_csv(out_csv, index=False)

        return self.format_metrics_summary(dataset_type, pred_len, metrics_storage, subject_runtimes)

if __name__ == "__main__":
    BASE_CONFIG = {
        "data_root": "/mnt/d/glucose_data/Internal",
        "hidden_size": 64,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "batch_size": 512,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "early_stop_patience": 10,
        "input_len": 48,
        "time_freq": "5min",
        "stride": 12,
        "train_sample_len": 3000,
        "val_sample_len": 1000 
    }
    
    wrapper = TFTWrapper(BASE_CONFIG)
    
    datasets_to_run = ['op', 're', 'lg']
    pred_lens = [6, 12]
    
    all_results = []
    
    for ds in datasets_to_run:
        for p_len in pred_lens:
            print(f"\\n=== Testing Dataset: {ds.upper()} | Horizon: {p_len*5}min ===")
            model = wrapper.train(ds, p_len)
            if model is not None:
                summary = wrapper.evaluate(model, ds, p_len)
                all_results.append(summary)
                pd.DataFrame([summary]).to_csv(os.path.join(current_dir, f"result_{ds}_{p_len}.csv"), index=False)
                
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv_path = os.path.join(current_dir, "tft_experiment_summary_all.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print("\\n=== TSFMs TFT benchmark test completed ===")
        print(summary_df)
