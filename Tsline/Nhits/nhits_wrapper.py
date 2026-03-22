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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.base_wrapper import BaseModelWrapper
from utils.metrics import calculate_all_metrics

from darts import TimeSeries
from darts.models import NHiTSModel
from darts.utils.callbacks import TFMProgressBar

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')

class NHITSWrapper(BaseModelWrapper):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_root = self.config.get("data_root", "/mnt/d/glucose_data/Internal")

    def load_dataset_as_series(self, folder_path: str, max_len: int = None) -> list:
        folder = Path(folder_path)
        if not folder.exists(): return []

        csv_files = sorted(list(folder.glob("*.csv")))
        series_list = []

        for i, f in enumerate(csv_files):
            try:
                df = pd.read_csv(f)
                vals = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                if len(vals) < (self.config['input_len'] + 12): continue
                if max_len is not None and len(vals) > max_len: vals = vals[:max_len]

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
        model_name = f"nhits_{dataset_type}_pred{pred_len}_deterministic"
        model_dir = os.path.join(current_dir, "saved_models", model_name)
        os.makedirs(model_dir, exist_ok=True)

        train_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Train")
        val_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Val")

        train_series = self.load_dataset_as_series(train_path, max_len=self.config['train_sample_len'])
        val_series = self.load_dataset_as_series(val_path, max_len=self.config['val_sample_len'])

        if not train_series: return None

        add_encoders = {'cyclic': {'past': ['hour', 'minute']}}
        
        from pytorch_lightning.callbacks import EarlyStopping
        early_stopper = EarlyStopping(monitor="val_loss", patience=self.config['early_stop_patience'], mode="min", verbose=True)

        pl_trainer_kwargs = {
            "accelerator": "auto",
            "devices": 1,
            "enable_model_summary": True,
            "callbacks": [TFMProgressBar(enable_train_bar_only=True), early_stopper],
        }

        # LR Finder
        temp_model = NHiTSModel(
            input_chunk_length=self.config['input_len'],
            output_chunk_length=pred_len,
            num_stacks=self.config['num_stacks'],
            num_blocks=self.config['num_blocks'],
            num_layers=self.config['num_layers'],
            layer_widths=self.config['layer_widths'],
            pooling_kernel_sizes=self.config['pooling_kernel_sizes'],
            n_freq_downsample=self.config['n_freq_downsample'],
            dropout=self.config['dropout'],
            MaxPool1d=self.config['MaxPool1d'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['max_epochs'],
            add_encoders=add_encoders,
            use_reversible_instance_norm=True,
            likelihood=None,
            loss_fn=torch.nn.MSELoss(),
            pl_trainer_kwargs=pl_trainer_kwargs,
            save_checkpoints=False
        )

        try:
            results = temp_model.lr_find(
                series=train_series[:50], val_series=val_series[:20],
                min_lr=1e-5, max_lr=1e-1, num_training=100
            )
            suggested_lr = results.suggestion()
            if suggested_lr is None or not (1e-6 < suggested_lr < 0.1): suggested_lr = 1e-3
        except:
            suggested_lr = 1e-3

        del temp_model
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        model = NHiTSModel(
            input_chunk_length=self.config['input_len'],
            output_chunk_length=pred_len,
            num_stacks=self.config['num_stacks'],
            num_blocks=self.config['num_blocks'],
            num_layers=self.config['num_layers'],
            layer_widths=self.config['layer_widths'],
            pooling_kernel_sizes=self.config['pooling_kernel_sizes'],
            n_freq_downsample=self.config['n_freq_downsample'],
            dropout=self.config['dropout'],
            MaxPool1d=self.config['MaxPool1d'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['max_epochs'],
            add_encoders=add_encoders,
            use_reversible_instance_norm=True,
            likelihood=None,
            loss_fn=torch.nn.MSELoss(),
            optimizer_kwargs={"lr": suggested_lr},
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name=model_name,
            work_dir=model_dir,
            save_checkpoints=True,
            force_reset=True
        )

        model.fit(series=train_series, val_series=val_series, max_samples_per_ts=self.config['max_samples_per_ts'], verbose=True)
        return NHiTSModel.load_from_checkpoint(model_name=model_name, work_dir=model_dir, best=True)

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

                preds = model.predict(n=pred_len, series=batch_inputs, num_samples=1, verbose=False)

                all_pred_vals = []
                all_true_vals = []

                for idx, p_ts in enumerate(preds):
                    p_val = p_ts.values().flatten()
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
                
        out_csv = os.path.join(current_dir, f"nhits_subject_metrics_{dataset_type}_pred{pred_len}.csv")
        pd.DataFrame(detailed_results).to_csv(out_csv, index=False)

        return self.format_metrics_summary(dataset_type, pred_len, metrics_storage, subject_runtimes)

if __name__ == "__main__":
    BASE_CONFIG = {
        "data_root": "/mnt/d/glucose_data/Internal",
        "num_stacks": 3,
        "num_blocks": 1,
        "num_layers": 2,
        "layer_widths": 512,
        "pooling_kernel_sizes": None,
        "n_freq_downsample": None,
        "dropout": 0.1,
        "MaxPool1d": True,
        "batch_size": 1024,
        "max_epochs": 100,
        "early_stop_patience": 15,
        "input_len": 48,
        "time_freq": "5min",
        "stride": 12,
        "train_sample_len": 3000,
        "val_sample_len": 1000,
        "max_samples_per_ts": 150 
    }
    
    wrapper = NHITSWrapper(BASE_CONFIG)
    
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
                
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv_path = os.path.join(current_dir, "nhits_experiment_summary_all.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print("\\n=== TSFMs N-HiTS benchmark test completed ===")
        print(summary_df)
