import os
import sys
import warnings
import torch
import numpy as np
import pandas as pd
import yaml
import functools
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Tuple

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.base_wrapper import BaseModelWrapper
from utils.metrics import calculate_all_metrics
from utils.data_io import load_glucose_series

import pytorch_lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.torch.distributions import NormalOutput
from gluonts.model.predictor import Predictor

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')

try:
    torch.serialization.add_safe_globals([functools.partial])
except Exception:
    pass

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

class RevIN:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def normalize_window(self, x: np.ndarray, context_len: int) -> Tuple[np.ndarray, float, float]:
        context_data = x[:context_len]
        mean = np.nanmean(context_data)
        std = np.nanstd(context_data)
        if std < self.eps or np.isnan(std): std = 1.0
        if np.isnan(mean): mean = 0.0
        return (x - mean) / std, mean, std

    def normalize_series(self, x: np.ndarray) -> np.ndarray:
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std < self.eps or np.isnan(std): std = 1.0
        if np.isnan(mean): mean = 0.0
        return (x - mean) / std

    def denormalize(self, x_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
        return x_norm * std + mean

class PatchTSTWrapper(BaseModelWrapper):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.revin = RevIN()
        self.data_root = "/mnt/d/glucose_data/Internal"
        set_seed(self.config["system"].get("seed", 42))

    def create_train_val_dataset(self, data_dir: str) -> ListDataset:
        freq = self.config["data"]["freq"]
        path_obj = Path(data_dir)
        if not path_obj.exists(): return None

        dataset_list = []
        csv_files = list(path_obj.glob("*.csv"))

        for csv_file in csv_files:
            values = load_glucose_series(str(csv_file))
            min_len = self.config["data"]["context_len"] + self.config["data"]["prediction_length"]
            if values is not None and len(values) > min_len:
                values_norm = self.revin.normalize_series(values)
                start = pd.Timestamp("2024-01-01 00:00:00")
                dataset_list.append({"target": values_norm, "start": start})

        if not dataset_list: return None
        return ListDataset(dataset_list, freq=freq)

    def create_test_dataset(self, data_dir: str):
        path_obj = Path(data_dir)
        csv_files = list(path_obj.glob("*.csv"))

        context_len = self.config["data"]["context_len"]
        pred_len = self.config["data"]["prediction_length"]
        total_len = context_len + pred_len
        stride = 12

        all_sequences = []
        revin_stats = []
        subject_mapping = []

        for csv_file in csv_files:
            raw_data = load_glucose_series(str(csv_file))
            if raw_data is None or len(raw_data) < total_len: continue

            sid = csv_file.stem.split('_')[0]
            for i in range(0, len(raw_data) - total_len + 1, stride):
                window = raw_data[i: i + total_len]
                norm_window, mean, std = self.revin.normalize_window(window, context_len)
                all_sequences.append(norm_window)
                revin_stats.append((mean, std))
                subject_mapping.append(sid)

        if not all_sequences: return None, None, None, None

        gluon_ds = ListDataset(
            [{"target": x, "start": pd.Timestamp("2024-01-01")} for x in all_sequences],
            freq=self.config["data"]["freq"]
        )
        return gluon_ds, np.array(all_sequences), revin_stats, subject_mapping

    def train(self, dataset_type: str, pred_len: int) -> str:
        self.config["data"]["prediction_length"] = pred_len
        self.config["model"]["prediction_length"] = pred_len
        
        train_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Train")
        val_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Val")

        train_ds = self.create_train_val_dataset(train_path)
        val_ds = self.create_train_val_dataset(val_path)

        if train_ds is None: raise ValueError("No train data")

        exp_name = f"{dataset_type}_p{pred_len}_ctx{self.config['data']['context_len']}_fullseq"
        out_dir = Path(os.path.join(current_dir, self.config["system"]["output_base"], exp_name))
        model_dir = out_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        mc, tc, dc = self.config["model"], self.config["training"], self.config["data"]

        estimator = PatchTSTEstimator(
            prediction_length=mc["prediction_length"],
            context_length=dc["context_len"],
            patch_len=mc["patch_len"],
            stride=mc["stride"],
            d_model=mc["d_model"],
            nhead=mc["nhead"],
            dim_feedforward=mc["dim_feedforward"],
            num_encoder_layers=mc["num_encoder_layers"],
            dropout=mc["dropout"],
            scaling=None,
            distr_output=NormalOutput(),
            lr=mc["lr"],
            weight_decay=mc["weight_decay"],
            batch_size=tc["batch_size"],
            num_batches_per_epoch=tc["num_batches_per_epoch"],
            trainer_kwargs={
                "max_epochs": tc["epochs"],
                "accelerator": "gpu",
                "devices": [0],
                "enable_progress_bar": False,
                "logger": False,
                "enable_checkpointing": True,
                "gradient_clip_val": 1.0,
                "callbacks": [
                    EarlyStopping(monitor="val_loss", patience=tc["patience"], mode="min")
                ]
            }
        )

        predictor = estimator.train(training_data=train_ds, validation_data=val_ds)
        predictor.serialize(model_dir)

        with open(model_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

        return str(model_dir)

    def evaluate(self, model_path: str, dataset_type: str, pred_len: int) -> Dict[str, Any]:
        self.config["data"]["prediction_length"] = pred_len
        predictor = Predictor.deserialize(Path(model_path))

        test_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Test")
        test_ds, raw_norm_seqs, revin_stats, sids = self.create_test_dataset(test_path)

        if test_ds is None: return {}

        grouped_indices = defaultdict(list)
        for idx, sid in enumerate(sids):
            grouped_indices[sid].append(idx)

        grouped_true = defaultdict(list)
        grouped_pred = defaultdict(list)
        metrics_log = defaultdict(list)
        subject_runtimes = []

        for sid, indices in grouped_indices.items():
            sub_samples = [test_ds[i] for i in indices]
            sub_ds = ListDataset(sub_samples, freq=self.config["data"]["freq"])

            t_start = time.time()
            forecasts = list(predictor.predict(sub_ds, num_samples=100))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            subject_runtimes.append(time.time() - t_start)

            for i, forecast in enumerate(forecasts):
                global_idx = indices[i]
                mean, std = revin_stats[global_idx]

                pred_real = self.revin.denormalize(forecast.mean, mean, std)
                true_norm = raw_norm_seqs[global_idx][-pred_len:]
                true_real = self.revin.denormalize(true_norm, mean, std)

                grouped_pred[sid].extend(pred_real)
                grouped_true[sid].extend(true_real)

        detailed_rows = []
        for sid in grouped_true:
            y_true = grouped_true[sid]
            y_pred = grouped_pred[sid]
            
            mets = calculate_all_metrics(y_true, y_pred)
            
            metrics_log["MAE"].append(mets["MAE"])
            metrics_storage = metrics_log  # pointer
            metrics_storage["RMSE"].append(mets["RMSE"])
            metrics_storage["R2"].append(mets["R2"])
            metrics_storage["CEA_AB"].append(mets["CEA_AB"])
            metrics_storage["CEA_CDE"].append(mets["CEA_CDE"])

            detailed_rows.append({
                "Subject": sid,
                "MAE": mets["MAE"], "RMSE": mets["RMSE"], "R2": mets["R2"],
                "CEA_AB": mets["CEA_AB"]
            })

        pd.DataFrame(detailed_rows).to_csv(Path(model_path) / "subject_metrics.csv", index=False)
        return self.format_metrics_summary(dataset_type, pred_len, metrics_storage, subject_runtimes)

if __name__ == "__main__":
    BASE_CONFIG = {
        "model": {
            "patch_len": 12, "stride": 12, "d_model": 64, "nhead": 4,
            "num_encoder_layers": 3, "dim_feedforward": 256,
            "dropout": 0.1, "lr": 5e-5, "weight_decay": 1e-4,
        },
        "data": {
            "freq": "5min", "context_len": 48
        },
        "training": {
            "batch_size": 1024, "epochs": 100, "patience": 15,
            "num_batches_per_epoch": 1500, "device": "cuda:0"
        },
        "system": {
            "seed": 42, "output_base": "output_patchtst_revin"
        }
    }
    
    wrapper = PatchTSTWrapper(BASE_CONFIG)
    
    datasets = ["op", "re", "lg"]
    pred_lengths = [6, 12]
    master_csv = "patchtst_experiment_summary_all.csv"
    results = []

    for ds in datasets:
        for pl_len in pred_lengths:
            print(f"\\n=== Testing Dataset: {ds.upper()} | Horizon: {pl_len*5}min ===")
            try:
                model_path = wrapper.train(ds, pl_len)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                summary = wrapper.evaluate(model_path, ds, pl_len)
                results.append(summary)
            except Exception as e:
                print(f"Failed {ds}-{pl_len}: {e}")

    if results:
        pd.DataFrame(results).to_csv(master_csv, index=False)
        print("\\n=== TSFMs PatchTST benchmark test completed ===")
        print(pd.DataFrame(results))
