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
from typing import List, Dict, Any
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.base_wrapper import BaseModelWrapper
from utils.metrics import calculate_all_metrics
from utils.data_io import load_glucose_series

import pytorch_lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.wavenet import WaveNetEstimator
from gluonts.model.predictor import Predictor

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')

try:
    torch.serialization.add_safe_globals([functools.partial])
except Exception:
    pass

class WaveNetWrapper(BaseModelWrapper):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_root = self.config.get("data_root", "/mnt/d/glucose_data/Internal")
        self._set_seed(self.config.get("seed", 42))

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        pl.seed_everything(seed)

    def _create_sliding_windows(self, data: np.ndarray, total_len: int, stride: int) -> List[np.ndarray]:
        sequences = []
        if len(data) < total_len: return sequences
        for i in range(0, len(data) - total_len + 1, stride):
            sequences.append(data[i: i + total_len])
        return sequences

    def _prepare_datasets(self, data_dir: str, is_inference: bool = False) -> tuple:
        context_len = self.config["context_len"]
        pred_len = self.config["prediction_length"]
        stride = self.config["stride"]
        freq = self.config["freq"]
        total_len = context_len + pred_len

        all_sequences = []
        subject_mapping = []
        raw_sequences = []

        path_obj = Path(data_dir)
        if not path_obj.exists(): return None, None, None, None

        csv_files = list(path_obj.glob("*.csv"))

        for csv_file in csv_files:
            raw_data = load_glucose_series(str(csv_file))
            if raw_data is not None:
                seqs = self._create_sliding_windows(raw_data, total_len, stride)
                all_sequences.extend(seqs)
                if is_inference:
                    stem = csv_file.stem
                    sid = stem.split('_')[0] if '_' in stem else stem
                    subject_mapping.extend([sid] * len(seqs))
                    raw_sequences.extend(seqs)

        if not all_sequences: return None, None, None, None

        custom_dataset = np.array(all_sequences)
        start = pd.Timestamp("2024-01-01 00:00:00")
        gluon_ds = ListDataset([{"target": x, "start": start} for x in custom_dataset], freq=freq)
        return gluon_ds, custom_dataset, subject_mapping, raw_sequences

    def train(self, dataset_type: str, pred_len: int) -> str:
        self.config["prediction_length"] = pred_len
        exp_name = f"{dataset_type}_pred{pred_len}_wavenet"
        output_dir = Path(os.path.join(current_dir, self.config.get("output_base", "output_batch_exp"), exp_name))
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        train_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Train")
        val_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Val")

        train_ds, _, _, _ = self._prepare_datasets(train_path, is_inference=False)
        val_ds, _, _, _ = self._prepare_datasets(val_path, is_inference=False)

        if train_ds is None: return ""

        monitor_metric = "val_loss" if val_ds is not None else "train_loss"

        estimator = WaveNetEstimator(
            freq=self.config["freq"],
            prediction_length=self.config["prediction_length"],
            num_bins=self.config["num_bins"],
            num_residual_channels=self.config["num_residual_channels"],
            num_skip_channels=self.config["num_skip_channels"],
            dilation_depth=self.config["dilation_depth"],
            num_stacks=self.config["num_stacks"],
            temperature=self.config["temperature"],
            num_feat_dynamic_real=0,
            num_feat_static_cat=0,
            num_feat_static_real=0,
            cardinality=[1],
            embedding_dimension=self.config["embedding_dimension"],
            use_log_scale_feature=self.config["use_log_scale_feature"],
            num_parallel_samples=self.config["num_parallel_samples"],
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            batch_size=self.config["batch_size"],
            num_batches_per_epoch=self.config["num_batches_per_epoch"],
            negative_data=self.config["negative_data"],
            trainer_kwargs={
                "max_epochs": self.config["epochs"],
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": [0] if torch.cuda.is_available() else "auto",
                "enable_progress_bar": False,
                "logger": False,
                "enable_checkpointing": True,
                "callbacks": [EarlyStopping(monitor=monitor_metric, patience=self.config["patience"], mode="min")]
            }
        )

        predictor = estimator.train(training_data=train_ds, validation_data=val_ds)
        predictor.serialize(models_dir)
        return str(models_dir)

    def evaluate(self, model_path: str, dataset_type: str, pred_len: int) -> Dict[str, Any]:
        self.config["prediction_length"] = pred_len
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        predictor = Predictor.deserialize(Path(model_path))

        test_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Test")
        test_ds_flat, _, subject_mapping, raw_sequences = self._prepare_datasets(test_path, is_inference=True)

        if test_ds_flat is None: return {}

        grouped_data = defaultdict(list)
        grouped_raw = defaultdict(list)
        for i, entry in enumerate(test_ds_flat):
            sid = subject_mapping[i]
            grouped_data[sid].append(entry)
            grouped_raw[sid].append(raw_sequences[i])

        metrics_storage = defaultdict(list)
        detailed_results = []
        subject_runtimes = []

        if torch.cuda.is_available():
            dummy = torch.randn(1, self.config["num_residual_channels"], 10).to("cuda")
            torch.cuda.synchronize()

        for sid, entries in grouped_data.items():
            sub_ds = ListDataset(entries, freq=self.config["freq"])

            t_start = time.time()
            forecast_it = predictor.predict(sub_ds, num_samples=100)
            forecasts = list(forecast_it)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            runtime = time.time() - t_start
            subject_runtimes.append(runtime)

            sub_preds, sub_trues = [], []
            for i, forecast in enumerate(forecasts):
                sub_preds.extend(forecast.mean)
                sub_trues.extend(grouped_raw[sid][i][-pred_len:])

            flat_pred = np.array(sub_preds)
            flat_true = np.array(sub_trues)

            if np.isnan(flat_pred).any():
                flat_pred = np.nan_to_num(flat_pred, nan=np.nanmean(flat_true))

            metrics = calculate_all_metrics(flat_true, flat_pred)

            metrics_storage["MAE"].append(metrics["MAE"])
            metrics_storage["RMSE"].append(metrics["RMSE"])
            metrics_storage["R2"].append(metrics["R2"])
            metrics_storage["CEA_AB"].append(metrics["CEA_AB"])
            metrics_storage["CEA_CDE"].append(metrics["CEA_CDE"])

            detailed_results.append({
                "Subject_ID": sid,
                "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"],
                "CEA_AB": metrics["CEA_AB"]
            })

        out_csv = os.path.join(current_dir, f"wavenet_subject_metrics_{dataset_type}_pred{pred_len}.csv")
        pd.DataFrame(detailed_results).to_csv(out_csv, index=False)

        return self.format_metrics_summary(dataset_type, pred_len, metrics_storage, subject_runtimes)

if __name__ == "__main__":
    BASE_CONFIG = {
        "data_root": "/mnt/d/glucose_data/Internal",
        "freq": "5min",
        "context_len": 48,
        "stride": 12,
        "dilation_depth": 6,
        "num_stacks": 1,
        "num_bins": 256,
        "num_residual_channels": 48,
        "num_skip_channels": 64,
        "temperature": 1.0,
        "embedding_dimension": 5,
        "use_log_scale_feature": True,
        "num_parallel_samples": 100,
        "lr": 4e-3,
        "weight_decay": 1e-4,
        "negative_data": False,
        "batch_size": 1024,
        "epochs": 100,
        "patience": 15,
        "num_batches_per_epoch": 1000,
        "seed": 42,
        "output_base": "output_batch_exp"
    }

    wrapper = WaveNetWrapper(BASE_CONFIG)
    datasets = ["op", "re", "lg"]
    pred_lengths = [6, 12]
    
    all_results = []
    
    for ds in datasets:
        for p_len in pred_lengths:
            print(f"\\n=== Testing Dataset: {ds.upper()} | Horizon: {p_len*5}min ===")
            model_path = wrapper.train(ds, p_len)
            if model_path:
                summary = wrapper.evaluate(model_path, ds, p_len)
                all_results.append(summary)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(current_dir, "wavenet_experiment_summary_all.csv"), index=False)
        print("\\n=== TSFMs WaveNet benchmark test completed ===")
        print(summary_df)
