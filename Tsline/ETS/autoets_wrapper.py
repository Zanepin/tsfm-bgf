import os
import sys
import warnings
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from darts import TimeSeries
from darts.models import AutoETS
from darts.dataprocessing.transformers import Scaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.base_wrapper import BaseModelWrapper
from utils.metrics import calculate_all_metrics

warnings.filterwarnings("ignore")
import logging
logging.getLogger("darts").setLevel(logging.WARNING)
logging.getLogger("statsforecast").setLevel(logging.WARNING)

class AutoETSWrapper(BaseModelWrapper):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_root = self.config.get("data_root", "/mnt/d/glucose_data/Internal")

    def train(self, dataset_type: str, pred_len: int) -> str:
        # AutoETS fits per-subject at evaluation time.
        return "autoets_dummy_path"

    def _group_files_by_subject(self, folder_path: str) -> Dict[str, List[Path]]:
        groups = defaultdict(list)
        path_obj = Path(folder_path)
        if not path_obj.exists(): return groups
        for f in path_obj.glob("*.csv"):
            sid = f.name.split('_')[0]
            groups[sid].append(f)
        return groups

    def _load_and_scale_series(self, csv_path: Path, max_len=None):
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 50: return None, None
            vals = df.iloc[:, 0].values.astype("float32")
            if max_len and len(vals) > max_len: vals = vals[-max_len:]
            times = pd.date_range(start="2024-01-01", periods=len(vals), freq="5min")
            ts = TimeSeries.from_times_and_values(times, vals)
            scaler = Scaler()
            ts_scaled = scaler.fit_transform(ts)
            return ts_scaled, scaler
        except:
            return None, None

    def _create_windows_np(self, values: np.ndarray, context_len, pred_len, stride):
        windows = []
        total_len = context_len + pred_len
        if len(values) < total_len: return []
        for i in range(0, len(values) - total_len + 1, stride):
            windows.append(values[i: i + total_len])
        return windows

    def evaluate(self, model_path: str, dataset_type: str, pred_len: int) -> Dict[str, Any]:
        self.config["pred_len"] = pred_len
        train_dir = os.path.join(self.data_root, f"{dataset_type}_split2", "Train")
        test_dir = os.path.join(self.data_root, f"{dataset_type}_split2", "Test")

        train_groups = self._group_files_by_subject(train_dir)
        test_groups = self._group_files_by_subject(test_dir)
        common_ids = sorted(list(set(train_groups.keys()) & set(test_groups.keys())))

        context_len = self.config["data"]["context_len"]
        stride = self.config["data"]["stride"]
        mc = self.config["model"]

        metrics_storage = defaultdict(list)
        detailed_results = []
        subject_runtimes = []

        for sid in common_ids:
            train_files = train_groups[sid]
            test_files = test_groups[sid]
            if not train_files: continue

            best_train_file = max(train_files, key=lambda f: f.stat().st_size)
            train_ts_scaled, _ = self._load_and_scale_series(best_train_file, max_len=mc["train_limit"])
            if train_ts_scaled is None: continue

            model = AutoETS(season_length=mc["season_length"], model=mc["model"], damped=mc["damped"])
            try:
                model.fit(train_ts_scaled)
            except:
                continue

            all_preds, all_trues = [], []
            t_start = time.time()

            for t_file in test_files:
                try:
                    df = pd.read_csv(t_file)
                    if len(df) < context_len + pred_len: continue
                    vals = df.iloc[:, 0].values.astype("float32")
                    windows = self._create_windows_np(vals, context_len, pred_len, stride)
                    
                    for win in windows:
                        history_arr = win[:context_len]
                        true_future = win[context_len:]

                        local_scaler = Scaler()
                        hist_ts = TimeSeries.from_times_and_values(
                            pd.date_range("2000-01-01", periods=context_len, freq="5min"), history_arr
                        )
                        hist_ts_scaled = local_scaler.fit_transform(hist_ts)

                        pred_ts_scaled = model.predict(n=pred_len, series=hist_ts_scaled)
                        pred_ts = local_scaler.inverse_transform(pred_ts_scaled)

                        all_preds.append(pred_ts.values().flatten())
                        all_trues.append(true_future)
                except:
                    continue

            inference_time = time.time() - t_start
            
            if not all_preds: continue
            
            flat_preds = np.concatenate(all_preds)
            flat_trues = np.concatenate(all_trues)

            mask = ~np.isnan(flat_trues) & ~np.isnan(flat_preds)
            flat_trues, flat_preds = flat_trues[mask], flat_preds[mask]

            if len(flat_trues) < 2: continue

            subject_runtimes.append(inference_time)
            metrics = calculate_all_metrics(flat_trues, flat_preds)

            metrics_storage["MAE"].append(metrics["MAE"])
            metrics_storage["RMSE"].append(metrics["RMSE"])
            metrics_storage["R2"].append(metrics["R2"])
            metrics_storage["CEA_AB"].append(metrics["CEA_AB"])
            metrics_storage["CEA_CDE"].append(metrics["CEA_CDE"])

            detailed_results.append({
                "Subject_ID": sid,
                "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"],
                "CEA_AB": metrics["CEA_AB"], "Time": inference_time
            })

        out_csv = os.path.join(current_dir, f"autoets_subject_metrics_{dataset_type}_pred{pred_len}.csv")
        pd.DataFrame(detailed_results).to_csv(out_csv, index=False)

        return self.format_metrics_summary(dataset_type, pred_len, metrics_storage, subject_runtimes)

if __name__ == "__main__":
    BASE_CONFIG = {
        "data_root": "/mnt/d/glucose_data/Internal",
        "model": {
            "model": "ZZN",
            "season_length": 1,
            "damped": None,
            "train_limit": 2000
        },
        "data": {
            "freq": "5min",
            "context_len": 48,
            "stride": 12
        }
    }

    wrapper = AutoETSWrapper(BASE_CONFIG)
    datasets = ["op", "re"]
    pred_lengths = [6, 12]
    all_results = []

    for ds in datasets:
        for p_len in pred_lengths:
            print(f"\\n=== Testing Dataset: {ds.upper()} | Horizon: {p_len*5}min ===")
            model_path = wrapper.train(ds, p_len)
            summary = wrapper.evaluate(model_path, ds, p_len)
            if summary:
                all_results.append(summary)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(current_dir, "autoets_experiment_summary_all.csv"), index=False)
        print("\\n=== TSFMs AutoETS benchmark test completed ===")
        print(summary_df)
