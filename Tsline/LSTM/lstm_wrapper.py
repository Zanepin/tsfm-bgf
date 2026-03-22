import os
import sys
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.base_wrapper import BaseModelWrapper
from utils.metrics import calculate_all_metrics
from utils.data_io import load_glucose_series

warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ==== Model & Data Classes ====
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        return self.mean, self.stdev

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * 1e-8)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

class LSTMWithRevIN(nn.Module):
    def __init__(self, config):
        super(LSTMWithRevIN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.pred_len = config['pred_len']

        self.revin = RevIN(num_features=1, affine=config['revin_affine'])
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )
        self.fc = nn.Linear(self.hidden_size, self.pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        out = out.unsqueeze(-1)
        out = self.revin(out, 'denorm')
        return out.squeeze(-1)

class GlucoseDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "x": torch.tensor(seq[0], dtype=torch.float32).unsqueeze(-1),
            "y": torch.tensor(seq[1], dtype=torch.float32)
        }

# ==== Wrapper Implementation ====
class LSTMWrapper(BaseModelWrapper):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_root = "/mnt/d/glucose_data/Internal"
        set_seed(self.config.get("random_seed", 42))

    def _create_flat_windows(self, data_dir: str):
        ctx_len, pred_len, stride = self.config['context_len'], self.config['pred_len'], self.config['stride']
        total_len = ctx_len + pred_len
        all_pairs = []

        path_obj = Path(data_dir)
        if not path_obj.exists(): return []
        csv_files = list(path_obj.glob("*.csv"))

        for f in csv_files:
            raw = load_glucose_series(str(f))
            if raw is None or len(raw) < total_len: continue
            n_samples = (len(raw) - total_len) // stride + 1
            for i in range(n_samples):
                start = i * stride
                window = raw[start: start + total_len]
                all_pairs.append((window[:ctx_len], window[ctx_len:]))
        return all_pairs

    def _get_loader(self, dataset_type: str, mode='train'):
        dir_path = os.path.join(self.data_root, f"{dataset_type}_split2", "Train" if mode == 'train' else "Val")
        pairs = self._create_flat_windows(dir_path)
        if not pairs:
            if mode == 'train': raise ValueError(f"No train data in {dir_path}")
            else: print(f"Warning: No val data in {dir_path}")

        return DataLoader(
            GlucoseDataset(pairs),
            batch_size=self.config['batch_size'],
            shuffle=(mode == 'train'),
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

    def train(self, dataset_type: str, pred_len: int) -> str:
        self.config['pred_len'] = pred_len
        exp_name = f"{dataset_type}_p{pred_len}_lstm_revin"
        out_dir = Path(os.path.join(current_dir, self.config['output_base'], exp_name))
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "best_model.pth"

        train_loader = self._get_loader(dataset_type, 'train')
        val_loader = self._get_loader(dataset_type, 'val')

        model = LSTMWithRevIN(self.config).to(self.config['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'])
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_cnt = 0

        for epoch in range(self.config['epochs']):
            model.train()
            train_losses = []
            for batch in train_loader:
                x = batch['x'].to(self.config['device'])
                y = batch['y'].to(self.config['device'])

                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = avg_train_loss

            if val_loader is not None and len(val_loader) > 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['x'].to(self.config['device'])
                        y = batch['y'].to(self.config['device'])
                        pred = model(x)
                        v_loss = criterion(pred, y)
                        val_losses.append(v_loss.item())
                avg_val_loss = np.mean(val_losses)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_cnt = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_cnt += 1

            if patience_cnt >= self.config['patience']:
                break

        return str(model_path)

    def evaluate(self, model_path: str, dataset_type: str, pred_len: int) -> Dict[str, Any]:
        self.config['pred_len'] = pred_len
        
        test_dir = os.path.join(self.data_root, f"{dataset_type}_split2", "Test")
        ctx_len, stride = self.config['context_len'], self.config['stride']
        total_len = ctx_len + pred_len

        subject_files = defaultdict(list)
        path_obj = Path(test_dir)
        for f in path_obj.glob("*.csv"):
            sid = f.name.split('_')[0]
            subject_files[sid].append(f)

        subject_data = {}
        for sid, files in subject_files.items():
            pairs = []
            for f in files:
                raw = load_glucose_series(str(f))
                if raw is None or len(raw) < total_len: continue
                n_samples = (len(raw) - total_len) // stride + 1
                for i in range(n_samples):
                    start = i * stride
                    window = raw[start: start + total_len]
                    pairs.append((window[:ctx_len], window[ctx_len:]))
            if pairs: subject_data[sid] = pairs

        model = LSTMWithRevIN(self.config).to(self.config['device'])
        model.load_state_dict(torch.load(model_path))
        model.eval()

        metrics_storage = defaultdict(list)
        subject_runtimes = []
        detailed_rows = []

        for sid, data_list in subject_data.items():
            xs = np.array([d[0] for d in data_list])
            ys = np.array([d[1] for d in data_list])

            x_tensor = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1).to(self.config['device'])

            t0 = time.time()
            with torch.no_grad():
                pred_tensor = model(x_tensor)
            subject_runtimes.append(time.time() - t0)

            y_pred = pred_tensor.cpu().numpy().flatten()
            y_true = ys.flatten()

            mets = calculate_all_metrics(y_true, y_pred)
            
            metrics_storage["MAE"].append(mets["MAE"])
            metrics_storage["RMSE"].append(mets["RMSE"])
            metrics_storage["R2"].append(mets["R2"])
            metrics_storage["CEA_AB"].append(mets["CEA_AB"])
            metrics_storage["CEA_CDE"].append(mets["CEA_CDE"])

            detailed_rows.append({
                "Subject": sid,
                "MAE": mets["MAE"], "RMSE": mets["RMSE"], "R2": mets["R2"],
                "CEA_AB": mets["CEA_AB"]
            })

        out_csv = os.path.join(os.path.dirname(model_path), f"subject_metrics_{dataset_type}_pred{pred_len}.csv")
        pd.DataFrame(detailed_rows).to_csv(out_csv, index=False)

        return self.format_metrics_summary(dataset_type, pred_len, metrics_storage, subject_runtimes)

if __name__ == "__main__":
    BASE_CONFIG = {
        "context_len": 48,
        "pred_len": 12,
        "stride": 12,
        "batch_size": 128,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 60,
        "lr": 1e-4,
        "patience": 10,
        "revin_affine": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,
        "num_workers": 4,
        "output_base": "output_lstm_revin_val_stop"
    }

    wrapper = LSTMWrapper(BASE_CONFIG)
    
    datasets = ["op", "re", "lg"]
    pred_lengths = [6, 12]
    master_csv = "lstm_experiment_summary_all.csv"
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
        pd.DataFrame(results).to_csv(os.path.join(current_dir, master_csv), index=False)
        print("\\n=== TSFMs LSTM benchmark test completed ===")
        print(pd.DataFrame(results))
