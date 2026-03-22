import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import logging
import time
from sklearn.metrics import r2_score
from model import TimeMoEModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid


class TimeMoEEvaluator:
    def __init__(
            self,
            model_path: str = None,
            use_finetuned: bool = False,
            pred_len: int = 12,
            context_len: int = 48,
            val_dir: str = None,
            dataset: str = "op",
            output_dir: str = None,
            device: str = "cuda",
    ):
        self.model_path = model_path
        self.use_finetuned = use_finetuned
        self.context_len = context_len
        self.pred_len = pred_len
        self.dataset = dataset
        self.device = device
        
        if val_dir is None:
            val_dir = f"/mnt/d/glucose_data/internal/{dataset}_split2/test"
        self.val_dir = Path(val_dir)
        
        if output_dir is None:
            output_dir = f"./output/{dataset}_pred{pred_len}"
        self.output_dir = Path(output_dir)
        
        logger.info(f"Evaluator initialized:")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Validation directory: {self.val_dir}")
        logger.info(f"  Context length: {context_len}")
        logger.info(f"  Prediction length: {pred_len}")
        
        self.load_model()
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load metadata"""
        metadata_path = self.output_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found: {metadata_path}")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata:")
        logger.info(f"  Model type: {metadata.get('model_type')}")
        logger.info(f"  Subjects: {metadata.get('num_val_subjects')}")
        
        return metadata
    
    def load_model(self):
        """Load the model"""
        if self.use_finetuned and self.model_path and os.path.exists(self.model_path):
            self.model = TimeMoEModel(
                model_path=self.model_path,
                device=self.device,
            )
            self.model_name = f"Fine-tuned-TimeMoE-{os.path.basename(self.model_path)}"
            logger.info(f"Loaded fine-tuned model from {self.model_path}")
        else:
            self.model = TimeMoEModel(
                model_path="Maple728/TimeMoE-200M",
                device=self.device,
            )
            self.model_name = "Pre-trained-TimeMoE-200M"
            logger.info("Loaded pre-trained TimeMoE-200M model")
    
    @staticmethod
    def _extract_subject_id(filename: str) -> str:
        """Extract subject ID from filename"""
        parts = filename.split('_')
        return parts[0] if parts else filename
    
    @staticmethod
    def mae(pred: np.ndarray, true: np.ndarray) -> float:
        return np.mean(np.abs(pred - true))
    
    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return np.sqrt(np.mean((pred - true) ** 2))
    
    @staticmethod
    def r2(pred: np.ndarray, true: np.ndarray) -> float:
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        if len(pred_flat) != len(true_flat):
            logger.warning("R2 calculation: shape mismatch")
            return np.nan
        return r2_score(true_flat, pred_flat)
    
    def load_csv_files_by_subjects(self) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
        """Load evaluation data by subject"""
        if not self.val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {self.val_dir}")
        
        logger.info(f"Loading evaluation data from {self.val_dir}")
        
        csv_files = sorted(list(self.val_dir.glob("*.csv")), key=lambda x: x.stem)
        logger.info(f"Found {len(csv_files)} CSV files")
        
        subjects_data = {}
        total_len = self.context_len + self.pred_len
        stride = 12
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'GlucoseValue' not in df.columns:
                    logger.warning(f"File {csv_file.name}: 'GlucoseValue' not found")
                    continue
                
                glucose_values = df['GlucoseValue'].values
                glucose_values = glucose_values[~np.isnan(glucose_values)]
                
                if len(glucose_values) < total_len:
                    continue
                
                series = glucose_values.astype(np.float32)
                subject_id = self._extract_subject_id(csv_file.stem)
                
                if subject_id not in subjects_data:
                    subjects_data[subject_id] = ([], [])
                
                max_start_idx = len(series) - total_len
                for i in range(0, max_start_idx + 1, stride):
                    context = series[i:i + self.context_len]
                    target = series[i + self.context_len:i + total_len]
                    subjects_data[subject_id][0].append(context)
                    subjects_data[subject_id][1].append(target)
            
            except Exception as e:
                logger.warning(f"Error loading {csv_file.name}: {e}")
                continue
        
        total_windows = sum(len(contexts) for contexts, _ in subjects_data.values())
        logger.info(f"Data extraction complete:")
        logger.info(f"  Total subjects: {len(subjects_data)}")
        logger.info(f"  Total windows: {total_windows}")
        
        return subjects_data
    
    def evaluate_subject(self, contexts: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate a single subject"""
        if not contexts:
            return {
                'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
                'CEA_AB': np.nan, 'CEA_CDE': np.nan,
                'inference_time_median': np.nan, 'num_samples': 0
            }
        
        predictions = []
        inference_times = []
        batch_size = 128
        
        for batch_idx in range(0, len(contexts), batch_size):
            batch_contexts = contexts[batch_idx:batch_idx + batch_size]
            
            try:
                start_time = time.perf_counter()
                
                batch_preds = self.model.predict(
                    context=batch_contexts,
                    prediction_length=self.pred_len,
                )
                
                end_time = time.perf_counter()
                batch_time = end_time - start_time
                inference_times.append(batch_time)
                
                if batch_preds.ndim == 1:
                    batch_preds = batch_preds.reshape(1, -1)
                
                if batch_preds.shape[1] != self.pred_len:
                    if batch_preds.shape[1] > self.pred_len:
                        batch_preds = batch_preds[:, :self.pred_len]
                    else:
                        pad_w = self.pred_len - batch_preds.shape[1]
                        batch_preds = np.pad(batch_preds, ((0, 0), (0, pad_w)), mode='edge')
                
                predictions.extend([row for row in batch_preds])
            
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                continue
        
        if not predictions:
            return {
                'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
                'CEA_AB': np.nan, 'CEA_CDE': np.nan,
                'inference_time_median': np.nan, 'num_samples': 0
            }
        
        predictions = np.array(predictions)
        targets = np.array(targets[:len(predictions)])
        
        if predictions.shape != targets.shape:
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]
        
        mae = self.mae(predictions, targets)
        rmse = self.rmse(predictions, targets)
        r2 = self.r2(predictions, targets)
        
        cea_analyzer = ClarkeErrorGrid()
        cea_result = cea_analyzer.run(targets.flatten(), predictions.flatten())
        cea_ab = cea_result.get('AB_percentage', np.nan)
        cea_cde = cea_result.get('CDE_percentage', np.nan)
        
        inference_time_median = np.median(inference_times) if inference_times else np.nan
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CEA_AB': cea_ab,
            'CEA_CDE': cea_cde,
            'inference_time_median': inference_time_median,
            'num_samples': len(predictions)
        }
    
    def evaluate(self) -> dict:
        """Evaluate all subjects"""
        subjects_data = self.load_csv_files_by_subjects()
        
        if not subjects_data:
            raise ValueError("No evaluation data available")
        
        logger.info(f"Starting evaluation on {len(subjects_data)} subjects")
        
        subject_results = {}
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        cea_ab_scores = []
        cea_cde_scores = []
        inference_times = []
        total_samples = 0
        
        for subject_id in sorted(subjects_data.keys()):
            contexts, targets = subjects_data[subject_id]
            logger.info(f"Evaluating subject {subject_id} ({len(contexts)} windows)...")
            
            result = self.evaluate_subject(contexts, targets)
            subject_results[subject_id] = result
            
            if not np.isnan(result['MAE']):
                mae_scores.append(result['MAE'])
                rmse_scores.append(result['RMSE'])
                r2_scores.append(result['R2'])
                total_samples += result['num_samples']
                
                if not np.isnan(result['CEA_AB']):
                    cea_ab_scores.append(result['CEA_AB'])
                    cea_cde_scores.append(result['CEA_CDE'])
                
                if not np.isnan(result['inference_time_median']):
                    inference_times.append(result['inference_time_median'])
            
            logger.info(
                f"Subject {subject_id}: MAE={result['MAE']:.6f}, "
                f"RMSE={result['RMSE']:.6f}, R2={result['R2']:.6f}, "
                f"CEA_AB={result['CEA_AB']:.2f}%")
        
        return {
            'MAE_mean': np.mean(mae_scores) if mae_scores else np.nan,
            'MAE_std': np.std(mae_scores) if mae_scores else np.nan,
            'RMSE_mean': np.mean(rmse_scores) if rmse_scores else np.nan,
            'RMSE_std': np.std(rmse_scores) if rmse_scores else np.nan,
            'R2_mean': np.mean(r2_scores) if r2_scores else np.nan,
            'R2_std': np.std(r2_scores) if r2_scores else np.nan,
            'CEA_AB_mean': np.mean(cea_ab_scores) if cea_ab_scores else np.nan,
            'CEA_AB_std': np.std(cea_ab_scores) if cea_ab_scores else np.nan,
            'CEA_CDE_mean': np.mean(cea_cde_scores) if cea_cde_scores else np.nan,
            'CEA_CDE_std': np.std(cea_cde_scores) if cea_cde_scores else np.nan,
            'inference_time_median': np.median(inference_times) if inference_times else np.nan,
            'num_subjects': len(mae_scores),
            'total_samples': total_samples,
            'context_length': self.context_len,
            'prediction_length': self.pred_len,
            'model_name': self.model_name,
            'model_type': 'Fine-tuned' if self.use_finetuned else 'Pre-trained',
            'dataset_name': self.dataset,
            'subject_results': subject_results
        }
    
    @staticmethod
    def save_results(eval_results: dict, summary_path: str, subject_path: str):
        """Save evaluation results"""
        main_results = {k: v for k, v in eval_results.items() if k != 'subject_results'}
        df_summary = pd.DataFrame([main_results])
        
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            df_summary = pd.concat([existing_df, df_summary], ignore_index=True)
        
        column_order = [
            'model_type', 'model_name', 'dataset_name',
            'context_length', 'prediction_length', 'num_subjects', 'total_samples',
            'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std',
            'CEA_AB_mean', 'CEA_AB_std', 'CEA_CDE_mean', 'CEA_CDE_std',
            'inference_time_median'
        ]
        
        all_columns = list(df_summary.columns)
        ordered_columns = [col for col in column_order if col in all_columns]
        remaining_columns = [col for col in all_columns if col not in column_order]
        df_summary = df_summary[ordered_columns + remaining_columns]
        
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Summary results saved to {summary_path}")
        
        subject_results = eval_results.get('subject_results', {})
        if subject_results:
            rows = []
            for subject_id in sorted(subject_results.keys()):
                result = subject_results[subject_id]
                rows.append({
                    'id': subject_id,
                    'MAE': result['MAE'],
                    'RMSE': result['RMSE'],
                    'R2': result['R2'],
                    'CEA_AB': result.get('CEA_AB', np.nan),
                    'CEA_CDE': result.get('CEA_CDE', np.nan),
                    'inference_time_median': result.get('inference_time_median', np.nan),
                    'num_samples': result['num_samples']
                })
            
            df_subjects = pd.DataFrame(rows)
            df_subjects.to_csv(subject_path, index=False)
            logger.info(f"Subject-level metrics saved to {subject_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Time-MoE Model Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Evaluate using pre-trained model
  python eval.py --dataset op --pred_len 12
  
  # Evaluate using fine-tuned model
  python eval.py --dataset op --use_finetuned --model_path ./output/op_pred12/checkpoints
  
  # Evaluate on CPU (not recommended, script might terminate automatically if CUDA is unavailable)
  python eval.py --dataset op --pred_len 12 --device cpu
        """
    )
    
    parser.add_argument('--dataset', type=str, choices=["op", "re", "lg"], default='op',
                        help="Dataset selection")
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data CSV directory')
    parser.add_argument('--use_finetuned', action='store_true', default=False,
                        help='Use fine-tuned model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Fine-tuned model checkpoint path')
    parser.add_argument('--context_len', type=int, default=48,
                        help='Input sequence length (default: 48)')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Prediction length')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device selection (default: cuda, terminates if unavailable)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    summary_output_path = f"./eval_results_{args.dataset}_pred{args.pred_len}_timemoe.csv"
    subject_output_path = f"./subject_metrics_{args.dataset}_pred{args.pred_len}_timemoe.csv"
    
    model_path = args.model_path
    if args.use_finetuned and not model_path:
        model_path = f"./output/{args.dataset}_pred{args.pred_len}/checkpoints"
        logger.info(f"Auto-detected checkpoint: {model_path}")
    
    try:
        evaluator = TimeMoEEvaluator(
            model_path=model_path,
            use_finetuned=args.use_finetuned,
            pred_len=args.pred_len,
            context_len=args.context_len,
            val_dir=args.val_dir,
            dataset=args.dataset,
            output_dir=args.output_dir,
            device=args.device,
        )
        
        logger.info("=" * 60)
        logger.info("TIME-MOE MODEL EVALUATION")
        logger.info("=" * 60)
        
        eval_results = evaluator.evaluate()
        
        evaluator.save_results(eval_results, summary_output_path, subject_output_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: {eval_results['model_name']}")
        logger.info(f"Dataset: {eval_results['dataset_name']}")
        logger.info(f"Subjects: {eval_results['num_subjects']}")
        logger.info(f"Total samples: {eval_results['total_samples']}")
        logger.info(f"\nMAE: {eval_results['MAE_mean']:.6f} ± {eval_results['MAE_std']:.6f}")
        logger.info(f"RMSE: {eval_results['RMSE_mean']:.6f} ± {eval_results['RMSE_std']:.6f}")
        logger.info(f"R²: {eval_results['R2_mean']:.6f} ± {eval_results['R2_std']:.6f}")
        logger.info(f"CEA A+B: {eval_results['CEA_AB_mean']:.2f}% ± {eval_results['CEA_AB_std']:.2f}%")
        logger.info(f"CEA C+D+E: {eval_results['CEA_CDE_mean']:.2f}% ± {eval_results['CEA_CDE_std']:.2f}%")
        logger.info(f"Inference time (median): {eval_results['inference_time_median']:.4f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()