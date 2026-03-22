import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from sklearn.metrics import r2_score
from model import ChronosT5Model
import logging
import traceback
from gluonts.dataset.arrow import ArrowFile
import pyarrow as pa
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.clarke_error_grid import ClarkeErrorGrid, calculate_clarke_metrics


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ==========================================
# Clarke Error Grid Analysis Module
# ==========================================
def read_arrow_metadata(arrow_path: str) -> Dict[str, any]:
    """
    Read metadata from an Arrow file.
    Returns: {
        'dataset_name': str,
        'total_series': int,
        'total_subjects': int,
        'subject_ids': List[str],
        'subject_counts': Dict[str, int]
    }
    """
    table = pa.ipc.open_file(pa.memory_map(str(arrow_path), 'r')).read_all()
    metadata = table.schema.metadata

    if not metadata or b'subject_ids' not in metadata:
        logger.warning(f"No subject metadata found in {arrow_path}")
        logger.warning("This arrow file may have been created before metadata support was added.")
        logger.warning("Please regenerate the arrow file using the updated train.py script.")
        return None

    result = {
        'dataset_name': metadata.get(b'dataset_name', b'').decode('utf-8'),
        'total_series': int(metadata.get(b'total_series', b'0').decode('utf-8')),
        'total_subjects': int(metadata.get(b'total_subjects', b'0').decode('utf-8')),
        'subject_ids': json.loads(metadata.get(b'subject_ids', b'[]').decode('utf-8')),
        'subject_counts': json.loads(metadata.get(b'subject_counts', b'{}').decode('utf-8')),
    }

    return result


def create_subject_mapping(subject_ids: List[str]) -> Dict[int, str]:
    """Create a mapping from sequence index to subject ID"""
    return {idx: subject_id for idx, subject_id in enumerate(subject_ids)}


class TimeSeriesEvaluator:
    def __init__(self,
                 model_path: str = None,
                 use_finetuned: bool = False,
                 pred_len: int = 12,
                 arrow_path: str = "./processed_data/op_split2/val_data.arrow",
                 dataset: str = "op"):
        self.model_path = model_path
        self.use_finetuned = use_finetuned
        self.input_len = 48
        self.pred_len = pred_len
        self.arrow_path = arrow_path
        self.dataset = dataset
        self.model = None
        self.model_name = None

        # Load subject info from Arrow file
        self.metadata = self.load_subject_info_from_arrow()
        if self.metadata is None:
            raise ValueError(
                f"Failed to load metadata from {arrow_path}. "
                "Please regenerate the arrow file using the updated train.py script with metadata support."
            )

        self.subject_ids = self.metadata['subject_ids']
        self.subject_counts = self.metadata['subject_counts']
        self.subject_mapping = create_subject_mapping(self.subject_ids)

        self.verify_data_consistency()
        self.load_model()

    def load_subject_info_from_arrow(self) -> Dict[str, any]:
        """Load subject information from Arrow file metadata"""
        logger.info(f"Loading subject information from Arrow metadata: {self.arrow_path}")

        if not os.path.exists(self.arrow_path):
            raise FileNotFoundError(f"Arrow file not found: {self.arrow_path}")

        metadata = read_arrow_metadata(self.arrow_path)

        if metadata:
            logger.info(f"✓ Metadata loaded successfully:")
            logger.info(f"  Dataset: {metadata['dataset_name']}")
            logger.info(f"  Total series: {metadata['total_series']}")
            logger.info(f"  Total subjects: {metadata['total_subjects']}")
            logger.info(f"  First 3 subjects: {list(metadata['subject_counts'].items())[:3]}")

        return metadata

    def verify_data_consistency(self):
        """Verify that the count in metadata matches the sequence count in the arrow file"""
        logger.info("Verifying data consistency...")

        # Get expected total from metadata
        expected_total = self.metadata['total_series']

        # Calculate actual count in arrow file
        dataset = ArrowFile(self.arrow_path)
        actual_total = sum(1 for _ in dataset)

        logger.info(f"Metadata expected total: {expected_total}")
        logger.info(f"Arrow file actual total: {actual_total}")
        logger.info(f"Number of subjects: {self.metadata['total_subjects']}")

        if expected_total != actual_total:
            raise ValueError(
                f"Data inconsistency detected! "
                f"Metadata shows {expected_total} series, "
                f"but arrow file contains {actual_total} series."
            )

        # Verify subject_ids list length
        if len(self.subject_ids) != actual_total:
            raise ValueError(
                f"Subject IDs list length mismatch! "
                f"Expected {actual_total}, got {len(self.subject_ids)}"
            )

        logger.info("✓ Data consistency verified!")

    def load_model(self):
        """Load the model"""
        if self.use_finetuned and self.model_path and os.path.exists(self.model_path):
            self.model = ChronosT5Model()
            self.model.load_finetuned_model(self.model_path)
            self.model_name = f"Fine-tuned-{os.path.basename(self.model_path)}"
            logger.info(f"Loaded fine-tuned model from {self.model_path}")
        else:
            self.model = ChronosT5Model()
            self.model_name = "Pre-trained-ChronosT5"
            if self.use_finetuned and self.model_path:
                logger.warning(f"Fine-tuned model path {self.model_path} not found, using pre-trained model instead")
            else:
                logger.info("Loaded pre-trained model")

    @staticmethod
    def mae(pred: np.ndarray, true: np.ndarray) -> float:
        return np.mean(np.abs(pred - true))

    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return np.sqrt(np.mean((pred - true) ** 2))

    @staticmethod
    def r2(pred: np.ndarray, true: np.ndarray) -> float:
        # Ensure input is 1D array
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        if len(pred_flat) != len(true_flat):
            logger.warning(
                f"R2 score calculation warning: prediction and target shapes are different. Pred: {pred.shape}, True: {true.shape}")
            return np.nan
        return r2_score(true_flat, pred_flat)

    def prepare_eval_data_by_subjects(self) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
        """Extract evaluation data from .arrow validation data grouped by subjects (using Arrow metadata)"""
        if not os.path.exists(self.arrow_path):
            raise FileNotFoundError(f"Validation data file not found: {self.arrow_path}")

        logger.info(f"Loading evaluation data from {self.arrow_path}")

        dataset = ArrowFile(self.arrow_path)

        subjects_data = {}
        input_len = self.input_len
        pred_len = self.pred_len
        total_len = input_len + pred_len

        # Manually adjust sliding window stride
        stride = 12

        logger.info(f"Extracting sliding windows with stride={stride} grouped by subjects...")

        series_count = 0

        for entry in dataset:
            series = np.array(entry["target"], dtype=np.float32)

            # Determine subject ID using mapping from Arrow metadata
            if series_count not in self.subject_mapping:
                logger.warning(f"Series {series_count} not found in subject mapping, skipping")
                series_count += 1
                continue

            subject_id = self.subject_mapping[series_count]

            if subject_id not in subjects_data:
                subjects_data[subject_id] = ([], [])  # (contexts, targets)

            series_count += 1

            if len(series) < total_len:
                logger.debug(
                    f"Series {series_count} (Subject {subject_id}): too short ({len(series)} < {total_len}), skipping")
                continue

            max_start_idx = len(series) - total_len

            for i in range(0, max_start_idx + 1, stride):
                context = series[i:i + input_len]
                target = series[i + input_len:i + total_len]
                subjects_data[subject_id][0].append(context)
                subjects_data[subject_id][1].append(target)

            if series_count % 500 == 0:
                logger.info(f"Processed {series_count} series, {len(subjects_data)} subjects so far...")

        logger.info(f"Data extraction complete:")
        logger.info(f"  Total series processed: {series_count}")
        logger.info(f"  Total subjects: {len(subjects_data)}")

        # Count windows and original series per subject
        total_windows = 0
        for subject_id in sorted(subjects_data.keys()):
            contexts, targets = subjects_data[subject_id]
            windows_count = len(contexts)
            original_series_count = self.subject_counts[subject_id]
            total_windows += windows_count
            logger.debug(f"Subject {subject_id}: {original_series_count} series -> {windows_count} windows")

        logger.info(f"  Total windows extracted: {total_windows}")
        logger.info(f"  Average windows per subject: {total_windows / len(subjects_data):.1f}")

        return subjects_data

    def evaluate_subject(self, contexts: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate performance for a single subject"""
        if not contexts:
            return {
                'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 
                'CEA_AB': np.nan, 'CEA_CDE': np.nan, 
                'inference_time_median': np.nan, 'num_samples': 0
            }

        predictions = []
        successful_predictions = 0
        inference_times = []  # Record inference time for each batch

        # Batch prediction
        batch_size = 128

        for batch_idx in range(0, len(contexts), batch_size):
            batch_contexts = contexts[batch_idx:batch_idx + batch_size]
            batch_contexts = [torch.tensor(c, dtype=torch.float32) if not isinstance(c, torch.Tensor) else c for c in
                              batch_contexts]

            try:
                # Record inference start time
                import time
                start_time = time.perf_counter()
                
                batch_preds = self.model.forecast_mean(
                    context=batch_contexts,
                    prediction_length=self.pred_len,
                    num_samples=10
                )
                
                # Record inference end time
                end_time = time.perf_counter()
                batch_time = end_time - start_time
                inference_times.append(batch_time)

                # Standardize prediction output to (B, pred_len)
                batch_preds = np.asarray(batch_preds)
                if batch_preds.ndim == 1:
                    batch_preds = batch_preds.reshape(1, -1)
                elif batch_preds.ndim > 2:
                    batch_preds = np.squeeze(batch_preds)
                    if batch_preds.ndim == 1:
                        batch_preds = batch_preds.reshape(1, -1)
                    elif batch_preds.ndim != 2:
                        raise ValueError(f"Unexpected prediction shape after squeeze: {batch_preds.shape}")

                # Ensure prediction length matches
                if batch_preds.shape[1] != self.pred_len:
                    if batch_preds.shape[1] > self.pred_len:
                        batch_preds = batch_preds[:, :self.pred_len]
                    else:
                        pad_w = self.pred_len - batch_preds.shape[1]
                        batch_preds = np.pad(batch_preds, ((0, 0), (0, pad_w)), mode='edge')

                predictions.extend([row for row in batch_preds])
                successful_predictions += batch_preds.shape[0]

            except Exception as e:
                logger.error(f"Error predicting batch for subject: {e}")
                continue

        if not predictions:
            return {
                'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 
                'CEA_AB': np.nan, 'CEA_CDE': np.nan, 
                'inference_time_median': np.nan, 'num_samples': 0
            }

        predictions = np.array(predictions)
        targets = np.array(targets[:len(predictions)])

        # Ensure shapes match
        if predictions.shape != targets.shape:
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]

        # Calculate evaluation metrics
        mae = self.mae(predictions, targets)
        rmse = self.rmse(predictions, targets)
        r2 = self.r2(predictions, targets)
        
        # Calculate Clarke Error Grid metrics
        cea_result = calculate_clarke_metrics(targets.flatten(), predictions.flatten())
        cea_ab = cea_result.get('AB_percentage', np.nan)
        cea_cde = cea_result.get('CDE_percentage', np.nan)
        
        # Calculate median inference time
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
        """Evaluate performance for all subjects"""
        subjects_data = self.prepare_eval_data_by_subjects()

        if not subjects_data:
            raise ValueError("No evaluation data available.")

        logger.info(f"Starting evaluation on {len(subjects_data)} subjects")

        subject_results = {}
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        cea_ab_scores = []
        cea_cde_scores = []
        inference_times = []  # Collect median inference time for each subject
        total_samples = 0

        for subject_id in sorted(subjects_data.keys()):
            contexts, targets = subjects_data[subject_id]
            original_series_count = self.subject_counts[subject_id]
            logger.info(
                f"Evaluating subject {subject_id} ({original_series_count} series -> {len(contexts)} windows)...")

            result = self.evaluate_subject(contexts, targets)
            subject_results[subject_id] = result

            if not np.isnan(result['MAE']):
                mae_scores.append(result['MAE'])
                rmse_scores.append(result['RMSE'])
                r2_scores.append(result['R2'])
                total_samples += result['num_samples']
                
                # Add CEA metrics
                if not np.isnan(result['CEA_AB']):
                    cea_ab_scores.append(result['CEA_AB'])
                    cea_cde_scores.append(result['CEA_CDE'])
                
                # Add inference time
                if not np.isnan(result['inference_time_median']):
                    inference_times.append(result['inference_time_median'])

            logger.info(
                f"Subject {subject_id}: MAE={result['MAE']:.6f}, RMSE={result['RMSE']:.6f}, R2={result['R2']:.6f}, "
                f"CEA_AB={result['CEA_AB']:.2f}%, CEA_CDE={result['CEA_CDE']:.2f}%, "
                f"Inference Time={result['inference_time_median']:.4f}s")

        # Calculate mean and standard deviation across all subjects
        mae_mean = np.mean(mae_scores) if mae_scores else np.nan
        mae_std = np.std(mae_scores) if mae_scores else np.nan
        rmse_mean = np.mean(rmse_scores) if rmse_scores else np.nan
        rmse_std = np.std(rmse_scores) if rmse_scores else np.nan
        r2_mean = np.mean(r2_scores) if r2_scores else np.nan
        r2_std = np.std(r2_scores) if r2_scores else np.nan
        cea_ab_mean = np.mean(cea_ab_scores) if cea_ab_scores else np.nan
        cea_ab_std = np.std(cea_ab_scores) if cea_ab_scores else np.nan
        cea_cde_mean = np.mean(cea_cde_scores) if cea_cde_scores else np.nan
        cea_cde_std = np.std(cea_cde_scores) if cea_cde_scores else np.nan
        
        # Calculate global median inference time
        inference_time_median = np.median(inference_times) if inference_times else np.nan

        # Get directory info for the arrow file
        arrow_dir = os.path.basename(os.path.dirname(self.arrow_path))

        return {
            'MAE_mean': mae_mean,
            'MAE_std': mae_std,
            'RMSE_mean': rmse_mean,
            'RMSE_std': rmse_std,
            'R2_mean': r2_mean,
            'R2_std': r2_std,
            'CEA_AB_mean': cea_ab_mean,
            'CEA_AB_std': cea_ab_std,
            'CEA_CDE_mean': cea_cde_mean,
            'CEA_CDE_std': cea_cde_std,
            'inference_time_median': inference_time_median,
            'num_subjects': len(mae_scores),
            'total_samples': total_samples,
            'total_series': self.metadata['total_series'],
            'context_length': self.input_len,
            'prediction_length': self.pred_len,
            'model_name': self.model_name,
            'data_source': arrow_dir,
            'arrow_file': os.path.basename(self.arrow_path),
            'model_type': 'Fine-tuned' if self.use_finetuned else 'Pre-trained',
            'dataset_name': self.metadata['dataset_name'],
            'subject_results': subject_results
        }

    @staticmethod
    def save_results(eval_results: dict, summary_path: str, subject_path: str):
        """
        Save evaluation results to two CSV files.

        Args:
            eval_results: Dictionary of evaluation results
            summary_path: Path for the summary results file
            subject_path: Path for the subject-level metrics file
        """
        # 1. Save summary results (excluding subject_results)
        main_results = {k: v for k, v in eval_results.items() if k != 'subject_results'}
        df_summary = pd.DataFrame([main_results])

        # If file exists, append instead of overwrite
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            df_summary = pd.concat([existing_df, df_summary], ignore_index=True)

        # Rearrange column order
        column_order = [
            'model_type', 'model_name', 'data_source', 'dataset_name', 'arrow_file',
            'context_length', 'prediction_length', 'num_subjects', 'total_series', 'total_samples',
            'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std',
            'CEA_AB_mean', 'CEA_AB_std', 'CEA_CDE_mean', 'CEA_CDE_std',
            'inference_time_median'
        ]

        # Keep all columns but sort by specified order
        all_columns = list(df_summary.columns)
        ordered_columns = [col for col in column_order if col in all_columns]
        remaining_columns = [col for col in all_columns if col not in column_order]
        df_summary = df_summary[ordered_columns + remaining_columns]

        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Summary results saved to {summary_path}")

        # 2. Save subject-level metrics
        subject_results = eval_results.get('subject_results', {})

        if not subject_results:
            logger.warning("No subject results to save")
            return

        # Prepare data rows
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

        # Create DataFrame and save
        df_subjects = pd.DataFrame(rows)
        df_subjects.to_csv(subject_path, index=False)
        logger.info(f"Subject-level metrics saved to {subject_path}")
        logger.info(f"  Total subjects: {len(rows)}")
        logger.info(f"  Columns: {', '.join(df_subjects.columns)}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Chronos T5 Time Series Model Evaluation (Arrow Metadata Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Evaluate op dataset using pre-trained model
  python eval.py --dataset op --pred_len 12

  # Evaluate using fine-tuned model
  python eval.py --dataset op --use_finetuned --model_path ./output/run-0/checkpoint-final

  # Use custom arrow file
  python eval.py --val_arrow ./custom_data/val_data.arrow --pred_len 12

Note: 
  - Arrow file must contain metadata (generated by the updated train.py)
  - No external subject.json file required
  - Subject information is read directly from Arrow metadata
        """
    )

    parser.add_argument('--dataset', type=str, choices=["op", "re", "lg"], default='op',
                        help="Select dataset: 'op', 're' or 'lg' (Default: op)")

    parser.add_argument('--val_arrow', type=str, default=None,
                        help='Custom evaluation set path (Default: auto-selected based on dataset)')

    parser.add_argument('--use_finetuned', action='store_true', default=False,
                        help='Use fine-tuned model instead of pre-trained model')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to fine-tuned model checkpoint')

    parser.add_argument('--pred_len', type=int, default=12,
                        help='Prediction length (Default: 12)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set data directories
    if args.val_arrow is None:
        # If evaluation set not specified, use dataset default
        if args.dataset == "op":
            eval_arrow_paths = "./processed_data/op_split2/val_data.arrow"
        elif args.dataset == "re":
            eval_arrow_paths = "./processed_data/re_split2/val_data.arrow"
        elif args.dataset == "lg":
            eval_arrow_paths = "./processed_data/lg_split2/val_data.arrow"
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}. Must be 'op', 're' or 'lg'")
    else:
        # Use specified evaluation set path
        eval_arrow_paths = args.val_arrow
        logger.info(f"Using custom validation data path: {eval_arrow_paths}")
    
    arrow_str = eval_arrow_paths.split('/')[-2]

    # Generate output filenames based on dataset and pred_len
    summary_output_path = f"./eval_results_{arrow_str}_{args.pred_len}.csv"
    subject_output_path = f"./subject_metrics_{arrow_str}_{args.pred_len}.csv"

    # If using fine-tuned model but no path provided, attempt auto-detection
    model_path = args.model_path
    if args.use_finetuned and not model_path:
        # Select checkpoint automatically based on dataset and pred_len
        if args.dataset == "op" and args.pred_len == 12:
            model_path = "./output/run-0/checkpoint-final"
        elif args.dataset == "re" and args.pred_len == 12:
            model_path = "./output/run-2/checkpoint-final"
        elif args.dataset == "op" and args.pred_len == 6:
            model_path = "./output/run-1/checkpoint-final"
        elif args.dataset == "re" and args.pred_len == 6:
            model_path = "./output/run-3/checkpoint-final"
        elif args.dataset == "lg" and args.pred_len == 12:
            model_path = "./output/run-6/checkpoint-final"
        elif args.dataset == "lg" and args.pred_len == 6:
            model_path = "./output/run-7/checkpoint-final"
        else:
            logger.warning(f"No default checkpoint path for dataset={args.dataset}, pred_len={args.pred_len}")
            model_path = None

        if model_path:
            logger.info(f"Auto-detected checkpoint: {model_path}")

    try:
        evaluator = TimeSeriesEvaluator(
            model_path=model_path,
            use_finetuned=args.use_finetuned,
            pred_len=args.pred_len,
            arrow_path=eval_arrow_paths,
            dataset=args.dataset
        )

        logger.info("=" * 60)
        logger.info("CHRONOS MODEL EVALUATION - ARROW METADATA-BASED")
        logger.info("=" * 60)
        logger.info(f"Model Type: {'Fine-tuned' if args.use_finetuned else 'Pre-trained'}")
        logger.info(f"Model Name: {evaluator.model_name}")
        logger.info(f"Context Length: {evaluator.input_len}")
        logger.info(f"Prediction Length: {args.pred_len}")
        logger.info(f"Data Source: {args.dataset}")
        logger.info(f"Validation Data: {eval_arrow_paths}")
        logger.info(f"Dataset Name (from metadata): {evaluator.metadata['dataset_name']}")
        logger.info(f"Total Subjects: {len(evaluator.subject_counts)}")
        logger.info(f"Total Series: {evaluator.metadata['total_series']}")
        logger.info("=" * 60)

        results = evaluator.evaluate()

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS - METADATA-BASED SUBJECT METRICS")
        print("=" * 60)
        print(f"  Model Type:        {results['model_type']}")
        print(f"  Model Name:        {results['model_name']}")
        print(f"  Data Source:       {results['data_source']}")
        print(f"  Dataset Name:      {results['dataset_name']}")
        print(f"  Context Length:    {results['context_length']}")
        print(f"  Pred Length:       {results['prediction_length']}")
        print(f"  Subjects:          {results['num_subjects']}")
        print(f"  Total Series:      {results['total_series']:,}")
        print(f"  Total Samples:     {results['total_samples']:,}")
        print(f"  MAE (Mean ± Std):  {results['MAE_mean']:.6f} ± {results['MAE_std']:.6f}")
        print(f"  RMSE (Mean ± Std): {results['RMSE_mean']:.6f} ± {results['RMSE_std']:.6f}")
        print(f"  R² (Mean ± Std):   {results['R2_mean']:.6f} ± {results['R2_std']:.6f}")
        print(f"  CEA Zone A+B (%):  {results['CEA_AB_mean']:.2f} ± {results['CEA_AB_std']:.2f}")
        print(f"  CEA Zone C+D+E (%): {results['CEA_CDE_mean']:.2f} ± {results['CEA_CDE_std']:.2f}")
        print(f"  Inference Time (Median): {results['inference_time_median']:.4f}s")
        print("=" * 60)

        # Save all results (Summary + Subject-level)
        evaluator.save_results(results, summary_output_path, subject_output_path)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        traceback.print_exc()