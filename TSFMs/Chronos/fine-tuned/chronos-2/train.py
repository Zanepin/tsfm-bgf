import os
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import json
import shutil
from typing import List, Dict, Tuple
from collections import defaultdict
from chronos import BaseChronosPipeline, Chronos2Pipeline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Chronos2Trainer:
    def __init__(
            self,
            seed: int = 42,
            dataset: str = "op",
            pred_len: int = 12,
            train_dir: str = None,
            val_dir: str = None,
            output_dir: str = None,
            model_id: str = "amazon/chronos-2",
    ):
        self.seed = seed
        self.dataset = dataset
        self.pred_len = pred_len
        self.model_id = model_id

        if train_dir is None:
            train_dir = f"/mnt/d/glucose_data/internal/{dataset}_split2/train"
        if val_dir is None:
            val_dir = f"/mnt/d/glucose_data/internal/{dataset}_split2/test"
        if output_dir is None:
            output_dir = f"./output/{dataset}_pred{pred_len}"

        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)

        logger.info(f"Chronos-2 Trainer initialized:")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Model: {model_id}")
        logger.info(f"  Train directory: {self.train_dir}")
        logger.info(f"  Validation directory: {self.val_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Prediction length: {pred_len}")
        logger.info(f"  Seed: {seed}")

    @staticmethod
    def _extract_subject_id(filename: str) -> str:
        """Extract subject ID from filename"""
        parts = filename.split('_')
        return parts[0] if parts else filename

    def load_csv_files_for_chronos2(self, input_dir: Path) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Load data from CSV files and convert to the format required by Chronos-2
        
        Chronos-2 Input Format:
        {
            "target": np.ndarray,
            "past_covariates": {col_name: np.ndarray, ...},  # Optional
            "future_covariates": {col_name: None or np.ndarray, ...},  # Optional
        }
        
        Returns: (train_inputs, subject_counts)
        """
        logger.info(f"Loading CSV files from {input_dir}")

        if not input_dir.exists():
            raise ValueError(f"Directory {input_dir} does not exist")

        csv_files = sorted(list(input_dir.glob("*.csv")), key=lambda x: x.stem)
        logger.info(f"Found {len(csv_files)} CSV files")

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {input_dir}")

        train_inputs = []
        subject_ids = []
        successful_loads = 0
        failed_loads = 0

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'GlucoseValue' not in df.columns:
                    logger.warning(f"File {csv_file.name}: 'GlucoseValue' column not found, skipping")
                    failed_loads += 1
                    continue

                glucose_values = df['GlucoseValue'].values
                glucose_values = glucose_values[~np.isnan(glucose_values)]
                
                if len(glucose_values) == 0:
                    logger.warning(f"File {csv_file.name}: No valid data, skipping")
                    failed_loads += 1
                    continue

                glucose_array = glucose_values.astype(np.float32)
                subject_id = self._extract_subject_id(csv_file.stem)

                # Chronos-2 Format
                train_inputs.append({
                    "target": glucose_array,
                })
                subject_ids.append(subject_id)
                successful_loads += 1

                if successful_loads % 1000 == 0:
                    logger.info(f"Loaded {successful_loads} files successfully")

            except Exception as e:
                logger.warning(f"Error loading {csv_file.name}: {e}")
                failed_loads += 1

        logger.info(f"Data loading complete from {input_dir}:")
        logger.info(f"  Successfully loaded: {successful_loads} files")
        logger.info(f"  Failed to load: {failed_loads} files")
        logger.info(f"  Total time series: {len(train_inputs)}")

        if len(train_inputs) == 0:
            raise ValueError("No valid time series data loaded")

        subject_counts = defaultdict(int)
        for subject_id in subject_ids:
            subject_counts[subject_id] += 1

        return train_inputs, dict(sorted(subject_counts.items()))

    def save_metadata(self, train_inputs: List[Dict], val_inputs: List[Dict], 
                      subject_counts: Dict[str, int]):
        """Save training metadata"""
        metadata = {
            'dataset': self.dataset,
            'model_id': self.model_id,
            'model_type': 'chronos-2',
            'prediction_length': self.pred_len,
            'num_train_series': len(train_inputs),
            'num_val_series': len(val_inputs),
            'num_subjects': len(subject_counts),
            'subject_counts': subject_counts,
            'seed': self.seed
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")

    def train(self, 
              gpu_id: int = 0,
              num_steps: int = 1000,
              batch_size: int = 32,
              learning_rate: float = 1e-5,
              finetune_mode: str = "full",
              logging_steps: int = 100):
        """
        Complete training pipeline (using Chronos-2 fit API)
        
        Args:
            gpu_id: GPU ID
            num_steps: Training steps
            batch_size: Batch size
            learning_rate: Learning rate (full: 1e-5, lora: 1e-4)
            finetune_mode: "full" or "lora"
            logging_steps: Logging interval
        """
        
        logger.info("\nLoading training data...")
        train_inputs, train_subjects = self.load_csv_files_for_chronos2(self.train_dir)
        
        logger.info("\nLoading validation data...")
        val_inputs, val_subjects = self.load_csv_files_for_chronos2(self.val_dir)
        
        self.save_metadata(train_inputs, val_inputs, val_subjects)

        logger.info(f"\nData summary:")
        logger.info(f"  Training series: {len(train_inputs)}")
        logger.info(f"  Validation series: {len(val_inputs)}")
        logger.info(f"  Training subjects: {len(train_subjects)}")
        logger.info(f"  Validation subjects: {len(val_subjects)}")

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"\nLoading Chronos-2 model: {self.model_id}")
        
        try:
            pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=device,
            )
            logger.info(f"✅ Successfully loaded Chronos-2 model")
        except Exception as e:
            logger.error(f"Failed to load Chronos-2 model: {e}")
            raise

        logger.info(f"\nStarting fine-tuning...")
        logger.info(f"  Mode: {finetune_mode}")
        logger.info(f"  Steps: {num_steps}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Device: {device}")

        try:
            # Use the Chronos-2 fit method
            if finetune_mode == "lora":
                finetuned_pipeline = pipeline.fit(
                    inputs=train_inputs,
                    prediction_length=self.pred_len,
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    logging_steps=logging_steps,
                    finetune_mode="lora",
                    validation_inputs=val_inputs,
                )
            else:
                finetuned_pipeline = pipeline.fit(
                    inputs=train_inputs,
                    prediction_length=self.pred_len,
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    logging_steps=logging_steps,
                    finetune_mode="full",
                    validation_inputs=val_inputs,
                )

            # fit method already saves the model automatically
            # We just need to copy it to the target location
            checkpoint_path = self.output_dir / "checkpoint-final"
            
            # Find the model path saved by the fit method
            # fit method defaults to saving in chronos-2-finetuned/{timestamp}/finetuned-ckpt
            finetuned_dir = Path("chronos-2-finetuned")
            if finetuned_dir.exists():
                # Find the latest checkpoint
                subdirs = [d for d in finetuned_dir.iterdir() if d.is_dir()]
                if subdirs:
                    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
                    source_ckpt = latest_dir / "finetuned-ckpt"
                    
                    if source_ckpt.exists():
                        logger.info(f"\nCopying model from {source_ckpt} to {checkpoint_path}")
                        
                        # If target path exists, delete it first
                        if checkpoint_path.exists():
                            shutil.rmtree(checkpoint_path)
                        
                        # Copy the entire directory
                        shutil.copytree(source_ckpt, checkpoint_path)
                        logger.info(f"Model successfully copied to {checkpoint_path}")
                    else:
                        logger.warning(f"Source checkpoint not found: {source_ckpt}")
                else:
                    logger.warning(f"No subdirectories found in {finetuned_dir}")
            else:
                logger.warning(f"Finetuned directory not found: {finetuned_dir}")
                # If the auto-saved path is not found, attempt to save manually
                logger.info(f"\nAttempting to save model manually to {checkpoint_path}")
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                finetuned_pipeline.model.save_pretrained(checkpoint_path)

            logger.info(f"\nTraining completed successfully!")
            logger.info(f"  Model saved to: {checkpoint_path}")
            
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"\nTraining failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Chronos-2 Model Fine-tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Full fine-tuning (recommended for small datasets)
  python train.py --dataset op --pred_len 12 --lr 1e-5

  # LoRA fine-tuning (faster, consumes less VRAM)
  python train.py --dataset op --pred_len 12 --finetune_mode lora --lr 1e-4

  # Custom parameters
  python train.py --dataset lg --pred_len 6 --num_steps 2000 --batch_size 64

Note:
  - Chronos-2 has only one model: amazon/chronos-2
  - Recommended learning rate for Full fine-tuning: 1e-5
  - Recommended learning rate for LoRA fine-tuning: 1e-4
        """
    )

    parser.add_argument("--dataset", type=str, choices=["op", "re", "lg"], default="op",
                        help="Select dataset (default: op)")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Training data CSV directory")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Validation data CSV directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2",
                        help="Model ID (default: amazon/chronos-2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--pred_len", type=int, default=12,
                        help="Prediction length (default: 12)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID (default: 0)")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Training steps (default: 500)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (full: 1e-5, lora: 1e-4)")
    parser.add_argument("--finetune_mode", type=str, choices=["full", "lora"], default="full",
                        help="Fine-tuning mode: full or lora (default: full)")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging interval (default: 100)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Automatically adjust learning rate if using LoRA without manual override
    if args.finetune_mode == "lora" and args.lr == 1e-5:
        logger.info("LoRA mode detected, adjusting learning rate to 1e-4")
        args.lr = 1e-4

    trainer = Chronos2Trainer(
        seed=args.seed,
        dataset=args.dataset,
        pred_len=args.pred_len,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
    )

    checkpoint_path = trainer.train(
        gpu_id=args.gpu_id,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        finetune_mode=args.finetune_mode,
        logging_steps=args.logging_steps,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    logger.info(f"{'='*60}")