import os
import yaml
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import logging
import argparse
import re
import json
from typing import List, Union, Dict, Tuple
from gluonts.dataset.arrow import ArrowWriter
import pyarrow as pa
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ChronosTrainer:
    def __init__(
            self,
            config_path: str = "./config.yaml",
            seed: int = 42,
            dataset: str = "op",
            pred_len: int = 12,
            train_dir: str = None,
            val_dir: str = None,
            output_dir: str = None,
    ):
        self.config_path = config_path
        self.seed = seed
        self.dataset = dataset
        self.pred_len = pred_len

        # Set data directories
        if train_dir is None:
            train_dir = f"/mnt/d/glucose_data/internal/{dataset}_split2/train"
        if val_dir is None:
            val_dir = f"/mnt/d/glucose_data/internal/{dataset}_split2/test"
        if output_dir is None:
            output_dir = f"./processed_data/{dataset}_split2"

        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        set_seed(seed)

        logger.info(f"Trainer initialized:")
        logger.info(f"  Dataset: {dataset}")
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

    def load_csv_files_with_metadata(self, input_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load data from CSV files and record the subject_id for each sequence.
        Returns: (time_series_list, subject_ids)
        """
        logger.info(f"Loading CSV files from {input_dir}")

        if not input_dir.exists():
            raise ValueError(f"Directory {input_dir} does not exist")

        csv_files = list(input_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {input_dir}")

        # Sort by filename to ensure consistent order
        csv_files = sorted(csv_files, key=lambda x: x.stem)

        time_series_list = []
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
                if len(glucose_values) == 0:
                    logger.warning(f"File {csv_file.name}: No data in GlucoseValue column, skipping")
                    failed_loads += 1
                    continue

                # Remove NaN values
                glucose_values = glucose_values[~np.isnan(glucose_values)]
                if len(glucose_values) == 0:
                    logger.warning(f"File {csv_file.name}: All values are NaN, skipping")
                    failed_loads += 1
                    continue

                glucose_array = glucose_values.astype(np.float32)
                subject_id = self._extract_subject_id(csv_file.stem)

                time_series_list.append(glucose_array)
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
        logger.info(f"  Total time series: {len(time_series_list)}")

        if len(time_series_list) == 0:
            raise ValueError("No valid time series data loaded")

        return time_series_list, subject_ids

    def get_subject_statistics(self, subject_ids: List[str]) -> Dict[str, int]:
        """Count the number of sequences for each subject"""
        subject_counts = defaultdict(int)
        for subject_id in subject_ids:
            subject_counts[subject_id] += 1
        return dict(sorted(subject_counts.items()))

    def convert_to_arrow_with_metadata(
            self,
            path: Union[str, Path],
            time_series: List[np.ndarray],
            subject_ids: List[str],
            compression: str = "lz4",
    ):
        """
        Convert time series to Arrow format and save subject mapping in schema metadata
        """
        if len(time_series) != len(subject_ids):
            raise ValueError(f"Mismatch: {len(time_series)} series but {len(subject_ids)} subject_ids")

        # Set an arbitrary start time
        start = np.datetime64("2000-01-01 00:00", "s")

        # Prepare dataset
        dataset = [
            {"start": start, "target": ts.astype(np.float32)} for ts in time_series
        ]

        # Prepare metadata: record subject_id and statistics for each sequence
        subject_counts = self.get_subject_statistics(subject_ids)

        metadata = {
            b"dataset_name": self.dataset.encode('utf-8'),
            b"total_series": str(len(time_series)).encode('utf-8'),
            b"total_subjects": str(len(subject_counts)).encode('utf-8'),
            # Save complete list of subject_ids (in sequence order)
            b"subject_ids": json.dumps(subject_ids).encode('utf-8'),
            # Save subject statistics
            b"subject_counts": json.dumps(subject_counts).encode('utf-8'),
        }

        # Write data using ArrowWriter
        writer = ArrowWriter(compression=compression)
        writer.write_to_file(dataset, path=path)

        # Read the newly written file, add metadata, and re-save
        table = pa.ipc.open_file(pa.memory_map(str(path), 'r')).read_all()

        # Get existing schema and add metadata
        schema = table.schema
        # Handle case where schema.metadata might be None
        existing_metadata = schema.metadata if schema.metadata is not None else {}
        # Merge metadata
        new_metadata = {**existing_metadata, **metadata}
        schema = schema.with_metadata(new_metadata)

        # Create new table and save
        table_with_metadata = pa.table(table.to_pandas(), schema=schema)
        with pa.OSFile(str(path), 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                writer.write_table(table_with_metadata)

        logger.info(f"Saved {len(time_series)} time series to {path}")
        logger.info(f"Metadata embedded: {len(subject_counts)} subjects")

        # Statistics
        lengths = [len(ts) for ts in time_series]
        values = np.concatenate(time_series)
        logger.info(f"Series statistics:")
        logger.info(f"  Lengths - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}")
        logger.info(f"  Values - Min: {values.min():.3f}, Max: {values.max():.3f}, Mean: {np.mean(values):.3f}")

        # Display subject distribution
        logger.info(f"Subject distribution:")
        for i, (subject_id, count) in enumerate(list(subject_counts.items())[:5]):
            logger.info(f"  {subject_id}: {count} series")
        if len(subject_counts) > 5:
            logger.info(f"  ... and {len(subject_counts) - 5} more subjects")

    @staticmethod
    def read_arrow_metadata(arrow_path: Union[str, Path]) -> Dict[str, any]:
        """
        Read metadata from Arrow file
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
            return None

        result = {
            'dataset_name': metadata.get(b'dataset_name', b'').decode('utf-8'),
            'total_series': int(metadata.get(b'total_series', b'0').decode('utf-8')),
            'total_subjects': int(metadata.get(b'total_subjects', b'0').decode('utf-8')),
            'subject_ids': json.loads(metadata.get(b'subject_ids', b'[]').decode('utf-8')),
            'subject_counts': json.loads(metadata.get(b'subject_counts', b'{}').decode('utf-8')),
        }

        return result

    def prepare_data(self, force_reload: bool = False):
        """Prepare training data (automatic check and conversion, metadata embedding)"""
        train_arrow = self.output_dir / "train_data.arrow"
        val_arrow = self.output_dir / "val_data.arrow"

        # Check if arrow files already exist
        if train_arrow.exists() and val_arrow.exists() and not force_reload:
            logger.info("\n" + "=" * 60)
            logger.info("Arrow files already exist. Verifying metadata...")
            logger.info("=" * 60)

            # Verify metadata
            val_metadata = self.read_arrow_metadata(val_arrow)
            if val_metadata:
                logger.info(f"✓ Validation data metadata found:")
                logger.info(f"  Dataset: {val_metadata['dataset_name']}")
                logger.info(f"  Total series: {val_metadata['total_series']}")
                logger.info(f"  Total subjects: {val_metadata['total_subjects']}")
                logger.info(f"  Subject counts: {list(val_metadata['subject_counts'].items())[:3]}...")
            else:
                logger.warning("⚠ No metadata found. Consider using --force_reload to regenerate.")

            logger.info("Use --force_reload to regenerate arrow files")
            return train_arrow, val_arrow

        logger.info("\n" + "=" * 60)
        logger.info("Starting CSV to Arrow data conversion with metadata embedding...")
        logger.info("=" * 60)

        # Load CSV files (with subject information)
        logger.info("\n[1/4] Loading training CSV files...")
        train_list, train_subjects = self.load_csv_files_with_metadata(self.train_dir)

        logger.info("\n[2/4] Loading validation CSV files...")
        val_list, val_subjects = self.load_csv_files_with_metadata(self.val_dir)

        # Convert to Arrow format (embedding metadata)
        logger.info("\n[3/4] Converting training data to Arrow format with metadata...")
        self.convert_to_arrow_with_metadata(train_arrow, train_list, train_subjects)

        logger.info("\n[4/4] Converting validation data to Arrow format with metadata...")
        self.convert_to_arrow_with_metadata(val_arrow, val_list, val_subjects)

        logger.info("\n" + "=" * 60)
        logger.info("DATA CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Training series: {len(train_list)}")
        logger.info(f"Validation series: {len(val_list)}")
        logger.info(f"Total series: {len(train_list) + len(val_list)}")

        val_subject_counts = self.get_subject_statistics(val_subjects)
        logger.info(f"Validation subjects: {len(val_subject_counts)}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info(f"✓ Subject metadata embedded in Arrow files")
        logger.info("=" * 60 + "\n")

        return train_arrow, val_arrow

    def create_config(self, train_arrow: Path):
        """Create training configuration file"""
        config = {
            'training_data_paths': [str(train_arrow.absolute())],
            'probability': [1.0],
            'context_length': 48,
            'prediction_length': self.pred_len,
            'min_past': 6,
            'max_steps': 15_000,
            'save_steps': 5_000,
            'log_steps': 1000,
            'per_device_train_batch_size': 32,
            'learning_rate': 0.0002,
            'optim': 'adamw_torch_fused',
            'num_samples': 10,
            'shuffle_buffer_length': 25000,
            'gradient_accumulation_steps': 1,
            'model_id': 'amazon/chronos-bolt-base',
            'model_type': 'seq2seq',
            'random_init': False,
            'tie_embeddings': True,
            'output_dir': './output/',
            'tf32': True,
            'torch_compile': True,
            'tokenizer_class': 'MeanScaleUniformBins',
            'tokenizer_kwargs': {
                'low_limit': -4.0,
                'high_limit': 4.0
            },
            'n_tokens': 2048,
            'lr_scheduler_type': 'linear',
            'warmup_ratio': 0.1,
            'dataloader_num_workers': 1,
            'max_missing_prop': 0.9,
            'use_eos_token': True,
            'seed': self.seed
        }

        # Save config file
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Training config saved to {self.config_path}")

    def train(self, gpu_id: int = 0, force_reload: bool = False):
        """Complete training pipeline (Data Preparation + Training)"""
        # Step 1: Prepare data (automatic metadata embedding)
        train_arrow, val_arrow = self.prepare_data(force_reload=force_reload)

        # Step 2: Create config file
        self.create_config(train_arrow)

        # Step 3: Start training
        logger.info("\n" + "=" * 60)
        logger.info("Starting model training...")
        logger.info("=" * 60)

        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Create output directory
        os.makedirs("./output", exist_ok=True)

        # Build training command
        cmd = [
            "python", "chronos-forecasting/scripts/training/train.py",
            "--config", self.config_path
        ]

        logger.info(f"Training command: {' '.join(cmd)}")
        logger.info(f"Using GPU: {gpu_id}\n")

        try:
            # Execute training
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("\n" + "=" * 60)
                logger.info("✓ Training completed successfully!")
                logger.info("=" * 60)
                logger.info(result.stdout)
            else:
                logger.error("\n" + "=" * 60)
                logger.error("✗ Training failed!")
                logger.error("=" * 60)
                logger.error(result.stderr)
                raise RuntimeError(f"Training failed with return code {result.returncode}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def get_latest_checkpoint(self) -> str:
        """Get the latest checkpoint"""
        output_dir = Path("./output")
        if not output_dir.exists():
            return None

        # Find the latest run directory
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
        if not run_dirs:
            return None

        # Select the latest run directory based on modification time
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)

        # Prioritize returning checkpoint-final
        checkpoint_final = latest_run / "checkpoint-final"
        if checkpoint_final.exists() and checkpoint_final.is_dir():
            return str(checkpoint_final)

        # Find folders named like checkpoint-<number>, take the largest number
        pattern = re.compile(r"^checkpoint-(\d+)$")
        numbered_checkpoints = []
        for d in latest_run.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    num = int(m.group(1))
                    numbered_checkpoints.append((num, d))
        if numbered_checkpoints:
            _, best_ckpt_dir = max(numbered_checkpoints, key=lambda t: t[0])
            return str(best_ckpt_dir)

        # Otherwise return the run directory itself
        return str(latest_run)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Chronos Model Training Script (Arrow Metadata Version - No external JSON needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Train op dataset (subject information automatically embedded in Arrow files)
  python train.py --dataset op --pred_len 12

  # Train lg dataset
  python train.py --dataset lg --pred_len 12

  # Use custom paths
  python train.py --train_dir /path/to/train --val_dir /path/to/test --pred_len 12

  # Force data re-conversion
  python train.py --dataset op --force_reload

Read metadata example:
  from train import ChronosTrainer
  metadata = ChronosTrainer.read_arrow_metadata("./processed_data/op_split2/val_data.arrow")
  print(metadata['subject_counts'])  # Number of sequences per subject
  print(metadata['subject_ids'])     # Subjects corresponding to all sequences (in order)
        """
    )

    parser.add_argument(
        "--dataset", type=str, choices=["op", "re", "lg"], default="op",
        help="Select dataset: 'op', 're' or 'lg' (default: op)")

    parser.add_argument(
        "--train_dir", type=str, default=None,
        help="Training data CSV directory")

    parser.add_argument(
        "--val_dir", type=str, default=None,
        help="Validation data CSV directory")

    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Arrow file output directory")

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")

    parser.add_argument(
        "--pred_len", type=int, default=12,
        help="Prediction length (default: 12)")

    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU ID (default: 0)")

    parser.add_argument(
        "--force_reload", action="store_true",
        help="Force re-conversion of CSV data (even if Arrow files exist)")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Create trainer
    trainer = ChronosTrainer(
        seed=args.seed,
        dataset=args.dataset,
        pred_len=args.pred_len,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
    )

    # Start full process (data preparation + training, automatic metadata embedding)
    trainer.train(gpu_id=args.gpu_id, force_reload=args.force_reload)

    # Get latest checkpoint
    latest_checkpoint = trainer.get_latest_checkpoint()
    if latest_checkpoint:
        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Latest checkpoint saved at:")
        logger.info(f"  {latest_checkpoint}")
        logger.info("=" * 60)

    # Demonstration: Reading subject metadata from Arrow file
    logger.info("\n" + "=" * 60)
    logger.info("Demonstration: Reading subject metadata from Arrow file")
    logger.info("=" * 60)
    val_arrow = trainer.output_dir / "val_data.arrow"
    if val_arrow.exists():
        metadata = ChronosTrainer.read_arrow_metadata(val_arrow)
        if metadata:
            logger.info(f"Dataset: {metadata['dataset_name']}")
            logger.info(f"Total series: {metadata['total_series']}")
            logger.info(f"Total subjects: {metadata['total_subjects']}")
            logger.info(f"First 3 subjects: {list(metadata['subject_counts'].items())[:3]}")