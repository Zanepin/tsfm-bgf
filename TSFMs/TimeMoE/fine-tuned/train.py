import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
from collections import defaultdict
import torch
import sys

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


class TimeMoEDataPreparator:
    """Prepare Time-MoE training data"""
    
    def __init__(
            self,
            dataset: str = "op",
            train_dir: str = None,
            val_dir: str = None,
            output_dir: str = None,
            pred_len: int = 12,
            seed: int = 42,
    ):
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Terminating process.")
            sys.exit(1)
        
        self.dataset = dataset
        self.pred_len = pred_len
        self.seed = seed
        
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
        
        self.train_jsonl_path = self.output_dir / "train_data.jsonl"
        self.val_jsonl_path = self.output_dir / "val_data.jsonl"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        set_seed(seed)
        
        logger.info(f"Data Preparator initialized:")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Train directory: {self.train_dir}")
        logger.info(f"  Validation directory: {self.val_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"  Prediction length: {pred_len}")

    @staticmethod
    def _extract_subject_id(filename: str) -> str:
        """Extract subject ID from filename"""
        parts = filename.split('_')
        return parts[0] if parts else filename

    def load_csv_to_sequences(self, input_dir: Path) -> tuple:
        """Load time series data from CSV files"""
        logger.info(f"Loading CSV files from {input_dir}")
        
        if not input_dir.exists():
            raise ValueError(f"Directory {input_dir} does not exist")
        
        csv_files = sorted(list(input_dir.glob("*.csv")), key=lambda x: x.stem)
        logger.info(f"Found {len(csv_files)} CSV files")
        
        sequences = []
        subject_ids = []
        successful_loads = 0
        failed_loads = 0
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'GlucoseValue' not in df.columns:
                    logger.warning(f"File {csv_file.name}: 'GlucoseValue' column not found")
                    failed_loads += 1
                    continue
                
                glucose_values = df['GlucoseValue'].values
                glucose_values = glucose_values[~np.isnan(glucose_values)]
                
                if len(glucose_values) == 0:
                    logger.warning(f"File {csv_file.name}: No valid data")
                    failed_loads += 1
                    continue
                
                subject_id = self._extract_subject_id(csv_file.stem)
                
                sequences.append(glucose_values.tolist())
                subject_ids.append(subject_id)
                successful_loads += 1
                
                if successful_loads % 1000 == 0:
                    logger.info(f"Loaded {successful_loads} files")
            
            except Exception as e:
                logger.warning(f"Error loading {csv_file.name}: {e}")
                failed_loads += 1
        
        logger.info(f"Data loading complete:")
        logger.info(f"  Successfully loaded: {successful_loads} files")
        logger.info(f"  Failed: {failed_loads} files")
        
        if len(sequences) == 0:
            raise ValueError("No valid data loaded")
        
        subject_counts = defaultdict(int)
        for subject_id in subject_ids:
            subject_counts[subject_id] += 1
        
        return sequences, dict(sorted(subject_counts.items()))

    def save_to_jsonl(self, sequences: List[List[float]], output_path: Path):
        """Save as Time-MoE JSONL format"""
        logger.info(f"Saving {len(sequences)} sequences to {output_path}")
        
        with open(output_path, 'w') as f:
            for seq in sequences:
                json_obj = {"sequence": seq}
                f.write(json.dumps(json_obj) + '\n')
        
        logger.info(f"✅ Data saved to {output_path}")

    def prepare_data(self):
        """Prepare training and validation data"""
        logger.info("\n" + "="*60)
        logger.info("Preparing training data...")
        train_sequences, train_subjects = self.load_csv_to_sequences(self.train_dir)
        self.save_to_jsonl(train_sequences, self.train_jsonl_path)
        
        logger.info("\n" + "="*60)
        logger.info("Preparing validation data...")
        val_sequences, val_subjects = self.load_csv_to_sequences(self.val_dir)
        self.save_to_jsonl(val_sequences, self.val_jsonl_path)
        
        metadata = {
            'dataset': self.dataset,
            'model_type': 'time-moe',
            'model_path': 'Maple728/TimeMoE-200M',
            'prediction_length': self.pred_len,
            'num_train_sequences': len(train_sequences),
            'num_val_sequences': len(val_sequences),
            'num_train_subjects': len(train_subjects),
            'num_val_subjects': len(val_subjects),
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'train_data_path': str(self.train_jsonl_path),
            'val_data_path': str(self.val_jsonl_path),
            'checkpoint_dir': str(self.checkpoint_dir),
            'seed': self.seed,
            'stride': 12,
            'attn_implementation': 'eager',
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("Data preparation completed!")
        logger.info(f"  Train sequences: {len(train_sequences)}")
        logger.info(f"  Val sequences: {len(val_sequences)}")
        logger.info(f"  Train subjects: {len(train_subjects)}")
        logger.info(f"  Val subjects: {len(val_subjects)}")
        logger.info(f"  Train data: {self.train_jsonl_path}")
        logger.info(f"  Val data: {self.val_jsonl_path}")
        logger.info(f"  Metadata: {metadata_path}")
        logger.info(f"{'='*60}\n")
        
        return str(self.train_jsonl_path), str(self.val_jsonl_path)


def generate_training_command(
        data_path: str,
        checkpoint_dir: str,
        model_path: str = "Maple728/TimeMoE-200M",
        from_scratch: bool = False,
        stride: int = 12,
        seed: int = 42,
        use_multi_gpu: bool = False,
) -> str:
    """Generate Time-MoE training command"""
    timemoe_main = "./time-MOE/main.py"
    
    if not os.path.exists(timemoe_main):
        logger.warning(f"Time-MoE training script not found: {timemoe_main}")
        logger.warning("Please ensure the time-MOE repository is cloned in the current directory")
    
    if use_multi_gpu:
        base_cmd = f"python ./time-MOE/torch_dist_run.py {timemoe_main}"
    else:
        base_cmd = f"python {timemoe_main}"
    
    cmd_parts = [
        base_cmd,
        f"--data_path {data_path}",
        f"--output_path {checkpoint_dir}",
        f"--model_path {model_path}",
        f"--stride {stride}",
        f"--seed {seed}",
        "--attn_implementation eager",
        "--micro_batch_size 4",
        "--global_batch_size 32",
        "--precision bf16",
    ]
    
    if from_scratch:
        cmd_parts.append("--from_scratch")
    
    return " ".join(cmd_parts)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Time-MoE Blood Glucose Prediction Fine-tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python train.py --dataset op --pred_len 12                    # Start training
  python train.py --dataset op --pred_len 12 --prepare_only    # Prepare data only
  python train.py --dataset op --pred_len 12 --multi_gpu       # Multi-GPU training
        """
    )
    
    parser.add_argument("--dataset", type=str, choices=["op", "re", "lg"], default="op",
                        help="Dataset selection")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Training data CSV directory")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Validation data CSV directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./output/{dataset}_pred{len})")
    parser.add_argument("--pred_len", type=int, default=12,
                        help="Prediction length")
    parser.add_argument("--model_path", type=str, default="Maple728/TimeMoE-200M",
                        help="Time-MoE model path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--stride", type=int, default=12,
                        help="Data sampling stride")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train from scratch")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use multi-GPU training")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Prepare data only, do not start training")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize Data Preparator
    preparator = TimeMoEDataPreparator(
        dataset=args.dataset,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        pred_len=args.pred_len,
        seed=args.seed,
    )
    
    # Prepare data
    train_data_path, val_data_path = preparator.prepare_data()
    
    # Generate training command
    training_cmd = generate_training_command(
        data_path=train_data_path,
        checkpoint_dir=str(preparator.checkpoint_dir),
        model_path=args.model_path,
        from_scratch=args.from_scratch,
        stride=args.stride,
        seed=args.seed,
        use_multi_gpu=args.multi_gpu,
    )
    
    logger.info("\n" + "="*60)
    logger.info("Training Command:")
    logger.info(f"  {training_cmd}")
    logger.info("="*60 + "\n")
    
    # Save training command to file
    cmd_file = preparator.output_dir / "training_command.sh"
    with open(cmd_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated training command for {args.dataset} pred_len={args.pred_len}\n")
        f.write(f"# Model: {args.model_path}\n")
        f.write(f"# Stride: {args.stride}\n")
        f.write(f"# Seed: {args.seed}\n\n")
        f.write("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n\n")
        f.write(training_cmd + "\n")
    
    # Update and save metadata
    metadata_path = preparator.output_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metadata['model_path'] = args.model_path
    metadata['stride'] = args.stride
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training command saved to: {cmd_file}")
    logger.info(f"You can run it manually with: bash {cmd_file}\n")
    
    if not args.prepare_only:
        logger.info("Starting training...\n")
        
        # Check Time-MoE repository
        timemoe_main = "./time-MOE/main.py"
        if not os.path.exists(timemoe_main):
            logger.error("\n" + "="*60)
            logger.error("ERROR: Time-MoE training script not found!")
            logger.error(f"Expected path: {timemoe_main}")
            logger.error("\nPlease clone the Time-MoE repository:")
            logger.error("  git clone https://github.com/Time-MoE/Time-MoE.git time-MOE")
            logger.error("="*60 + "\n")
            sys.exit(1)
        
        import subprocess
        try:
            logger.info(f"Executing: {training_cmd}\n")
            subprocess.run(training_cmd, shell=True, check=True)
            logger.info("\n✅ Training completed successfully!")
            logger.info(f"Checkpoints saved to: {preparator.checkpoint_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"\n❌ Training failed with error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Training interrupted by user")
            sys.exit(1)
    else:
        logger.info("Data preparation completed.")
        logger.info(f"To start training, run: bash {cmd_file}")