"""
Moirai 1.1 Fine-Tuning Data Processing Script
Process OP, RE and LG datasets with 90/10 train/val split.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import datasets
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from uni2ts.common.env import env

# Dataset configuration
DATASETS = {
    "op": Path("/mnt/d/glucose_data/internal/op_split2"),
    "re": Path("/mnt/d/glucose_data/internal/re_split2"),
    "lg": Path("/mnt/d/glucose_data/internal/lg_split2"),
}

FREQ = "5T"
MIN_LENGTH = 600  # Minimum sequence length filter (600 points = 50 hours)

def process_file(file_path):
    """
    Process a single CSV file.
    Returns: dict or None
    """
    try:
        item_id = Path(file_path).stem
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'full_time' not in df.columns or 'GlucoseValue' not in df.columns:
            return None
            
        # Parse timestamp - detect format via regex
        first_time = str(df['full_time'].iloc[0])
        if re.match(r'\d{4}/\d{2}/\d{2}', first_time):
            # Slash format (op/re datasets)
            df['full_time'] = pd.to_datetime(df['full_time'], format='%Y/%m/%d %H:%M:%S')
        elif re.match(r'\d{4}-\d{2}-\d{2}', first_time):
            # Dash format (lg dataset)
            df['full_time'] = pd.to_datetime(df['full_time'], format='%Y-%m-%d %H:%M:%S')
        else:
            raise ValueError(f"Unrecognized timestamp format: {first_time}")
        df = df.sort_values('full_time')
        df = df.set_index('full_time')
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Resample to 5-minute frequency
        df = df.asfreq(FREQ)
        
        # Linear interpolation (max 12 consecutive gaps)
        df['GlucoseValue'] = df['GlucoseValue'].interpolate(method='linear', limit=12)
        
        # Skip if all NaN
        if df['GlucoseValue'].isna().all():
            return None
            
        # Forward and backward fill
        df['GlucoseValue'] = df['GlucoseValue'].bfill().ffill()
        
        target = df['GlucoseValue'].values.astype(np.float32)
        start = df.index[0]
        
        # Length filter
        if len(target) < MIN_LENGTH:
            return None
            
        return {
            "item_id": item_id,
            "start": start,
            "target": target,
            "freq": FREQ
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def split_train_val_by_subject(train_items, val_ratio=0.1):
    """
    For op/re datasets: split validation set by file.
    Assign the last 10% of files per subject to validation.
    
    Args:
        train_items: List of processed items
        val_ratio: Validation ratio (default 0.1)
    
    Returns:
        train_items_split, val_items_split
    """
    # Group by Subject ID
    subject_groups = defaultdict(list)
    for item in train_items:
        # item_id format: {subject_id}_{num_id}
        subject_id = item['item_id'].split('_')[0]
        subject_groups[subject_id].append(item)
    
    train_split = []
    val_split = []
    
    print(f"\nSplit strategy: last {int(val_ratio*100)}% sequences per subject as validation")
    
    for subject_id, items in subject_groups.items():
        num_items = len(items)
        num_val = max(1, int(num_items * val_ratio))  # Keep at least 1 validation sample
        
        # Sort by item_id (ensures temporal order)
        items_sorted = sorted(items, key=lambda x: x['item_id'])
        
        # Split
        train_items_subj = items_sorted[:-num_val]
        val_items_subj = items_sorted[-num_val:]
        
        train_split.extend(train_items_subj)
        val_split.extend(val_items_subj)
        
        print(f"  Subject {subject_id}: {len(train_items_subj)} train, {len(val_items_subj)} val")
    
    return train_split, val_split

def split_train_val_by_datapoints(train_items, val_ratio=0.1):
    """
    For lg dataset: split validation set by data points.
    Use the last 10% of data points from each subject's last file.
    
    Args:
        train_items: List of processed items
        val_ratio: Validation ratio (default 0.1)
    
    Returns:
        train_items_split, val_items_split
    """
    # Group by Subject ID
    subject_groups = defaultdict(list)
    for item in train_items:
        subject_id = item['item_id'].split('_')[0]
        subject_groups[subject_id].append(item)
    
    train_split = []
    val_split = []
    
    print(f"\nSplit strategy (lg): last {int(val_ratio*100)}% data points from each subject's last file as validation")
    
    for subject_id, items in subject_groups.items():
        # Sort by item_id
        items_sorted = sorted(items, key=lambda x: x['item_id'])
        
        # All earlier files go to training
        train_split.extend(items_sorted[:-1])
        
        # Split last file by data points
        last_item = items_sorted[-1]
        target = last_item['target']
        seq_len = len(target)
        
        # Compute split index
        split_idx = int(seq_len * (1 - val_ratio))
        
        # Create training portion (first 90%)
        train_item = {
            'item_id': last_item['item_id'] + '_train',
            'start': last_item['start'],
            'target': target[:split_idx],
            'freq': last_item['freq']
        }
        train_split.append(train_item)
        
        # Create validation portion (last 10%)
        # Compute validation start time
        val_start = last_item['start'] + pd.Timedelta(minutes=5*split_idx)
        val_item = {
            'item_id': last_item['item_id'] + '_val',
            'start': val_start,
            'target': target[split_idx:],
            'freq': last_item['freq']
        }
        val_split.append(val_item)
        
        print(f"  Subject {subject_id}: {len(items_sorted)-1} full files + 1 split file (train={split_idx} pts, val={seq_len-split_idx} pts)")
    
    return train_split, val_split

def create_hf_dataset(data_list):
    """Create a HuggingFace dataset from a list of items."""
    def gen():
        for item in data_list:
            yield item
    
    features = datasets.Features({
        "item_id": datasets.Value("string"),
        "start": datasets.Value("timestamp[s]"),
        "freq": datasets.Value("string"),
        "target": datasets.Sequence(datasets.Value("float32")),
    })
    
    return datasets.Dataset.from_generator(gen, features=features)

def process_dataset(dataset_name, dataset_path):
    """
    Process a single dataset (OP, RE, or LG).
    
    Args:
        dataset_name: 'op', 're', or 'lg'
        dataset_path: Path to dataset directory
    
    Returns:
        dict with 'train', 'val', 'test' keys
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Process Train directory (to be split into train + val)
    train_dir = dataset_path / "Train"
    train_files = glob.glob(str(train_dir / "*.csv"))
    print(f"\n[Train] Found {len(train_files)} files")
    
    train_items = []
    for f in tqdm(train_files, desc="Processing Train"):
        item = process_file(f)
        if item:
            train_items.append(item)
    
    print(f"[Train] Kept {len(train_items)} / {len(train_files)} (filtered < {MIN_LENGTH})")
    
    # Split train/val (strategy depends on dataset type)
    if train_items:
        if dataset_name == 'lg':  # lg dataset: split by data points
            train_split, val_split = split_train_val_by_datapoints(train_items, val_ratio=0.1)
        else:  # op/re datasets: split by file
            train_split, val_split = split_train_val_by_subject(train_items, val_ratio=0.1)
        results['train'] = train_split
        results['val'] = val_split
        print(f"\nFinal split: Train={len(train_split)}, Val={len(val_split)}")
    else:
        raise ValueError(f"Dataset {dataset_name} Train directory has no valid samples!")
    
    # 2. Process Test directory
    test_dir = dataset_path / "Test"
    test_files = glob.glob(str(test_dir / "*.csv"))
    print(f"\n[Test] Found {len(test_files)} files")
    
    test_items = []
    for f in tqdm(test_files, desc="Processing Test"):
        item = process_file(f)
        if item:
            test_items.append(item)
    
    print(f"[Test] save {len(test_items)} / {len(test_files)}")
    results['test'] = test_items
    
    return results

def normalize_and_save(dataset_name, data_dict, save_base_path):
    """
    Apply global normalization and save dataset.
    
    Args:
        dataset_name: 'op', 're', or 'lg'
        data_dict: {'train': [...], 'val': [...], 'test': [...]}
        save_base_path: Output directory path
    """
    print(f"\n{'='*60}")
    print(f"Normalizing & Saving: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Compute global statistics (training set only)
    train_items = data_dict['train']
    if not train_items:
        raise ValueError(f"Dataset {dataset_name} has no training samples!")
    
    print("Computing global statistics (from training set)...")
    all_values = np.concatenate([item['target'] for item in train_items])
    mean = np.mean(all_values)
    std = np.std(all_values)
    print(f"Global Mean: {mean:.4f}, Global Std: {std:.4f}")
    
    if std < 1e-8:
        std = 1.0
        print("Warning: std is too small, setting to 1.0")
    
    # Normalize all splits
    for split_name in ['train', 'val', 'test']:
        items = data_dict[split_name]
        if not items:
            print(f"Warning: {split_name} split is empty, skipping")
            continue
        
        print(f"\nNormalizing {split_name}...")
        for item in items:
            item['target'] = (item['target'] - mean) / std
        
        # Save
        dataset_full_name = f"glucose_{dataset_name}_{split_name}"
        print(f"Saving {dataset_full_name} ({len(items)} samples)...")
        
        hf_dataset = create_hf_dataset(items)
        hf_dataset.info.dataset_name = dataset_full_name
        
        save_path = save_base_path / dataset_full_name
        hf_dataset.save_to_disk(save_path)
        print(f"Saved to: {save_path}")

def main():
    """Main function."""
    # Check environment variable
    if env.CUSTOM_DATA_PATH is None:
        raise ValueError("Environment variable CUSTOM_DATA_PATH is not set. Check .env file.")
    
    # Set save path
    save_base_path = env.CUSTOM_DATA_PATH / "lsf" / "wide"
    save_base_path.mkdir(parents=True, exist_ok=True)
    print(f"Dataset save path: {save_base_path}")
    
    # Process all datasets
    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            print(f"\nWarning: Dataset path does not exist: {dataset_path}")
            print(f"Skipping {dataset_name} dataset")
            continue
        
        # Process
        data_dict = process_dataset(dataset_name, dataset_path)
        
        # Normalize and save
        normalize_and_save(dataset_name, data_dict, save_base_path)
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
