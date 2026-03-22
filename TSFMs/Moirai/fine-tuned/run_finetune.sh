#!/bin/bash
# Moirai 1.1 Fine-Tuning Script (Parameterized)
# Usage: ./run_finetune.sh <dataset> <pred_len>
#   dataset:  op | re | lg
#   pred_len: 6 | 12
# Example: ./run_finetune.sh op 6

set -e

DATASET="${1:?Error: Please specify dataset (op|re|lg)}"
PRED_LEN="${2:?Error: Please specify prediction length (6|12)}"

# NOTE: Update this path to your local uni2ts clone
PROJECT_ROOT="/mnt/c/Users/Administrator/Documents/task/TSFMs/uni2ts"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"
mkdir -p logs

RUN_NAME="${DATASET}_pl${PRED_LEN}"
DATA_NAME="glucose_${DATASET}"

echo "========================================="
echo "Moirai 1.1 Fine-Tuning"
echo "  Dataset:           ${DATA_NAME}"
echo "  Prediction Length: ${PRED_LEN}"
echo "  Run Name:          ${RUN_NAME}"
echo "========================================="

$PYTHON_BIN cli/train.py \
  --config-path=conf/finetune \
  --config-name=default \
  model=moirai_1.1_R_base \
  data=${DATA_NAME} \
  val_data=${DATA_NAME} \
  exp_name=moirai_glucose \
  run_name=${RUN_NAME} \
  model.prediction_length=${PRED_LEN} \
  model.context_length=48 \
  model.patch_size=16 \
  2>&1 | tee logs/moirai_${DATASET}${PRED_LEN}_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "Training complete!"
echo "Checkpoint: outputs/finetune/moirai_glucose/moirai_1.1_R_base/full/${DATA_NAME}/${RUN_NAME}/checkpoints/"
echo "========================================="
