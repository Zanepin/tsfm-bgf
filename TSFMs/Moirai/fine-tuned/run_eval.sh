#!/bin/bash
# Moirai 1.1 Fine-Tuned Model Evaluation Script (Parameterized)
# Usage: ./run_eval.sh <checkpoint_path> <test_csv> <pred_len> <target_mean> <target_std>
# Example: ./run_eval.sh /path/to/checkpoint.ckpt glucose_op_test_full.csv 6 137.5710 53.2954
#
# Normalization parameters (target_mean / target_std) for each dataset:
#   op:  137.5710 / 53.2954
#   re:  160.2987 / 66.3162
#   lg:  147.4529 / 52.2994

set -e

CKPT_PATH="${1:?Error: Please provide checkpoint path}"
TEST_CSV="${2:?Error: Please provide test CSV filename}"
PRED_LEN="${3:?Error: Please provide prediction length (6|12)}"
TARGET_MEAN="${4:?Error: Please provide target_mean}"
TARGET_STD="${5:?Error: Please provide target_std}"

# NOTE: Update this path to your local uni2ts clone
PROJECT_ROOT="/mnt/c/Users/Administrator/Documents/task/TSFMs/uni2ts"

cd "$PROJECT_ROOT"

echo "========================================="
echo "Moirai Fine-Tuned Evaluation"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Test CSV:    ${TEST_CSV}"
echo "  Pred Length: ${PRED_LEN}"
echo "  Mean / Std:  ${TARGET_MEAN} / ${TARGET_STD}"
echo "========================================="

.venv/bin/python eval_clarke.py \
    "$CKPT_PATH" \
    "$TEST_CSV" \
    --prediction_length "$PRED_LEN" \
    --context_length 48 \
    --patch_size 16 \
    --stride 12 \
    --device cuda \
    --batch_size 256 \
    --target_mean "$TARGET_MEAN" \
    --target_std "$TARGET_STD"

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "========================================="
