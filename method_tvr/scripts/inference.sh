#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash method/scripts/inference.sh ANY_OTHER_PYTHON_ARGS
model_dir=$1
eval_split_name=$2  # [val]
eval_path=data/tvr_${eval_split_name}_release.jsonl
tasks=()
tasks+=(VCMR)
tasks+=(SVMR)
tasks+=(VR)
echo "tasks ${tasks[@]}"
python method_tvr/inference.py \
--model_dir ${model_dir} \
--tasks ${tasks[@]} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
