#!/usr/bin/env bash

DATASETS=(
    # sat-math

    # aqua-rat
    # math

    # lsat-ar
    # lsat-lr
    # lsat-rc

    # logiqa-en
    # logiqa-zh

    # sat-en
    # jec-qa-kd
    # jec-qa-ca

    ######################################

    # sat-en-without-passage

    # gaokao-geography
    # gaokao-history
    # gaokao-biology
    # gaokao-chemistry
    # gaokao-physics
    # gaokao-mathqa
    gaokao-english
    # gaokao-chinese
    # gaokao-mathcloze
)

SETTINGS=(zero-shot)

for d in "${DATASETS[@]}"; do
  for s in "${SETTINGS[@]}"; do
    echo "=== Running dataset=$d setting=$s ==="
    python run_prediction-api-sample-wise.py \
      --datasets "$d" \
      --settings "$s" \
      --dataset_dir data/v1_1 \
      --raw_prompt_path ./data/few_shot_prompts.csv \
      --output_dir outputs/qwen4b_rest \
      --chat_mode \
      --max_tokens 2048 \
      --skip_stage_2 \
      --skip_stage_3
  done
done

