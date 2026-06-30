#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment
source venv/bin/activate

export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_JIT=1

# Common arguments for the DiT-XL ImageNet checkpoint using a bash array
COMMON_ARGS=(
  --mode inference
  --model.hidden_size 1152
  --model.patch_size 2
  --model.depth 28
  --model.num_heads 16
  --model.mlp_ratio 4
  --dataset_name dummy
  --model.cfg_scale 1.5
  --model.class_dropout_prob 0.1
  --model.num_classes 1000
  --batch_size 1

  # NOTE: Set inference_generations to 50000 for the FULL evaluation!
  # Currently set to 2 just to test that the pipeline works without OOMing.
  --inference_generations 2
  --model.train_type shortcut
  --load_dir "Shortcut Model Checkpoints/imagenet-shortcut2-xl-fulldata-continue200000"
)

echo "==========================================="
echo "Starting Image Generation for NFE = 1"
echo "==========================================="
python train.py "${COMMON_ARGS[@]}" --inference_timesteps 1 --samples_dir "eval_samples/nfe_1"

echo "-------------------------------------------"
echo "Calculating metrics for NFE = 1..."
echo "-------------------------------------------"
CUDA_VISIBLE_DEVICES="" python calculate_metrics.py --samples_dir "eval_samples/nfe_1"

echo ""
echo "==========================================="
echo "Starting Image Generation for NFE = 2"
echo "==========================================="
python train.py "${COMMON_ARGS[@]}" --inference_timesteps 2 --samples_dir "eval_samples/nfe_2"

echo "-------------------------------------------"
echo "Calculating metrics for NFE = 2..."
echo "-------------------------------------------"
CUDA_VISIBLE_DEVICES="" python calculate_metrics.py --samples_dir "eval_samples/nfe_2"

echo "==========================================="
echo "Evaluation Pipeline Complete!"
echo "==========================================="
