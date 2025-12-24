#!/bin/bash
#
# Batch training script for ProtGPT2 distillation.
# Runs multiple training configurations sequentially.
#
# Usage:
#   ./scripts/batch_train.sh
#
# Edit the parameter_sets array to define configurations to train.

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define the parameter sets: "temperature alpha n_layer n_head n_embed train_size_prop learning_rate"
parameter_sets=(
   "10.0 0.1 4 4 512 0.2 0.001"
  # Add more parameter sets here if needed
  # "2.0 0.5 4 4 256 0.1 0.001"
  # "5.0 0.3 6 6 512 0.1 0.0001"
)

# Function to run training with given parameters
run_training() {
  local temperature="$1"
  local alpha="$2"
  local n_layer="$3"
  local n_head="$4"
  local n_embed="$5"
  local train_size_prop="$6"
  local lr="$7"

  echo "========================================================================"
  echo "Training with: temp=$temperature alpha=$alpha layers=$n_layer heads=$n_head embed=$n_embed prop=$train_size_prop lr=$lr"
  echo "========================================================================"

  python "$PROJECT_ROOT/scripts/train.py" \
    --temperature "$temperature" \
    --alpha "$alpha" \
    --n_layer "$n_layer" \
    --n_head "$n_head" \
    --n_embd "$n_embed" \
    --train_size_prop "$train_size_prop" \
    --learning_rate "$lr"

  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: Training completed"
  else
    echo "ERROR: Training failed with exit code $exit_code"
  fi

  echo ""
}

# Main loop
echo "Starting batch training with ${#parameter_sets[@]} configuration(s)"
echo ""

for params in "${parameter_sets[@]}"; do
  read -r temperature alpha n_layer n_head n_embed train_size_prop lr <<< "$params"
  run_training "$temperature" "$alpha" "$n_layer" "$n_head" "$n_embed" "$train_size_prop" "$lr"
done

echo "Batch training complete!"
