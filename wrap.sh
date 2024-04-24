#!/bin/bash

# Define the parameter sets
parameter_sets=(
  "10.0 0.1 4 4 512"
  "10.0 0.1 6 8 768"
  "10.0 0.1 12 16 1024"
  # Add more parameter sets here if needed
)

# Function to run the distill_using_nferruz_dataset.py script
run_distill_script() {
  local temperature="$1"
  local alpha="$2"
  local n_layer="$3"
  local n_head="$4"
  local n_embed="$5"

  echo "Running distill_using_nferruz_dataset.py with temperature=$temperature alpha=$alpha n_layer=$n_layer n_head=$n_head n_embed=$n_embed"

  /home/ubuntu/storage1/distilling_protgpt2/distill_using_nferruz_dataset.py \
    --temperature "$temperature" \
    --alpha "$alpha" \
    --n_layer="$n_layer" \
    --n_head="$n_head" \
    --n_embd="$n_embed"

  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "Finished distill_using_nferruz_dataset.py with temperature=$temperature alpha=$alpha n_layer=$n_layer n_head=$n_head n_embed=$n_embed"
  else
    echo "Error: distill_using_nferruz_dataset.py exited with code $exit_code"
  fi

  echo "-------------------------------------------------------------------------------------------------------------------------------------"
}

# Loop through the parameter sets and run the distill script
for params in "${parameter_sets[@]}"; do
  read -r temperature alpha n_layer n_head n_embed <<< "$params"
  run_distill_script "$temperature" "$alpha" "$n_layer" "$n_head" "$n_embed"
done