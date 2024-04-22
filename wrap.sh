#!/bin/bash

parameter_sets=(
  "10.0 0.1 6 8"
  "10.0 0.1 12 12"
)

for params in "${parameter_sets[@]}"
do
  temperature=$(echo $params | cut -d' ' -f1)
  alpha=$(echo $params | cut -d' ' -f2)
  n_layer=$(echo $params | cut -d' ' -f3)
  n_head=$(echo $params | cut -d' ' -f4)

  echo "Running distill.py with temperature=$temperature  alpha=$alpha n_layer=$n_layer n_head=$n_head"
  /home/ubuntu/storage1/distilling_protgpt2/distill.py --temperature $temperature --alpha $alpha --n_layer=$n_layer --n_head=$n_head

  echo "Finshed distill.py with temperature=$temperature  alpha=$alpha n_layer=$n_layer n_head=$n_head"
  echo "------------------------------------------------------------"
done