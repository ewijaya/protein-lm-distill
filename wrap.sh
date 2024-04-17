#!/bin/bash

parameter_sets=(
  "10.0 0.1"
  "1.0 0.9"
  "1.0 0.2"
  "5.0 0.8"
)

for params in "${parameter_sets[@]}"
do
  temperature=$(echo $params | cut -d' ' -f1)
  alpha=$(echo $params | cut -d' ' -f2)

  echo "Running distill.py with temperature=$temperature and alpha=$alpha"
  /home/ubuntu/storage1/distilling_protgpt2/distill.py --temperature $temperature --alpha $alpha

  echo "Finished running distill.py with temperature=$temperature and alpha=$alpha"
  echo "------------------------------------------------------------"
done