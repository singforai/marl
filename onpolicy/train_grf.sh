
#! /bin/bash


command="python train_grf.py --seed 0 --experiment_name newmodel --group_name newmodel --algorithm_name newmodel  --num_gpu 6  --level_dir level/level.json --scenario_name curriculum_learning"

$command 

