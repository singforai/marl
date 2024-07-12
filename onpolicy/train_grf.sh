
#! /bin/bash


#=============================== compound xT jrpo ==========================
command="python train_grf.py --seed {0} --experiment_name compound_xT --group_name compound_xT --xt_type compound_xt --use_wandb"

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

parallel -j 10 $command $args_gpu0 ::: 0 1 &
parallel -j 10 $command $args_gpu1 ::: 2 3 4

wait

#=============================== compound xT ==========================
# command="python train_grf.py --seed {0} --experiment_name compound_xT --group_name compound_xT --xt_type compound_xt --use_wandb"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 0 1 &
# parallel -j 10 $command $args_gpu1 ::: 2 3 4

# wait

#=============================== base xT ==========================
# command="python train_grf.py --seed {0} --experiment_name xT --group_name base_xT --xt_type base_xt "

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 10 $command $args_gpu0 ::: 0 1 2 3 4&
# parallel -j 10 $command $args_gpu1 ::: 5 6 7 8 9


# wait

#=============================== Original ==========================
command="python train_grf.py --seed {0} --experiment_name original --group_name original --use_xt --use_wandb"

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

parallel -j 10 $command $args_gpu0 ::: 0 2 4&
parallel -j 10 $command $args_gpu1 ::: 1 3


wait