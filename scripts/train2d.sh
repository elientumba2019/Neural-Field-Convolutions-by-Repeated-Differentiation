#!/bin/bash

#cd "/anaconda/etc/profile.d"
#source conda.sh
#conda activate elie_local

cd "../experiments"
pwd

# train network based on mode for repeated integration
python train_2d_neural_integral_field.py \
--summary "../logs" \
--experiment_name "Experiment2d" \
--batch 16384 \
--num-steps 300000 \
--num_channels 256 \
--num_layers 4 \
--schedule_gamma 0.5 \
--schedule_step 10000 \
--pe 4 \
--learn_rate 1e-3 \
--workers 12 \
--norm_exp 0 \
--monte_carlo "../data/GTs/images/GT_0.05.npy" \
--kernel_scale 20 \
--order 1 \

# to resume training set this to a checkpoint's path
# --init_ckpt ""


