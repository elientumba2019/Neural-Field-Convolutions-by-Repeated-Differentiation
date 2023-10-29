#!/bin/bash

#cd "/anaconda/etc/profile.d"
#source conda.sh
#conda activate elie_local

cd "../experiments"
pwd

# train network based on mode for repeated integration
python train_video_neural_integral_field.py \
--summary "../logs" \
--experiment_name "VideoExperiment" \
--batch 16384 \
--num-steps 300000 \
--num_channels 256 \
--num_layers 5 \
--in_channel 3 \
--schedule_gamma 0.5 \
--schedule_step 30000 \
--pe 4 \
--fit_type 1 \
--down_coeff 1 \
--learn_rate 1e-3 \
--workers 12 \
--norm_exp 0 \
--monte_carlo "../data/GTs/videos/GT_0.05.npy" \
--kernel_scale 20 \
--order 1 \

# to resume training set this to a checkpoint's path
#--init_ckpt ""

