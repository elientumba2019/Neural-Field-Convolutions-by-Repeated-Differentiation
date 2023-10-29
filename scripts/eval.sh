#!/bin/bash

# modality :
# 1 = 2d, 2 = 3d, 3 = video
# In this implementation, modalities 2 and 3, do not require optimized kernel
# When modality = 1 only height and width are considered

cd "../"
pwd

# image model
python eval.py \
--model_path="./trained_models/images/model.pth" \
--kernel_path='./trained_models/kernel/gaussian' \
--save_path="./output/images" \
--modality=1 \
--width=512 \
--height=512 \
--depth=100 \
--block_size=32 \
--kernel_scale=20


# geometry model
#python eval.py \
#--model_path="./trained_models/geometry/model.pth" \
#--kernel_path='./trained_models/kernel/gaussian' \
#--save_path="./output/geometry" \
#--modality=2 \
#--width=256 \
#--height=256 \
#--depth=256 \
#--block_size=32 \
#--kernel_scale=20


# video model
#python eval.py \
#--model_path="./trained_models/videos/model.pth" \
#--kernel_path='./trained_models/kernel/gaussian' \
#--save_path="./output/videos" \
#--modality=3 \
#--width=256 \
#--height=256 \
#--depth=100 \
#--block_size=32 \
#--kernel_scale=20
#
