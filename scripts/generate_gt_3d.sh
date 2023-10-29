#!/bin/bash

# cd "/anaconda/etc/profile.d"
# source conda.sh

cd "../utilities"
pwd

python mc_utils.py \
--path="../data/raw/geometry/armadillo.obj" \
--sample_number=5 \
--save_path="../data/GTs/geometry" \
--half_size=0.07 \
--order=0 \

