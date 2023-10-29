#!/bin/bash

# cd "/anaconda/etc/profile.d"
# source conda.sh

cd "../utilities"
pwd

python mc_utils.py \
--path="../data/raw/images/1.jpg" \
--sample_number=50 \
--save_path="../data/GTs/images" \
--half_size=0.05 \
--order=1 \

