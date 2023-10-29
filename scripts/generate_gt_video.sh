#!/bin/bash

# cd "/anaconda/etc/profile.d"
# source conda.sh

cd "../utilities"
pwd

python mc_utils.py \
--path="../data/raw/videos" \
--sample_number=100 \
--save_path="../data/GTs/videos" \
--half_size=0.3 \
--order=1 \

