#!/usr/bin/env bash

python3 -m tools.train_net --num-gpus 4 --config-file configs/COCO/kd_tfa/base.yml 

python3 -m tools.ckpt_surgery_2  \
    --src1 checkpoints/coco/bovw/base/model_final.pth  \
    --method randinit    --save-dir checkpoints/coco/bovw/  --coco

for shot in 10 30
do
    python3 -m tools.train_net --num-gpus 4 --config-file configs/COCO/kd_tfa/${shot}shot.yml
    rm checkpoints/coco/bovw/${shot}shot/model_final.pth
done

