#!/usr/bin/env bash

python3 bovw/coco/crop.py --train

python3 bovw/coco/generate.py

python3 -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 bovw/coco/train_bovw_cls.py 