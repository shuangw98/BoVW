# Few-Shot Object Detection by Knowledge Distillation Using Bag-of-Visual-Words Representations, ECCV 2022

This repo is built on [FSCE](https://github.com/megvii-research/FSCE) and developed with Pytorch 1.4.0 & Torchvision 0.5.0.

## Requirements
Python == 3.6.13

Pytorch == 1.4.0

Torchvision == 0.5.0

CUDA == 10.1

### build 
```
python setup.py build develop
```

## Training and Evaluation
*  1. Train the PA-BoVW model
```
sh run_bovw.sh
```
*  2. Train the object detector
```
sh run_coco.sh
```

## Citation
If you find our code helpful in your research, please cite the following publication:
```
@inproceedings{pei2022few,
  title={Few-Shot Object Detection by Knowledge Distillation Using Bag-of-Visual-Words Representations},
  author={Pei, Wenjie and Wu, Shuang and Mei, Dianwen and Chen, Fanglin and Tian, Jiandong and Lu, Guangming},
  booktitle={European Conference on Computer Vision},
  pages={283--299},
  year={2022},
  organization={Springer}
}
```

## Contact
Please feel free to contact me (Email: wushuang9811@outlook.com) if you have any questions.