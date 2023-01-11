import sys
import os
import glob
import random
import re

BASE_NAME2LALBEL = {'truck': 0, 'traffic light': 1, 'fire hydrant': 2, 'stop sign': 3,
                    'parking meter': 4, 'bench': 5, 'elephant': 6, 'bear': 7, 'zebra': 8,
                    'giraffe': 9, 'backpack': 10, 'umbrella': 11, 'handbag': 12, 'tie': 13, 
                    'suitcase': 14, 'frisbee': 15, 'skis': 16, 'snowboard': 17, 'sports ball': 18, 
                    'kite': 19, 'baseball bat': 20, 'baseball glove': 21, 'skateboard': 22, 
                    'surfboard': 23, 'tennis racket': 24, 'wine glass': 25, 'cup': 26, 'fork': 27, 
                    'knife': 28, 'spoon': 29, 'bowl': 30, 'banana': 31, 'apple': 32, 'sandwich': 33, 'orange': 34, 'broccoli': 35, 'carrot': 36, 'hot dog': 37, 'pizza': 38, 'donut': 39, 'cake': 40, 'bed': 41, 'toilet': 42, 'laptop': 43, 'mouse': 44, 'remote': 45, 'keyboard': 46, 'cell phone': 47, 'microwave': 48, 'oven': 49, 'toaster': 50, 'sink': 51, 'refrigerator': 52, 'book': 53, 'clock': 54, 'vase': 55, 'scissors': 56, 'teddy bear': 57, 'hair drier': 58, 'toothbrush': 59}

NAME2ID = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}

if __name__ == "__main__":
    if not os.path.exists("./weight/coco"):
        os.makedirs("./weight/coco")
    TRAIN_SAVE_PATH = "./weight/coco/train.txt"
    TEST_SAVE_PATH = "./weight/coco/test.txt"

    BASE_PATH = './datasets/coco/crop_instance/'
    SEPARATOR = ';'
    train_fh = open(TRAIN_SAVE_PATH,'w')
    test_fh = open(TEST_SAVE_PATH,'w')

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            if subdirname not in NAME2ID.keys():
                continue
            if NAME2ID[subdirname] in [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]:
                continue
            label = BASE_NAME2LALBEL[subdirname]
            subject_path = os.path.join(dirname, subdirname)
            file_list = os.listdir(subject_path)
            file_list = [name for name in file_list if name.endswith('.jpg')]
            print(len(file_list))
            random.shuffle(file_list)
            train_len = int(0.9 * len(file_list))
            for filename in file_list[:train_len]:
                train_fh.write(subdirname + '/' + filename + SEPARATOR + str(label) + "\n")
            for filename in file_list[train_len:]:
                test_fh.write(subdirname + '/' + filename + SEPARATOR + str(label) + "\n")
    train_fh.close()
    test_fh.close()
