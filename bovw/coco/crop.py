import cv2
import os
import numpy as np
from pycocotools.coco import COCO

def crop_object(img, bbox, target_size=(224, 224)):
    
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    width, height = x2 - x1, y2 - y1
    
    new_x1 = 0
    new_y1 = 0
    new_x2 = width
    new_y2 = height
    
    square = img[y1:y2+1, x1:x2+1]

    square = square.astype(np.float32, copy=False)
    square_scale = float(target_size[0]) / max(width, height)
    square = cv2.resize(square, target_size, interpolation=cv2.INTER_LINEAR)
    square = square.astype(np.uint8)

    new_x1 = int(new_x1 * square_scale)
    new_y1 = int(new_y1 * square_scale)
    new_x2 = int(new_x2 * square_scale)
    new_y2 = int(new_y2 * square_scale)

    support_data = square
    support_box = np.array([new_x1, new_y1, new_x2, new_y2]).astype(np.float32)
    return support_data, support_box


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train",action="store_true")
    args = parser.parse_args()
    
    TRAIN = args.train
    
    base_dir = './datasets/coco/'
    
    if TRAIN:
        annFile = './datasets/cocosplit/datasplit/trainvalno5k.json'
    else:
        annFile = './datasets/cocosplit/datasplit/5k.json'

    coco = COCO(annFile)
    categories = coco.loadCats(coco.getCatIds())
    classes             = {}
    coco_labels         = {}
    coco_labels_inverse = {}
    labels              = {}
    name_id             = {}
    id_name             = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']           
        coco_labels_inverse[c['id']] = len(classes)  
        classes[c['name']] = len(classes)            
        name_id[c['name']] = c['id']                
        id_name[c['id']] = c['name']                 
        labels[len(classes)] = c['name']            
        if TRAIN:
            save_dir = os.path.join(base_dir, 'crop_instance', c['name'])
        else:
            save_dir = os.path.join(base_dir, 'crop_instance_test', c['name'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    for img_id, id in enumerate(coco.imgs):
        if img_id % 100 == 0:
            print(img_id)
        img = coco.loadImgs(id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        set_img_base_path = os.path.join(base_dir, 'trainval2014')
        im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
        for item_id, ann in enumerate(anns):
            if TRAIN:
                if ann['category_id'] in [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]:
                    continue
            rect = ann['bbox']
            if rect[2] <= 4 or rect[3] <= 4:
                print(rect)
                continue
            bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

            crop_img, crop_box = crop_object(im, bbox)
            if TRAIN:
                file_path = os.path.join(base_dir, 'crop_instance', id_name[ann['category_id']], '{:04d}.jpg'.format(ann['id']))
            else:
                file_path = os.path.join(base_dir, 'crop_instance_test', id_name[ann['category_id']], '{:04d}.jpg'.format(ann['id']))
            cv2.imwrite(file_path, crop_img)
    