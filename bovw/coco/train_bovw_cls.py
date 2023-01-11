import os
import copy
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import *
import torch.nn.functional as F
from fsdet.modeling import build_resnet_backbone
from fsdet.config import get_cfg
from fsdet.layers import ShapeSpec
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from bovw import utils
from bovw.utils import convert_PIL_to_numpy, _apply_exif_orientation, read_image, GaussianBlur
from PIL import Image, ImageFilter, ImageOps, ImageEnhance 
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from bovw.checkpoint import align_and_update_state_dicts

import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_world_size
from bovw.logger import setup_logger

LR = 1e-4
OPTM_STEP = 1
SEED = 1386
N_EPOCHS = 12
BATCH_SIZE = 64
IMG_SIZE = 224
PIXEL_MEAN = [103.530, 116.280, 123.675]
PIXEL_STD = [1.0, 1.0, 1.0]


train_transforms = utils.Compose([
    utils.RandomResizedCropCoord(IMG_SIZE, scale=(0.08, 1.)),
    utils.RandomHorizontalFlipCoord(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur()], p=1.0),
])
valid_transforms = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
        ], p=1.)


def decov_loss(x):
    x_mean = x - torch.mean(x, dim=1, keepdim=True)
    x_cov = x_mean.mm(x_mean.T)
    loss = torch.norm(x_cov, p='fro') - (torch.diag(x_cov)**2).sum().sqrt()
    return 0.5 * loss


class PA_BoVW(nn.Module):
    def __init__(self, num_classes=60, num_words=1024):
        super(PA_BoVW, self).__init__()
        cfg = get_cfg()
        cfg.merge_from_file('./configs/ResNet101.yaml')
        self.resnet = build_resnet_backbone(cfg, ShapeSpec(channels=3))
        
        data = pickle.load(open("./weight/R-101.pkl", "rb"), encoding="latin1")
        new_state_dict = align_and_update_state_dicts(self.resnet.state_dict(), data)
        from collections import OrderedDict
        new_state_dict_ = OrderedDict()
        for k, v in new_state_dict.items():
            new_state_dict_[k] = torch.Tensor(new_state_dict[k])
        self.resnet.load_state_dict(new_state_dict_, strict=False)

        self.adapter = nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_words, num_classes)
        
        self.vocab = nn.Parameter(torch.Tensor(num_words, 512), requires_grad=True)
        
        nn.init.kaiming_normal_(self.vocab, a=np.sqrt(5))
    
    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.inference(x)
    
    def forward_train(self, x):

        feat = self.resnet(x)['res5']
        feat = self.adapter(feat)
        
        x_norm = F.normalize(feat, p=2, dim=1)
        vocab_norm = F.normalize(self.vocab, p=2, dim=1)
        vocab_norm = vocab_norm.unsqueeze(2).unsqueeze(3)
        dist = F.conv2d(x_norm, weight=vocab_norm)
        feat_pool = self.avgpool(dist)
        feat_pool = torch.flatten(feat_pool, 1)
        pred_cls = self.fc(feat_pool)
        
        loss_decov = decov_loss(self.vocab)
        
        return pred_cls, loss_decov
    
    def inference(self, x):
        x = self.resnet(x)['res5']
        x = self.adapter(x)
        x_norm = F.normalize(x, p=2, dim=1)
        vocab_norm = F.normalize(self.vocab, p=2, dim=1)
        vocab_norm = vocab_norm.unsqueeze(2).unsqueeze(3)
        dist = F.conv2d(x_norm, weight=vocab_norm)
        feat_pool = self.avgpool(dist)
        feat_pool = torch.flatten(feat_pool, 1)
        pred_cls = self.fc(feat_pool)
        return pred_cls


class CLSDataset(Dataset):
    def __init__(self, txt_file, base_dir='./datasets/coco/crop_instance/', transforms=None, training=True):
        f = open(txt_file, 'r')
        lines = f.readlines()
        self.base_dir = base_dir
        self.img_names = []
        self.labels = []
        self.transforms = transforms
        self.train_transforms = train_transforms
        self.training = training
        num_channels = len(PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(PIXEL_MEAN)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(PIXEL_STD)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        for line in lines:
            self.img_names.append(line.split(';')[0])
            self.labels.append(int(line.split(';')[1]))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        p = self.img_names[idx]
        label = self.labels[idx]
        p_path = self.base_dir + p
        image = Image.open(open(p_path, 'rb'))
        image = image.convert('RGB')
        if self.transforms:
            if self.training:
                image, _ = self.transforms(image)
                image = np.asarray(image)
                image = image[:, :, ::-1]
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                image = self.normalizer(image)
                return image, label
            else:
                image = np.asarray(image)
                image = image[:, :, ::-1]
                transformed = self.transforms(image=image)
                image = transformed['image']
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image = self.normalizer(image)
        return image, label


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)


def train_model(model, epoch, dataloader_train, criterion, optimizer):

    model.train() 
    
    losses_cls = AverageMeter()
    losses_decov = AverageMeter()
    accs = AverageMeter()
    
    optimizer.zero_grad()
    for idx, (img, labels) in enumerate(dataloader_train):
        
        images, labels = img.cuda(), labels.cuda().long()
        predicted, loss_decov = model(images)

        loss_cls = criterion(predicted, labels)
        loss_decov = loss_decov.mean()
        loss = loss_cls + loss_decov
        loss.backward()
        
        if (idx+1) % OPTM_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()
        predicted_classes = predicted.argmax(1)
        correctly_identified_sum = (predicted_classes==labels).sum().item()
        number_of_images = images.size(0)
                                    
        accs.update(correctly_identified_sum/number_of_images, number_of_images)
        losses_cls.update(loss_cls.item(), number_of_images)
        losses_decov.update(loss_decov.item(), number_of_images)
        if idx % 10 == 0:
            logger.info(
                f'Train: [{epoch}/{N_EPOCHS}][{idx}/{len(dataloader_train)}]  '
                f'loss_cls: {losses_cls.avg:.3f} loss_decov: {losses_decov.avg:.3f} '
                f'lr {optimizer.param_groups[0]["lr"]:.6f} ')
        
    return losses_cls.avg, accs.avg


def test_model(model, dataloader_valid, criterion):    
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader_valid):
            images, labels = images.cuda(), labels.cuda().long() 
            output_valid = model(images)
            loss = criterion(output_valid, labels)
            losses.update(loss.item(), images.size(0))
            accs.update((output_valid.argmax(1)==labels).sum().item()/images.size(0),images.size(0))
            if idx % 10 == 0:
                logger.info(
                    f'Test: [{idx}/{len(dataloader_valid)}]  '
                    f'loss_cls: {losses.avg:.3f}  '
                    f'accuracy {accs.avg:.6f} ')
        logger.info(f'Final accuracy: {accs.avg:.6f}')
    return losses.avg, accs.avg, loss
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True
    
    
    base_dir = './weight/coco/pabovw'
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    TRAINING = True
    model = PA_BoVW(60, 1024)
    model = model.cuda()
    
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    
    logger = setup_logger(output=base_dir, distributed_rank=dist.get_rank(), name="pabovw")
    
    if TRAINING:
        dataset_train = CLSDataset('./weight/coco/train.txt',
            transforms=train_transforms, training=True)
        dataset_valid = CLSDataset('./weight/coco/test.txt', 
            transforms=valid_transforms, training=False)
        
        sampler_train = DistributedSampler(dataset_train)

        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE,
            num_workers=16, shuffle=False, sampler=sampler_train)
        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, 
            num_workers=16, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss().cuda()
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[9, 11])
        best_acc = 0
        best_loss = 100

        for epoch in range(1, N_EPOCHS+1):
            if isinstance(dataloader_train.sampler, DistributedSampler):
                dataloader_train.sampler.set_epoch(epoch)
            train_loss, train_acc = train_model(model, epoch, dataloader_train, criterion, optimizer)
            val_loss, val_acc, loss = test_model(model, dataloader_valid, criterion)

            scheduler.step()

            if val_acc > best_acc and dist.get_rank() == 0:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(base_dir, 'model_best.pth'))
            logger.info(f'current_val_acc: {val_acc:.6f}  best_val_acc: {best_acc:.6f}  ')
