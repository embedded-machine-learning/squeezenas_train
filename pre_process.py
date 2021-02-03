# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset as base
from torchvision import transforms
from tqdm import (
    tqdm,
) 


def get_mean_std(loader):

    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


rs_norm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class main_dataset(base):

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            image_count,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = sorted(os.listdir(images_dir))
        self.ids = self.ids[0:image_count]
        self.mask_ids = sorted(os.listdir(masks_dir))
        self.mask_ids = self.mask_ids[0:image_count]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
    
    def __getitem__(self, i):
        
        #read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)
        #crop and reshape
        image = cv2.resize(image, (2048, 1024), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (2048, 1024), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask)
        image = image.permute(2,0,1)
        mask = mask.long()

        #normalize
        image = rs_norm(image)
          
        return image, mask
        
    def __len__(self):
        return len(self.ids)

#######VISUAL###########
class vis_dataset(base):
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = sorted(os.listdir(images_dir))
        #self.ids = self.ids[0:1000]
        self.mask_ids = sorted(os.listdir(masks_dir))
        #self.mask_ids = self.mask_ids[0:1000]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
    
    def __getitem__(self, i):
        
        #read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i],0)
        #crop and reshape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

#colormap function for RailSem19 dataset
def rs_colormap(label):

    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [192, 0, 128]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [230, 150, 140]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [90, 40, 40]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 254, 254]
    colormap[18] = [0, 68, 63]
    return colormap[label]


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(30,10))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, interpolation='nearest')
    plt.show()