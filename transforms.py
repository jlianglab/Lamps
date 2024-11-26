# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import numbers
import warnings
from collections.abc import Sequence
import numpy as np
from PIL import ImageOps, ImageFilter, Image
import torch
import random
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import _interpolation_modes_from_int
import torchvision.transforms.functional as FT
from md_aug import paint, local_pixel_shuffling,local_pixel_shuffling_500, nonlinear_transformation
import cv2
from crop import img_transforms,get_index, get_corresponding_indices
from einops import rearrange
import albumentations as A
import ipdb


def get_color_distortion(left=True):
        # p_blur = 1.0
        # p_sol = 0.0
    # s is the strength of color distortion.
    transform = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=(5,5)),
            #transforms.RandomApply([Solarization()], p=p_sol),
        ]
    )
    return transform


def mask_expander(mask, input_size=448, mask_patch_size=32, model_patch_size=4):
    scale = mask_patch_size // model_patch_size
    mask = mask.repeat(scale, axis=0).repeat(scale, axis=1)
    return mask


class Rearrange_and_Norm():
    def __call__(self, image):
        # image = cv2.resize(image, (self.size, self.size))
        image = rearrange(image, 'h w c-> c h w')/255
        return image

class DataAugmentation(object):
    def __init__(self,
                input_size=448):

        self.input_size = input_size

        #region consistency part
        self.overlap_initial_crop=transforms.RandomResizedCrop(1024, scale=(0.85,1.0),interpolation=Image.BICUBIC)


        self.random_resized_crops = []
        self.augmentations = []
        self.augmentations_glo = []
        self.augmentations_glo_noise = []
        self.augmentations_albu = []
        self.img_transforms = img_transforms()
        self.augmentations_genesis = []

        for i in range(2):
            transformList = []
            transformList.append(Rearrange_and_Norm())
            transformList.append(local_pixel_shuffling)
            transformList.append(nonlinear_transformation)
            transformList.append(transforms.RandomApply([paint], p=0.9))
            transformList.append(torch.from_numpy)
            transformList.append(transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]))
            transformSequence = transforms.Compose(transformList)
            self.augmentations_genesis.append(transformSequence)

        for i in range(2):
            transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.5),
                A.ElasticTransform(p=0.5, alpha=30, sigma=6,alpha_affine=20)
            ])
            self.augmentations_albu.append(transform)
        # Apply the transformations


        for i in range(2):
            transformList_simple=[]
            transformList_simple.append(Rearrange_and_Norm())
            transformList_simple.append(torch.from_numpy)
            transformList_simple.append(transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]))
            transformSequence_simple = transforms.Compose(transformList_simple)
            self.augmentations_glo.append(transformSequence_simple)


        for i in range(2):
            transformList_mg=[]
            transformList_mg.append(nonlinear_transformation)
            #transformList_mg.append(ElasticTransform(alpha=20, sigma=3))
            transformList_mg.append(Rearrange_and_Norm())
            transformList_mg.append(torch.from_numpy)
            transformList_mg.append(get_color_distortion(left=(i % 2 == 0)))
            transformList_mg.append(transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]))
            transformSequence_mg = transforms.Compose(transformList_mg)
            self.augmentations_glo_noise.append(transformSequence_mg)

    def coarse_to_fine_crop(self, epoch):
            y1 = 1.1*np.exp(-epoch/45)
            y2 = 1.1*np.exp(-epoch/50)
            
            return transforms.RandomResizedCrop(1024, scale=(y1,y2),interpolation=Image.BICUBIC)

    def __call__(self, image, epoch):
        crops = []
        crops_aug = []
        crops_mask = []
        crops_mask_comp = []
        local_crops = []

        # if epoch%3==2:
        #     randomcrop = self.coarse_to_fine_crop(epoch//3) # coarse to fine grain
        #     image = randomcrop(image)
        #     image = np.asarray(image)
        # else:
        image = self.overlap_initial_crop(image) # random resize and crop, (0.85,1) of initial image
        image = np.asarray(image)

        # patch, (idx_x1, idx_y1), mask, mask_complement = self.img_transforms(image) # get the crop, the top left corner indexed of two crops, mask, 1-mask
        patch, patch_local, mask, mask_complement, (x1, y1), (x2, y2), cover_rate = self.img_transforms(image) # get the crop, the top left corner indexed of two crops, mask, 1-mask
        mask = mask_expander(mask) # mask:[B,14,14]-->[B,112,112]
        
        sample_index1, sample_index2 = get_index((x1, y1), (x2, y2))
        # sample_index1, sample_index2 = get_index((idx_x1, idx_y1), (idx_x2, idx_y2), (k, l)) # the overlap mask of two crops (all 14*14)
        # print(patch.shape)
        patch1 = patch[:,:,0:3]
        patch2 = patch[:,:,3:6]
    
        # grids.append(sample_index1)
        # grids.append(sample_index2)
        # s2lmapping,l2smapping = get_corresponding_indices(sample_index1, sample_index2,(idx_x1, idx_y1), (idx_x2, idx_y2),(k, l)) # two target matrices of matrix matching, size 196*196

        for i in patch_local:
            image = self.augmentations_albu[0](image=i)['image']
            local_crops.append(self.augmentations_glo[0](image))

        #aug_whole = self.augment[0](imageData)
        # patch1 = self.augmentations_albu[0](image=patch1)['image'] # to student
        # patch2 = self.augmentations_albu[1](image=patch2)['image']

        crops.append(self.augmentations_glo[0](patch1))
        crops.append(self.augmentations_glo[1](patch2))

        # crops_aug.append(self.augmentations_genesis[0](patch1))
        # crops_aug.append(self.augmentations_genesis[1](patch2))
        crops_aug.append(self.augmentations_glo[0](self.augmentations_albu[0](image = patch1)['image']))
        crops_aug.append(self.augmentations_glo[0](self.augmentations_albu[1](image = patch2)['image']))

        crops_mask.append(mask[:,:,0])
        crops_mask.append(mask[:,:,1])

        crops_mask_comp.append(mask_complement[:,:,0])
        crops_mask_comp.append(mask_complement[:,:,1])

        # print(crops[0].type(), local_crops[0].type())
        return crops, crops_aug, local_crops, crops_mask, crops_mask_comp, sample_index1, sample_index2
        # return crops, crops_aug, local_crops, crops_mask, crops_mask_comp