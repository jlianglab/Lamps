

import torch.utils.data
import random
from os.path import isfile, join
from torchvision import transforms
from PIL import Image
import csv
import numpy as np
import re
import random
import copy
from torch.utils.data import Dataset
import os
from einops import rearrange
from md_aug import paint, local_pixel_shuffling,local_pixel_shuffling_500, nonlinear_transformation
import cv2
import time
from pydicom import dcmread
from os.path import isfile, join, exists
import sys

ALL_XRAYS={
    'nih14':["/mnt/dfs/jpang12/datasets/nih_xray14/images/images","/data/jliang12/jpang12/dataset/nih_xray14/images/images", '.png'],
    'jsrt':["/mnt/dfs/jpang12/datasets/JSRT/All247images/images/","/data/jliang12/jpang12/dataset/JSRT/All247images/images/",".png"],
    'mendeleyv2':["/mnt/dfs/jpang12/datasets/Mendeley-V2/CellData/chest_xray/","/data/jliang12/jpang12/dataset/Mendeley-V2/CellData/chest_xray/",".jpeg"],
    'montgomery': ["/mnt/dfs/jpang12/datasets/MontgomeryCountyX-ray/MontgomerySet/CXR_png/", "/data/jliang12/jpang12/dataset/MontgomerySet/CXR_png/", ".png"],
    'shenzhen': ["/mnt/dfs/jpang12/datasets/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/", "/data/jliang12/mhossei2/Dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/", ".png"],
    'rsna': ["/mnt/dfs/jpang12/datasets/rsna-pneumonia-detection-challenge/", "/data/jliang12/jpang12/dataset/rsna-pneumonia-detection-challenge/", ".png"],
    'chexpert': ["/mnt/dfs/jpang12/datasets/CheXpert-v1.0/", "/data/jliang12/mhossei2/Dataset/CheXpert-v1.0/", ".jpg"],
    'padchest': ["/mnt/dfs/jpang12/datasets/PadChest/image_zips", "/data/jliang12/jpang12/dataset/PadChest/image_zips/", ".png"],
    'mimiccxr': ["/mnt/dfs/jpang12/datasets/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0", "/data/jliang12/jpang12/dataset/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0", ".jpg"],
    'indiana': ["/mnt/dfs/jpang12/datasets/Indiana_ChestX-ray/images/images_normalized/", "/data/jliang12/jpang12/dataset/Indiana_ChestX-ray/images/images_normalized/", ".jpeg"],
    'convidx': ["/mnt/dfs/jpang12/datasets/COVIDx/", "/data/jliang12/jpang12/dataset/COVIDx/", ".png"],
    'convidradiography': ["/mnt/dfs/jpang12/datasets/COVID-19_Radiography_Dataset/", "/data/jliang12/jpang12/dataset/COVID-19_Radiography_Dataset/", ".png"],
    'vindrcxr': ["/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/", "/data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/", ".jpeg"],
    'rocird_covid': ["/mnt/dfs/jpang12/datasets/rocird_covid/png", "/data/jliang12/jpang12/dataset/RICORD_covid/png", ".png"],
    'nih_tb_portals': ["/mnt/dfs/jpang12/datasets/nih_tb_portal", "/data/jliang12/jpang12/dataset/nih_tb_portals",  ".png"],
    'plco': ["/mnt/dfs/jpang12/datasets/PLCOI-880", "/data/jliang12/jpang12/dataset/PLCOI-880", ".tif"],
    'siimacr':["", "/data/jliang12/zzhou187/dataset/Pneumothorax_segmentation/", ".dcm"],
}



class MaskGenerator:
    def __init__(self, input_size=448, mask_patch_size=16, model_patch_size=4, mask_ratio=0.5):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def get_mask(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask



def build_md_transform(mode, dataset = "chexray"):
    transformList_mg = []
    transformList_simple = []

    if dataset == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])


    if mode=="train":
        transformList_mg.append(local_pixel_shuffling)
        transformList_mg.append(nonlinear_transformation)
        transformList_mg.append(transforms.RandomApply([paint], p=0.9))
        transformList_mg.append(torch.from_numpy)
        transformList_mg.append(normalize)
        transformSequence_mg = transforms.Compose(transformList_mg)

        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)

        return transformSequence_mg, transformSequence_simple
    else:
        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)
        return transformSequence_simple, transformSequence_simple




def build_simple_transform(dataset = "chexray"):
    transformList = []

    if dataset == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])

    transformList.append(torch.from_numpy)
    transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)


    return transformSequence




class Popar_allXray(Dataset):
    def __init__(self,index_path, include_test_data=False, augment = None, machine="lab", use_rgb = True, channel_first = True, writter = sys.stdout, views = "frontal", image_size=448,patch_size=32):
        self.index_path = index_path
        self.img_list = []
        self.augment = augment
        self.use_rgb =use_rgb
        self.channel_first = channel_first
        self.image_size = image_size
        self.patch_size = patch_size
        if views=="frontal":
            self.data_index = join(self.index_path, "data_index", "frontal")
        else:
            self.data_index = join(self.index_path, "data_index", "all")

        if machine =="lab":
            _indictor = 0
        else:
            _indictor = 1
        self.writter = writter
        for i, (key, value) in enumerate(ALL_XRAYS.items()):
            print("Initializing [{}/{}]: {} dataset".format(i+1,len(ALL_XRAYS),key),file = self.writter)
            with open(join(join(self.data_index,key), "train.txt"), 'r') as fr:
                line = fr.readline()
                while line:
                    self.img_list.append([join(value[_indictor], line.split(' ')[0].strip()), value[-1]])
                    line = fr.readline()
            if include_test_data and exists(join(key, "test.txt")):
                with open(join(join(self.data_index,key), "test.txt"), 'r') as fr:
                    line = fr.readline()
                    while line:
                        self.img_list.append([join(value[_indictor], line.split(' ')[0].strip()), value[-1]])
                        line = fr.readline()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path_info = self.img_list[index]
        try:
            if "dcm" in img_path_info[-1]:
                img_data = dcmread(img_path_info[0]).pixel_array
            else:
                img_data = cv2.imread(img_path_info[0], cv2.IMREAD_GRAYSCALE)


            if img_data is None:
                self.__getitem__(random.randrange(0, index))
            # img_data = cv2.resize(img_data,(self.image_size, self.image_size),interpolation=cv2.INTER_AREA)

            if np.min(img_data)<0 or np.max(img_data)>255:
                img_data = cv2.normalize(src=img_data, dst=img_data, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            img_data = img_data.astype(np.uint8)
            if self.use_rgb:
                img_data = np.repeat(img_data[:, :, np.newaxis], 3, axis=2)
                # if self.channel_first:
                #     img_data = rearrange(img_data, 'h w c ->c h w')
            # img_data = img_data / 255
            # print(img_data.shape)
            img_data = Image.fromarray(img_data)

            

            # gt_whole = self.augment[1](img_data)
            # if random.random() < 0.5:
            #     randperm = torch.arange(0, (self.image_size // self.patch_size) ** 2, dtype=torch.long)
            #     aug_whole = self.augment[0](img_data)
            # else:
            #     aug_whole = gt_whole
            #     randperm = torch.randperm((self.image_size // self.patch_size) ** 2, dtype=torch.long)
            randperm = torch.randperm((self.image_size//self.patch_size)**2, dtype=torch.long)
            

            if self.augment != None: imageData = self.augment(img_data)
            # # return randperm, gt_whole, aug_whole
            return imageData, randperm

        except Exception as e:
            print("error in: ", img_path_info[0])
            print(e,file = self.writter)
            self.__getitem__( random.randrange(0, index))




class Popar_chestxray(Dataset):
    def __init__(self, image_path_file, augment, image_size=448,patch_size=32):
        self.img_list = []
        self.augment = augment
        self.patch_size = patch_size
        self.image_size = image_size

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline().strip()
                    if line:
                        lineItems = line.split(" ")
                        imagePath = os.path.join(pathImageDirectory, lineItems[0])
                        self.img_list.append(imagePath)



    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),(self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        gt_whole = self.augment[1](imageData)
        if random.random()<0.5:
            randperm = torch.arange(0,(self.image_size//self.patch_size)**2, dtype=torch.long)
            aug_whole = self.augment[0](imageData)
        else:
            aug_whole = gt_whole
            randperm = torch.randperm((self.image_size//self.patch_size)**2, dtype=torch.long)

        return randperm, gt_whole, aug_whole

    def __len__(self):
        return len(self.img_list)


class Popar_chestxray_mim_coordsps(Dataset):
    def __init__(self, image_path_file, transform, image_size=448,patch_size=32, mask_ratio= 0.2):
        self.img_list = []
        self.transform = transform
        self.patch_size = patch_size
        self.image_size = image_size
        self.mask_generator = MaskGenerator(input_size =image_size , mask_patch_size= patch_size, mask_ratio=mask_ratio)
        self.dummy_mask_generator = MaskGenerator(input_size =image_size , mask_patch_size= patch_size, mask_ratio=0)

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline().strip()
                    if line:
                        lineItems = line.split(" ")
                        imagePath = os.path.join(pathImageDirectory, lineItems[0])
                        self.img_list.append(imagePath)

    def normalize(self, x, newRange=(-3, 3)):  # x is an array. Default range is between zero and one
        xmin, xmax = torch.min(x), torch.max(x)  # get max and min from input array
        norm = (x - xmin) / (xmax - xmin)  # scale between zero and one

        if newRange == (0, 1):
            return (norm)  # wanted range is the same as norm
        elif newRange != (0, 1):
            return norm * (newRange[1] - newRange[0]) + newRange[0]  # scale to a different range.
        # add other conditions here. For example, an error message

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),(self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        perm_gt = []
        for y in range(0, self.image_size // self.patch_size):
            for x in range(0, self.image_size // self.patch_size):
                perm_gt.append((y,x))

        if random.random()<0.5:
            random.shuffle(perm_gt)
            aug_image = np.zeros(imageData.shape)
            counter = 0
            for y in range(0, self.image_size // self.patch_size):
                for x in range(0, self.image_size // self.patch_size):
                    coords = perm_gt[counter]
                    aug_image[:,y * self.patch_size:y * self.patch_size + self.patch_size,
                    x * self.patch_size:x * self.patch_size + self.patch_size] = imageData[:, coords[0] * self.patch_size:coords[0] * self.patch_size + self.patch_size,
                                                                                 coords[1] * self.patch_size:coords[1] * self.patch_size + self.patch_size]
                    counter += 1
            perm_gt = torch.tensor(perm_gt)

            return self.normalize(perm_gt), self.dummy_mask_generator.get_mask(), self.transform(aug_image), self.transform(imageData)
        else:
            aug_image = np.zeros(imageData.shape)
            counter = 0
            for y in range(0, self.image_size // self.patch_size):
                for x in range(0, self.image_size // self.patch_size):
                    coords = perm_gt[counter]
                    aug_image[:, y * self.patch_size:y * self.patch_size + self.patch_size,
                    x * self.patch_size:x * self.patch_size + self.patch_size] = imageData[:,
                                                                                 coords[0] * self.patch_size:coords[0] * self.patch_size + self.patch_size,
                                                                                 coords[1] * self.patch_size:coords[1] * self.patch_size + self.patch_size]
                    counter += 1
            perm_gt = torch.tensor(perm_gt)

            return self.normalize(perm_gt), self.mask_generator.get_mask(), self.transform(aug_image), self.transform(imageData)



    def __len__(self):
        return len(self.img_list)




class Popar_allXray_mim_coordsps(Dataset):
    def __init__(self,include_test_data=False, transform = None, machine="sol", use_rgb = True, channel_first = True, writter = sys.stdout, views = "frontal", image_size=448,patch_size=32,mask_ratio= 0.2):
        self.img_list = []
        self.transform = transform
        self.use_rgb =use_rgb
        self.channel_first = channel_first
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_generator = MaskGenerator(input_size =image_size , mask_patch_size= patch_size, mask_ratio=mask_ratio)
        self.dummy_mask_generator = MaskGenerator(input_size =image_size , mask_patch_size= patch_size, mask_ratio=0)



        if views=="frontal":
            self.data_index = join("data_index", "frontal")
        else:
            self.data_index = join("data_index", "all")

        if machine =="lab":
            _indictor = 0
        else:
            _indictor = 1
        self.writter = writter
        for i, (key, value) in enumerate(ALL_XRAYS.items()):
            print("Initializing [{}/{}]: {} dataset".format(i+1,len(ALL_XRAYS),key),file = self.writter)
            with open(join(join(self.data_index,key), "train.txt"), 'r') as fr:
                line = fr.readline()
                while line:
                    self.img_list.append([join(value[_indictor], line.split(' ')[0].strip()), value[-1]])
                    line = fr.readline()
            if include_test_data and exists(join(key, "test.txt")):
                with open(join(join(self.data_index,key), "test.txt"), 'r') as fr:
                    line = fr.readline()
                    while line:
                        self.img_list.append([join(value[_indictor], line.split(' ')[0].strip()), value[-1]])
                        line = fr.readline()

    def __len__(self):
        return len(self.img_list)
    def normalize(self, x, newRange=(-3, 3)):  # x is an array. Default range is between zero and one
        xmin, xmax = torch.min(x), torch.max(x)  # get max and min from input array
        norm = (x - xmin) / (xmax - xmin)  # scale between zero and one

        if newRange == (0, 1):
            return (norm)  # wanted range is the same as norm
        elif newRange != (0, 1):
            return norm * (newRange[1] - newRange[0]) + newRange[0]  # scale to a different range.
        # add other conditions here. For example, an error message
    def __getitem__(self, index):
        img_path_info = self.img_list[index]
        try:
            if "dcm" in img_path_info[-1]:
                img_data = dcmread(img_path_info[0]).pixel_array
            else:
                img_data = cv2.imread(img_path_info[0], cv2.IMREAD_GRAYSCALE)


            if img_data is None:
                self.__getitem__(random.randrange(0, index))
            img_data = cv2.resize(img_data,(self.image_size, self.image_size),interpolation=cv2.INTER_AREA)

            if np.min(img_data)<0 or np.max(img_data)>255:
                img_data = cv2.normalize(src=img_data, dst=img_data, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            img_data = img_data.astype(np.uint8)
            if self.use_rgb:
                img_data = np.repeat(img_data[:, :, np.newaxis], 3, axis=2)
                if self.channel_first:
                    img_data = rearrange(img_data, 'h w c ->c h w')

            img_data = img_data / 255

            perm_gt = []
            for y in range(0, self.image_size // self.patch_size):
                for x in range(0, self.image_size // self.patch_size):
                    perm_gt.append((y, x))

            if random.random() < 0.5:
                random.shuffle(perm_gt)
                aug_image = np.zeros(img_data.shape)
                counter = 0
                for y in range(0, self.image_size // self.patch_size):
                    for x in range(0, self.image_size // self.patch_size):
                        coords = perm_gt[counter]
                        aug_image[:, y * self.patch_size:y * self.patch_size + self.patch_size,
                        x * self.patch_size:x * self.patch_size + self.patch_size] = img_data[:,coords[0] * self.patch_size:coords[0] * self.patch_size + self.patch_size,
                                                                                     coords[1] * self.patch_size:coords[ 1] * self.patch_size + self.patch_size]
                        counter += 1
                perm_gt = torch.tensor(perm_gt)

                return self.normalize(perm_gt), self.dummy_mask_generator.get_mask(), self.transform(aug_image), self.transform(img_data)
            else:
                aug_image = np.zeros(img_data.shape)
                counter = 0
                for y in range(0, self.image_size // self.patch_size):
                    for x in range(0, self.image_size // self.patch_size):
                        coords = perm_gt[counter]
                        aug_image[:, y * self.patch_size:y * self.patch_size + self.patch_size,
                        x * self.patch_size:x * self.patch_size + self.patch_size] = img_data[:,coords[0] * self.patch_size:coords[ 0] * self.patch_size + self.patch_size,
                                                                                     coords[1] * self.patch_size:coords[1] * self.patch_size + self.patch_size]
                        counter += 1
                perm_gt = torch.tensor(perm_gt)

                return self.normalize(perm_gt), self.mask_generator.get_mask(), self.transform(aug_image), self.transform(img_data)

        except Exception as e:
            print("error in: ", img_path_info[0])
            print(e,file = self.writter)
            self.__getitem__( random.randrange(0, index))













class Popar_chestxray_graycode(Dataset):
    def __init__(self, image_path_file, augment, image_size=448,patch_size=32, gray_code_path = "data_index/graycode.txt"):
        self.img_list = []
        self.augment = augment
        self.patch_size = patch_size
        self.image_size = image_size
        self.graycodes = []

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline()
                    if line:
                        lineItems = line.split()
                        imagePath = os.path.join(pathImageDirectory, lineItems[0])
                        self.img_list.append(imagePath)


        with open(gray_code_path,"r") as fr:
            line = fr.readline()
            while line:
                code = line.strip()
                self.graycodes.append([int(c) for c in code])
                line = fr.readline()

        self.graycodes = torch.Tensor(self.graycodes)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),(self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        gt_whole = self.augment[1](imageData)
        if random.random()<0.5:
            randperm = torch.arange(0,(self.image_size//self.patch_size)**2, dtype=torch.long)
            aug_whole = self.augment[0](imageData)
        else:
            aug_whole = gt_whole
            randperm = torch.randperm((self.image_size//self.patch_size)**2, dtype=torch.long)

        return randperm, gt_whole, aug_whole,  self.graycodes[randperm,:]

    def __len__(self):
        return len(self.img_list)


class Popar_imagenet(Dataset):
    def __init__(self, image_path_file, augment, image_size=448,patch_size=32):
        self.img_list = []
        self.augment = augment
        self.patch_size = patch_size
        self.image_size = image_size

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline()
                    if line:
                        lineItems = line.split()
                        imagePath = os.path.join(pathImageDirectory, lineItems[2])
                        self.img_list.append(imagePath)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),(self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        gt_whole = self.augment[1](imageData)
        if random.random()<0.5:
            randperm = torch.arange(0,(self.image_size//self.patch_size)**2, dtype=torch.long)
            aug_whole = self.augment[0](imageData)
        else:
            aug_whole = gt_whole
            randperm = torch.randperm((self.image_size//self.patch_size)**2, dtype=torch.long)

        return randperm, gt_whole, aug_whole

    def __len__(self):
        return len(self.img_list)





ALL_FUNDUS={
    'ADAM':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/ADAM"],
    'Augmented_ocular_diseases':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/Augmented_ocular_diseases"],
    'BiDR':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/BiDR"],
    'Diabetic_Retinopathy_Arranged':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/Diabetic_Retinopathy_Arranged"],
    'EyePACS':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/EyePACS"],
    'FIRE':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/FIRE"],
    'GAMMA':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/GAMMA"],
    'JSIEC':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/JSIEC"],
    'Messidor-2':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/Messidor-2"],
    'MuReD':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/MuReD"],
    'ODIR-5K':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/ODIR-5K"],
    'PALM':["/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/PALM"],
}



class Popar_allFundus(Dataset):
    def __init__(self,index_path, include_test_data=False, augment = None, use_rgb = True, channel_first = True, image_size=448,patch_size=32,epoch=0):
        self.index_path = index_path
        self.img_list = []
        self.augment = augment
        self.use_rgb =use_rgb
        self.channel_first = channel_first
        self.image_size = image_size
        self.patch_size = patch_size
        self.epoch = epoch




        for i, (key, value) in enumerate(ALL_FUNDUS.items()):
            print("Initializing [{}/{}]: {} dataset".format(i+1,len(ALL_FUNDUS),key))
            try:
                with open(join(join(self.index_path,key), "train_val.txt"), 'r') as fr:
                    line = fr.readline()
                    while line:
                        self.img_list.append(join(value[0], line.strip()))
                        line = fr.readline()
            except:
                with open(join(join(self.index_path,key), "train.txt"), 'r') as fr:
                    line = fr.readline()
                    while line:
                        self.img_list.append(join(value[0], line.strip()))
                        line = fr.readline()

            if include_test_data and exists(join(self.index_path, key, "test.txt")):
                with open(join(join(self.index_path,key), "test.txt"), 'r') as fr:
                    line = fr.readline()
                    while line:
                        self.img_list.append(join(value[0], line.strip()))
                        line = fr.readline()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path_info = self.img_list[index]

        # if img_path_info.lower().endswith('.tif'):
        #     img_data = cv2.imread(img_path_info, cv2.IMREAD_UNCHANGED)
        # else:
        img_data = cv2.imread(img_path_info)


        # if img_data is None:
        #     self.__getitem__(random.randrange(0, index))
        # img_data = cv2.resize(img_data,(self.image_size, self.image_size),interpolation=cv2.INTER_AREA)

        if np.min(img_data)<0 or np.max(img_data)>255:
            img_data = cv2.normalize(src=img_data, dst=img_data, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        img_data = img_data.astype(np.uint8)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data, mask = crop_image_from_gray(img_data)
        img_data = resize_padding(img_data)
        # print(imageData.shape)

        img_data = Image.fromarray(img_data)

        

        # gt_whole = self.augment[1](img_data)
        # if random.random() < 0.5:
        #     randperm = torch.arange(0, (self.image_size // self.patch_size) ** 2, dtype=torch.long)
        #     aug_whole = self.augment[0](img_data)
        # else:
        #     aug_whole = gt_whole
        #     randperm = torch.randperm((self.image_size // self.patch_size) ** 2, dtype=torch.long)
        randperm = torch.randperm((self.image_size//self.patch_size)**2, dtype=torch.long)
        

        if self.augment != None: imageData = self.augment(img_data, self.epoch)
        # # return randperm, gt_whole, aug_whole
        return imageData, randperm
    
    
def resize_padding(image,new_size=(1024,1024)):
    h,w = image.shape[:2]  # current shape [height, width]
    r = min(new_size[0] / h, new_size[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    # 计算需要填充的边的像素
    dw, dh = new_size[1] - new_unpad[0], new_size[0] - new_unpad[1]  
    dw /= 2  # 除以2即最终每边填充的像素
    dh /= 2
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    # round(dw,dh - 0.1)直接让小于1的为0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # padding
    # image = np.pad(image,((3,2),(2,3)),'constant',constant_values = (0,0))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) 
    return image


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img, mask