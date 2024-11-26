# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CUDA_VISIBLE_DEVICES="1,2,3,4" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28302 main.py --arch swin_base --batch_size_per_gpu 8
# CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port 28308 main.py --arch swin_base --batch_size_per_gpu 8


import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from infonce import *
import utils
import vision_transformer as vits
import models.swin_transformer as swins
from vision_transformer import DINOHead,SimMIM_head,SimMIM_head_SWIN, DenseHead
from ImageFolder_vindr import ImageFolder_vindr,ChestX_ray14,ShenzhenCXR,LDPolyp
from config import config
from config import update_config
from config import save_config
from models import build_model
from transforms import DataAugmentation
from losses import globalconsis_loss
from einops import rearrange
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import recall_score
from torch import autograd
import ipdb
from models.swin_transformer import MaskedAutoencoderViT
from utils import Composer, Decomposer
import random
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from torch.utils.data import Dataset
import seaborn as sns
from timm.models.swin_transformer import SwinTransformer

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='swin_base', type=str,
        choices=['cvt_tiny', 'cvt_small', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'vil_14121', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--mask_mode', type=str, default='interpolation', help="""Using MIM in student branch to restore the images""")
    parser.add_argument('--MIM', action='store_true', help="""Using MIM in student branch to restore the images""")
    parser.add_argument('--mae_manner',action='store_true', help="""train in MAE manner: only input non-mask patches to encoder""")
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0.8, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=80, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=302, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    parser.add_argument('--use_dense_prediction', default=False, type=utils.bool_flag,
        help="Whether to use dense prediction in projection head (Default: False)")
    # Misc
    parser.add_argument('--data_path', default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/extrap_shuffle_compdecomp_consis", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=25, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Used for ddp, please ignore and do not set this argument.")
    parser.add_argument('--cfg',default='./swin_configs/swin_base_img224_window7.yaml', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser


class ChestX_ray14_KDE(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, num_class=14, patch_size=448):
        self.img_list = []
        self.img_label = []
        self.patch_size = patch_size
        self.augment = transforms.Compose([transforms.Resize((448,448)),
                                            # Rearrange_and_Norm(),
                                            # torch.from_numpy,
                                           transforms.ToTensor(),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])])

        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))



    def random_crop_and_mask(self,image, scale_range=(0.2, 0.49)):
        """
        Randomly crop a part of the image and create a mask of the original image with the
        cropped part hidden.

        Parameters:
            image (PIL.Image): The original image.
            scale_range (tuple): A 2-tuple defining the minimum and maximum scale of the cropped area.

        Returns:
            PIL.Image: Cropped image.
            PIL.Image: Masked image with the cropped area hidden.
        """
        # randomly choose the crop number
        # k, l = choices([(1,2), (2,1), (2,2)])[0]
        k, l = 2,2

        # Get original image size
        orig_width, orig_height = image.size

        # Determine size of the crop
        scale = random.uniform(scale_range[0], scale_range[1])
        crop_width = int(orig_width * scale * k)
        crop_height = int(orig_height * scale * l)

        # Determine position of the crop
        # print(orig_width, crop_width, scale, k)
        # print(orig_width - crop_width)
        left = random.randint(0, orig_width - crop_width)
        upper = random.randint(0, orig_height - crop_height)
        right = left + crop_width
        lower = upper + crop_height

        # Crop the image
        whole_crop = image.crop((left, upper, right, lower))
        # whole_crop = np.asarray(whole_crop)

        # sub-crops
        sub_crops = []
        for i in range(k):
            for j in range(l):
                sub_crops.append(image.crop((left+crop_width/k*i, upper+crop_height/l*j, left+crop_width/k*(i+1), upper+crop_height/l*(j+1))))

        return whole_crop, sub_crops
    

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        initial_crop_transform = transforms.RandomResizedCrop(
            1024,  # Final size of the crop
            scale=(0.4, 1),  # Scale range
        )
        origin_image = initial_crop_transform(imageData)

        # Get random crop and masked image
        whole_crop, sub_crop = self.random_crop_and_mask(origin_image)
        # origin_image.save(os.path.join('./save_image/', f"{index}_origin_image.jpg"))
        # masked_image.save(os.path.join('./save_image/', f"{index}_masked_image.jpg"))
        # cropped_image.save(os.path.join('./save_image/', f"{index}_cropped_image.jpg"))

        whole_crop = self.augment(whole_crop)
        sub_crops = []
        for i in range(len(sub_crop)):
            sub_crops.append(self.augment(sub_crop[i])) 


        # Optionally, convert the images to PyTorch tensors here
        return whole_crop, sub_crops

    def __len__(self):
        return len(self.img_list)


def save_kde_plot(similarities, file_path):
    with open('./simi_result_12N_contrast_16.txt', 'w') as file:
        file.write('\n'.join([str(sim) for sim in similarities]))
    # t_stat, p_val = ttest_ind(similarities, similarities_2)
    # print(np.array(similarities).mean(),np.array(similarities_2).mean())
    # print(f"p_val: {p_val:.30f}")
    sns.kdeplot(similarities, shade=True,bw_adjust=3)
    plt.title('KDE of Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.savefig(file_path)
    plt.close()

from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)



def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if os.path.exists(os.path.join(args.output_dir, "log.txt")):
        log_writer = open(os.path.join(args.output_dir, "log.txt"), 'a')
    else:
        log_writer = open(os.path.join(args.output_dir, "log.txt"), 'w')


    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")

    if 'swin' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        student.head = DINOHead(
            student.num_features,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                student.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)
    
    student = build_model(config, use_dense_prediction=args.use_dense_prediction)
    teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
    embed_dim = student.num_features
    student = utils.MultiCropWrapper(student, MaskedAutoencoderViT(),
        DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ),DenseHead(),args)
    teacher = utils.MultiCropWrapper_teacher(
        teacher, MaskedAutoencoderViT(),
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        DenseHead(),
    )

    composer = Composer()
    decomposer = Decomposer()


    # move networks to gpu
    # student, teacher = student.cuda(), teacher.cuda()
    student = SwinTransformer(img_size=448,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                         num_heads=(4, 8, 16, 32), num_classes=3)


    checkpoint = torch.load('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/sslgenesis_peac/checkpoint0150.pth', map_location="cpu")
    checkpoint_dino = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dinocheckpoint0300_swin.pth', map_location="cpu")

    try:
        checkpoint = checkpoint['teacher']
    except:
        # checkpoint = checkpoint['model']
        checkpoint = checkpoint['state_dict']
    #checkpoint = checkpoint['student']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}


    try:
        checkpoint_dino = checkpoint_dino['teacher']
    except:
        # checkpoint = checkpoint['model']
        checkpoint_dino = checkpoint_dino['state_dict']
    #checkpoint = checkpoint['student']
    checkpoint_dino = {k.replace("module.", ""): v for k, v in checkpoint_dino.items()}
    checkpoint_dino = {k.replace("vit_model.", ""): v for k, v in checkpoint_dino.items()}
    checkpoint_dino = {k.replace("backbone.", ""): v for k, v in checkpoint_dino.items()}
    checkpoint_dino = {k.replace("swin_model.", ""): v for k, v in checkpoint_dino.items()}



    # msg = student.load_state_dict(checkpoint_model, strict=False)
    msg = student.load_state_dict(checkpoint_dino, strict=False)
    print(msg)
    
    comp_dict = {k.replace("composer.", ""): v for k, v in checkpoint_model.items() if "composer" in k}
    msg = composer.load_state_dict(comp_dict, strict=False)
    print(msg)
    decomp_dict = {k.replace("decomposer.", ""): v for k, v in checkpoint_model.items() if "decomposer" in k}
    msg = decomposer.load_state_dict(decomp_dict, strict=False)
    print(msg)
    student = student.cuda()
    composer = composer.cuda()
    decomposer = decomposer.cuda()

    dataset = ChestX_ray14_KDE(args.data_path,'./data/xray14/official/val_official.txt')
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")


    similarties_list = []
    accuracies = []
    
    #torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        for it, (whole_crop, sub_crops) in enumerate(data_loader): # sub_crops list, len()=4
            print(it)
            # print(mask.shape)
            # update weight decay and learning rate according to their schedule
            # if it==1000:
            #     return np.array(similarties_list_1000)
            whole_crop = whole_crop.cuda(non_blocking=True).float()
            for i in range(len(sub_crops)):
                sub_crops[i] = sub_crops[i].cuda(non_blocking=True).float()

            # swin
            # whole_crop_feature = model.forward_features(whole_crop).mean(dim=1) # return the global embedding of the input image
            # sub_crops_feature = model.forward_features(sub_crops[0]).mean(dim=1)
            # for i in range(1,len(sub_crops)):
            #     sub_crops_feature+=model.forward_features(sub_crops[i]).mean(dim=1)


            # vit
            whole_crop_feature = student.forward_features(whole_crop).mean(dim=1) # return the global embedding of the input image
            # sub_crops_feature = []
            # for i in range(len(sub_crops)):
            #     sub_feature, _middle_features = student(sub_crops[i])
            #     sub_crops_feature.append(sub_feature)


            # whole_crop_feature = F.softmax(whole_crop_feature, dim=-1)
            # sub_crops_feature = F.softmax(sub_crops_feature, dim=-1)
            sub_decomp = decomposer(whole_crop_feature)
            whole_comp = composer(sub_decomp)
            
            # Compute similarity between the whole crop and sub-crop
            for i in range(whole_crop_feature.shape[0]):
                # ipdb.set_trace()

                similarity = compute_similarity(whole_crop_feature[i].unsqueeze(0).cpu(), whole_comp[i].unsqueeze(0).cpu())
                # ipdb.set_trace()
                print(similarity)

                similarties_list.append(similarity)
            #print(accuracy)\
        
        print(np.mean(similarties_list))
        similarties_list = np.array(similarties_list)
        np.save('./test_properties/dino.npy', similarties_list)
        # save_kde_plot(similarties_list, './test_properties/autoencoder_comdecomp.png')

    







if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # train_dino(args)

    sim_sslgenesis = np.load('./test_properties/sslgenesis.npy')
    sim_sslgenesis = sim_sslgenesis.squeeze(1).squeeze(1)

    sim_dino = np.load('./test_properties/dino.npy')
    sim_dino = sim_dino.squeeze(1).squeeze(1)

    plt.figure(figsize=(7, 6))
    sns.kdeplot(sim_sslgenesis, fill=True,bw_adjust=1,color='hotpink', label='Multi-perspectives')
    sns.kdeplot(sim_dino, fill=True,bw_adjust=1,color='peachpuff', label='DINO')
    plt.legend(loc='best', borderaxespad=0., prop={'size': 15})
    plt.xlabel('Feature Similarity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.savefig('./test_properties/autoencoder_compdecomp.png')
