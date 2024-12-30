# cutmix or mixup 
import os
import csv
import numpy as np
from PIL import Image
import glob
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import itertools
import torch
import math

def cutmix_data(img1=None, img2=None, label1=0, label2=1, mask=None, transform=None):
    """
    :cutmix 
    :param real_image: 真实图像 (Tensor)
    :param fake_image: 假图像 (Tensor)
    :param mask: 混合 mask
    return: 返回混合图像 (Tensor)
    """
    if isinstance(img1, str):
        img1, is_success = read_image(img1)
        img1, label1 = apply_transform(img1, label1, transform, is_dire=False)
    
    if isinstance(img2, str):
        # print('img2:', img2)
        img2, is_success = read_image(img2)
        img2, label2 = apply_transform(img2, label2, transform, is_dire=False)
    
    cutmix_img = mask * img1 + (1 - mask) * img2
    cutmix_label = 0 if label1 == 0 and label2 == 0 else 1

    return cutmix_img, cutmix_label

def mixup_data(img1, img2, label1, label2, alpha=1):
    """Compute the mixup data for binary classification (e.g., real/fake images). Return mixed inputs, mixed target, and lambda"""
    
    # 从 Beta 分布中采样一个 lam 值
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # mixup img
    mixed_img = lam * img1 + (1 - lam) * img2

    # labels(0和0还是0，1和0为1)
    print(label1)
    print(label2)
    mixed_label = 0 if label1.item() == 0 and label2.item() == 0 else 1

    return mixed_img, mixed_label

def generate_mask(img, label, mixing_label, lam):
    """
    :param img: 输入图像，形状为 (C, H, W)
    :param lam: 混合比例 lambda，表示前景的比例（为0）
    :return: mask, 后景 * mask + 前景 * (1 - mask)
    """
    H, W = img.shape[1], img.shape[2]  # 获取图像的高度和宽度

    # 定义 patch 的大小
    patch_size = 14

    # 计算 patch 的数量
    patch_H_number = H // patch_size
    patch_W_number = W // patch_size

    # 初始化一个全1的 mask，与图像大小相同
    mixing_mask = torch.ones((H, W), dtype=torch.float32)
    label_mask = torch.full((H, W), label, dtype=torch.float32)

    # 计算要置为0的patch数量，基于 lambda
    num_patches = patch_H_number * patch_W_number
    num_zero_patches = int(num_patches * (1 - lam))

    # 随机选择若干个 patch 的索引，将其置为 0
    zero_indices = random.sample(range(num_patches), num_zero_patches)
    for idx in zero_indices:
        row = idx // patch_W_number
        col = idx % patch_W_number
        start_y = row * patch_size
        start_x = col * patch_size

        # 将对应的 14x14 区域置为 0
        mixing_mask[start_y:start_y + patch_size, start_x:start_x + patch_size] = 0
        label_mask[start_y:start_y + patch_size, start_x:start_x + patch_size] = mixing_label

    return mixing_mask, label_mask