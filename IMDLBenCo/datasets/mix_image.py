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
import torchvision
import math
from pathlib import Path

# def read_image(image_path, resize_size=None):
#     # try:
#     #     image = cv2.imread(image_path)
#     #     if resize_size is not None:
#     #         image = resize_long_size(image, long_size=resize_size)
#     #     # Revert from BGR
#     #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #     return image, True
#     # except:
#     #     print(f'{image_path} read error!!!')
#     #     return np.zeros(shape=(512, 512, 3), dtype=np.uint8), False
    
#     try:
#         # 使用Pillow加载图像，Pillow会自动处理ICC配置文件
#         image = Image.open(image_path)
        
#         # 如果需要调整大小
#         if resize_size is not None:
#             image = image.resize((resize_size, resize_size))
        
#         # 转换为 NumPy 数组并确保 RGB 格式
#         image = np.array(image.convert("RGB"))
        
#         return image, True
#     except Exception as e:
#         print(f'{image_path} read error!!! {e}')
#         return np.zeros(shape=(512, 512, 3), dtype=np.uint8), False

# def generate_patch_mask(img, lam):
#     """
#     :param img: 输入图像，形状为 (C, H, W)
#     :param lam: 混合比例 lambda，表示前景的比例（为0）
#     :return: mask, 后景 * mask + 前景 * (1 - mask)
#     """
#
#     H, W = img.shape[1], img.shape[2]  # 获取图像的高度和宽度
#
#     # 定义 patch 的大小
#     # patch_size = 14
#     patch_size = 56
#
#     # 计算 patch 的数量
#     patch_H_number = H // patch_size
#     patch_W_number = W // patch_size
#
#     # 初始化一个全1的 mask，与图像大小相同
#     mask = torch.ones((H, W), dtype=torch.float32)
#
#     # 计算要置为0的patch数量，基于 lambda
#     num_patches = patch_H_number * patch_W_number
#     num_zero_patches = int(num_patches * (1 - lam))
#
#     # 随机选择若干个 patch 的索引，将其置为 0
#     zero_indices = random.sample(range(num_patches), num_zero_patches)
#     for idx in zero_indices:
#         row = idx // patch_W_number
#         col = idx % patch_W_number
#         start_y = row * patch_size
#         start_x = col * patch_size
#
#         # 将对应的 56x56 区域置为 0
#         mask[start_y:start_y + patch_size, start_x:start_x + patch_size] = 0
#
#     return mask

def generate_patch_mask(H, W, lam):
    """
    :param img H W: 输入图像形状 H, W
    :param lam: 混合比例 lambda，表示前景的比例（为0）
    :return: mask, 后景 * mask + 前景 * (1 - mask)
    """

    # patch_size = 56
    # 计算 patch 的数量
    # patch_H_number = H // patch_size
    # patch_W_number = W // patch_size

    patch_H_number = 4
    patch_W_number = 4
    
    patch_size_H = H / patch_H_number
    patch_size_W = W / patch_W_number

    # 初始化一个全1的 mask，与图像大小相同
    mask = np.full((H, W, 1), 1, dtype=np.float32)

    # 计算要置为0的patch数量，基于 lambda
    num_patches = patch_H_number * patch_W_number
    num_zero_patches = int(num_patches * (1 - lam))

    # 随机选择若干个 patch 的索引，将其置为 0
    zero_indices = random.sample(range(num_patches), num_zero_patches)
    for idx in zero_indices:
        row = idx // patch_W_number
        col = idx % patch_W_number
        start_y = int(row * patch_size_H)
        start_x = int(col * patch_size_W)

        # 将对应的区域置为 0
        mask[start_y:start_y + int(patch_size_H), start_x:start_x + int(patch_size_W)] = 0

    return mask

# def apply_transform(image1, image2, label1, label2, transform, is_dire=False):   
#     # 只在 transform 存在并且 is_dire 为 False 时应用 transform
#     if transform is not None and not is_dire:
#         if image2 is None:
#             try:
#                 if isinstance(transform, torchvision.transforms.transforms.Compose):
#                     image1 = transform(Image.fromarray(image1))
#                 else:
#                     data = transform(image=image1)
#                     image1 = data["image"]
#             except Exception as e:
#                 print(f"Transform error: {e}")
#                 print('-------------------------')
#                 # 在转换失败时，返回默认的零填充图像
#                 image1 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
#                 if isinstance(transform, torchvision.transforms.transforms.Compose):
#                     image1 = transform(Image.fromarray(image1))
#                 else:
#                     data = transform(image=image1)
#                     image1 = data["image"]
#                 label1 = 0
#             return image1, label1
            
#         else:
#             try:
#                 if isinstance(transform, torchvision.transforms.transforms.Compose):
#                     image1 = transform(Image.fromarray(image1))
#                     image2 = transform(Image.fromarray(image2))
#                 else:
#                     data = transform(image=image1, rec_image=image2)
#                     image1 = data["image"]
#                     image2 = data["rec_image"]
#             except Exception as e:
#                 print(f"Transform error: {e}")
#                 print('-------------------------')
#                 # 在转换失败时，返回默认的零填充图像
#                 image1 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
#                 if isinstance(transform, torchvision.transforms.transforms.Compose):
#                     image1 = transform(Image.fromarray(image1))
#                 else:
#                     data = transform(image=image1)
#                     image1 = data["image"]
#                 label1 = 0
#                 image2 = None
#                 label2 = 0     

#             return image1, image2, label1, label2

        
        
# def read_transforms(size=300, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
#                             is_crop=False,):
#     resize_fuc = A.RandomCrop(height=size, width=size) if is_crop else A.LongestMaxSize(max_size=size)
#     aug_hard = [
#         resize_fuc,
#         A.Normalize(mean=mean, std=std),
#         ToTensorV2()
#     ]
#     return A.Compose(aug_hard, additional_targets={'rec_image': 'image'})

def save_image_from_tensor(tensor, filename):
    """
    Convert a tensor to a PIL image and save it.
    :param tensor: The tensor to save
    :param filename: The filename to save the image
    """
    # Scale the tensor from [0, 1] to [0, 255] if it's float
    if tensor.max() <= 1.0:  # Check if the tensor is in [0, 1] range (common for ToTensor)
        tensor = tensor * 255.0

    # Convert the tensor to uint8 type
    tensor = tensor.byte()

    # Convert the tensor to a NumPy array (H, W, C)
    np_image = tensor.permute(1, 2, 0).numpy()

    # Convert the NumPy array to a PIL image and save
    pil_image = Image.fromarray(np_image)
    pil_image.save(filename)
    
def read_and_crop(img1_path, img2_path=None, crop_size=(224, 224)):
    """
    读取图像并执行裁剪：
    1. 如果有两个图像输入，则确保两个图像使用相同的位置裁剪。
    2. 如果只有一个图像输入，则对这张图像执行随机裁剪。
    :param img1_path: 第一个图像路径
    :param img2_path: 第二个图像路径 (默认为 None)
    :param crop_size: 裁剪大小 (默认 224x224)
    :return: 裁剪后的 img1 和 img2 (NumPy 数组)
    """
    # 读取第一个图像
    img1 = np.array(Image.open(img1_path).convert('RGB'))  # 使用PIL读取并转为RGB模式
    
    # 如果只有一个图像输入，执行随机裁剪
    if img2_path is None:
        return random_crop_single(img1, (224, 224))
    
    # 读取第二个图像
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    
    # 对两个图像执行相同的裁剪
    return random_crop_dual(img1, img2, crop_size)

def random_crop_single(img, crop_size=(224, 224)):
    """
    对单张图像执行随机裁剪
    :param img: 输入图像 (NumPy 数组)
    :param crop_size: 裁剪大小 (默认 224x224)
    :return: 裁剪后的 img (NumPy 数组)
    """
    # 获取图像的高度和宽度
    h, w, _ = img.shape
    
    # 确保裁剪区域不会超过图像的边界
    crop_h, crop_w = crop_size
    if h < crop_h or w < crop_w:
        raise ValueError(f"图像的尺寸 ({h}, {w}) 小于裁剪尺寸 ({crop_h}, {crop_w})")
    
    # 随机生成裁剪的起始位置 (x, y)
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)

    # 执行裁剪
    img_cropped = img[y:y + crop_h, x:x + crop_w]

    return img_cropped

def random_crop_dual(img1, img2, crop_size=(224, 224)):
    """
    对两个图像执行相同位置的随机裁剪
    :param img1: 第一个图像 (NumPy 数组)
    :param img2: 第二个图像 (NumPy 数组)
    :param crop_size: 裁剪大小 (默认 224x224)
    :return: 裁剪后的 img1 和 img2 (NumPy 数组)
    """
    # 获取图像的高度和宽度
    h, w, _ = img1.shape
    
    # 确保裁剪区域不会超过图像的边界
    crop_h, crop_w = crop_size
    if h < crop_h or w < crop_w:
        raise ValueError(f"图像的尺寸 ({h}, {w}) 小于裁剪尺寸 ({crop_h}, {crop_w})")
    
    # 随机生成裁剪的起始位置 (x, y)
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)

    # 对两个图像执行相同的裁剪
    img1_cropped = img1[y:y + crop_h, x:x + crop_w]
    img2_cropped = img2[y:y + crop_h, x:x + crop_w]

    return img1_cropped, img2_cropped



def cutmix_data(img1_path=None, img2_path=None, label1=0, label2=1, lam=0):
    """
    :cutmix 
    :img1_path, img2_path:输入两个图像的路径
    :lam:混合比例
    return: 返回混合图像 (Tensor)
    """

    # 读取第一个图像
    img1 = np.array(Image.open(img1_path).convert('RGB'))  # 使用PIL读取并转为RGB模式
    ori_image = img1
    # 读取第二个图像
    img2 = np.array(Image.open(img2_path).convert('RGB')) # H W C

    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]

    img1_label = np.full((img1.shape[0], img1.shape[1], 3), label1, dtype=np.float32)
        
    mask = generate_patch_mask(H1, W1, lam).squeeze(axis=-1) # (H, W, 1)去掉最后一维度
    mask_expanded = np.repeat(mask[..., None], 3, axis=-1)
    
    if 'inpainting' in img2_path: # real和rec一起变换
        cutmix_img = mask_expanded * img1 + (1 - mask_expanded) * img2
        img2_label = np.full((img1.shape[0], img1.shape[1], 3), label2, dtype=np.float32)
        mask_label = mask_expanded * img1_label + (1 - mask_expanded) * img2_label
    else:
        # 根据lam计算图2需要裁剪的大小
        area = H1 * W1 * lam
        if area > H2 * W2: # 若图2的面积不够裁剪，把整个图2贴上去
            new_img2 = img2
            new_mask = np.ones((H2, W2), dtype=np.float32)
            cut_h, cut_w = min(H1, H2), min(W1, W2)  # 确保不超过目标图像尺寸
        else:
            # 计算裁剪的高度和宽度
            new_lam = area / (H2 * W2)
            cut_h = int(H2 * np.sqrt(new_lam))
            cut_w = int(W2 * np.sqrt(new_lam))
            
            # 限制裁剪尺寸不超过目标图像尺寸
            cut_h = min(cut_h, H1)
            cut_w = min(cut_w, W1)

            # 随机选择裁剪起点
            y1 = np.random.randint(0, H2 - cut_h + 1)
            x1 = np.random.randint(0, W2 - cut_w + 1)

            # 裁剪图像
            new_img2 = img2[y1:y1 + cut_h, x1:x1 + cut_w]
            new_mask = np.ones((cut_h, cut_w), dtype=np.float32)

        # 粘贴到目标图像中的位置
        paste_y = np.random.randint(0, H1 - cut_h + 1)
        paste_x = np.random.randint(0, W1 - cut_w + 1)

        try:
            # 确保不越界粘贴
            cutmix_img = img1.copy()
            cutmix_img[paste_y:paste_y + cut_h, paste_x:paste_x + cut_w] = new_img2
        except ValueError as e:
            print(f"Error encountered: {e}")
            print(f'new_img2 shape: {new_img2.shape}, cutmix_img shape: {cutmix_img.shape}')
            print(f'Paste coordinates: paste_y={paste_y}, paste_x={paste_x}, cut_h={cut_h}, cut_w={cut_w}')
            raise

        mask_label = img1_label.copy()
        mask_label[paste_y:paste_y + cut_h, paste_x:paste_x + cut_w] = 1 # 图像篡改，所以label为1

    cutmix_label = 1 # 凡是cutmix的都是伪造的

    return ori_image, cutmix_img, cutmix_label, mask_label


def mixup_data(img1_path=None, img2_path=None, mask=None, alpha=None, transform=None):
    """
    :mixup 
    :param real_image: 真实图像 (Tensor)
    :param fake_image: 假图像 (Tensor)
    :param mask: 混合 mask(需要mixup的区域)
    :alpha: mixup的比例
    return: 返回混合图像 (Tensor)
    """
    # real_img, _ = read_image(img1_path)
    # fake_img, _ = read_image(img2_path)
    # real_img, fake_img, label1, label2 = apply_transform(real_img, fake_img, 0, 1, transform, is_dire=False)
    
    real_img, fake_img = read_and_crop(img1_path, img2_path)
    # fake_img = read_and_crop(img2_path)
        
    mixup_fake_real = alpha * real_img + (1 - alpha) * fake_img
    # 将图像值限制在 0 到 255 之间，并转换为 uint8
    mixup_fake_real = np.clip(mixup_fake_real, 0, 255).astype(np.uint8)
    
    mask = mask.numpy() 
    mask_expanded = np.repeat(mask[..., None] , 3, axis=-1)
    mixed_img = mask_expanded * real_img + (1 - mask_expanded) * mixup_fake_real    
    mixed_label = 1
    
    real_label = np.full((real_img.shape[0], real_img.shape[1], 3), 0, dtype=np.float32)
    fake_label = np.full((real_img.shape[0], real_img.shape[1], 3), 1, dtype=np.float32)
    mask_label = mask_expanded * real_label + (1 - mask_expanded) * fake_label
    
    
    transform_tensor = A.Compose([ToTensorV2()])
    mixup_img_be_aug = transform_tensor(image=mixed_img)['image'] # 增强之前
    mask_label_tensor = transform_tensor(image=mask_label)['image']
    
    mixup_img_aug = transform(image=mixed_img)['image']
    
    # Save the tensor images after transformation
    # save_image_from_tensor(mixup_img_be_aug, "mixup_img_tensor.png")##########
    # save_image_from_tensor(mask_label_tensor, "mixup_mask_label_tensor.png")###########
    
    return mixup_img_aug, mixup_img_be_aug, mixed_label, mask_label_tensor[0,:,:]

if __name__ == '__main__':
    import torch
    import math
    import random
    import torchvision.transforms as transforms
    from PIL import Image    
    # from transform import create_train_transforms
    
    # 生成 mask
    lam = 0.5
    H, W = 225, 480
    test_mask = generate_patch_mask(H, W, lam)

    # 将 mask 转换为 0-255 的整数范围，以便保存为图像
    test_mask = (test_mask * 255).astype(np.uint8)

    # 保存为 PNG 图像
    mask_image = Image.fromarray(test_mask.squeeze(), mode='L')
    mask_image.save('test_mask.png')
    
    ori_image, cutmix_img, cutmix_label, mask_label = cutmix_data(
        img1_path='/root/autodl-tmp/AIGC_data/MSCOCO/train2017/000000000071.jpg', 
        img2_path='/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-v1-4/train2017/000000000025.jpg', 
        label1=0, label2=1, lam=0.5) 
#     cutmix_img_aug, cutmix_img_be_aug, aug_label, aug_mask_label = mixup_data(        
#     img1_path='/root/autodl-tmp/AIGC_data/MSCOCO/train2017/000000000025.jpg', 
#     img2_path='/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-inpainting/train2017/000000000025.png', 
#     mask=test_mask, alpha=0.5, transform=create_train_transforms(224))
    
#     print('cutmix_label:', aug_label)
    
    # 保存 cutmix_img 为图像文件
    cutmix_img = Image.fromarray(cutmix_img)
    cutmix_img.save('cutmix_img.png')