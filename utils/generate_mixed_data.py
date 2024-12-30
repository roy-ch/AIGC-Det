from data.cutmix import *

import random
from PIL import Image
from tqdm import tqdm

import os
import numpy as np
import torch
import torchvision.transforms as transforms

# list sequence: [real_train, real_val, fake_train, fake_val]

# Source for mixing up
subset_name = "stable-diffusion-2-inpainting"

paths = [
    "/root/autodl-tmp/AIGC_data/MSCOCO/train2017",
    "/root/autodl-tmp/AIGC_data/MSCOCO/val2017",
    "/root/autodl-tmp/AIGC_data/DRCT-2M/{}/train2017".format(subset_name),
    "/root/autodl-tmp/AIGC_data/DRCT-2M/{}/val2017".format(subset_name)
]

# Image store path
save_paths = [
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/cutmix-mscoco/train2017",
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/cutmix-mscoco/val2017",
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/{}/train2017".format(subset_name),
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/{}/val2017".format(subset_name)
]

# Mask store path
mask_save_paths = [
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/cutmix-mscoco/masks/train2017",
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/cutmix-mscoco/masks/val2017",
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/{}/masks/train2017".format(subset_name),
    "/root/autodl-tmp/AIGC_data/CUTMIX_DRCT-2M/{}/masks/val2017".format(subset_name)
]

# Generation settings
LAM = [0.2, 0.3, 0.4, 0.5, 0.6]

FAKE_TRAIN_NUM = 50000
FAKE_VAL_NUM = FAKE_TRAIN_NUM // 20
REAL_TRAIN_NUM = 5000
REAL_VAL_NUM = REAL_TRAIN_NUM // 20

SETTING_NUMS = [REAL_TRAIN_NUM, REAL_VAL_NUM, FAKE_TRAIN_NUM, FAKE_VAL_NUM]

def get_image_paths(dir_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_paths = []
    for root, dirs, files in sorted(os.walk(dir_path)):
        for file in sorted(files):
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def resize_to_match(img_tensor, target_tensor):
    # 获取目标图片的高度和宽度
    target_size = target_tensor.shape[-2:]  # (H, W)
    # 使用 torchvision.transforms.Resize 调整大小
    transform = transforms.Resize(target_size)
    resized_img_tensor = transform(img_tensor)
    return resized_img_tensor

def main():
    # Initialize
    
    for save_path in save_paths + mask_save_paths:
        os.makedirs(save_path, exist_ok=True)
        os.system("rm -rf {}/*".format(save_path))
        
    img_lists = []
    for path in paths:
        img_list = [{"image_path": image_path, "label": 0 if "MSCOCO" in path else 1} for image_path in get_image_paths(path)]
        random.shuffle(img_list)
        img_lists.append(img_list)
        
    for i in range(4):
        with tqdm(total=SETTING_NUMS[i]) as pbar:
            j = 0
            bias = 0
            while j < SETTING_NUMS[i]+bias:
                if i in [0, 1]:
                    img_path_1 = random.choice(img_lists[i])
                    img_path_2 = random.choice(img_lists[i])
                elif i in [2, 3]:
                    img_path_1 = random.choice(img_lists[i-2])
                    img_path_2 = random.choice(img_lists[i])
                
                    
                img_1 = torch.tensor(np.array(Image.open(img_path_1["image_path"]).convert("RGB"))).permute(2, 0, 1)
                label_1 = img_path_1["label"]
                img_2 = torch.tensor(np.array(Image.open(img_path_2["image_path"]).convert("RGB"))).permute(2, 0, 1)
                img_2 = resize_to_match(img_2, img_1)
                label_2 = img_path_2["label"]
                
                try:
                    lam = random.choice(LAM)
                    img, label, mask = mix_and_save_images(img_1, img_2, label_1, label_2, torch.tensor(lam))

                    img = img.permute(1, 2, 0)
                    img = img.detach().cpu().numpy()
                    img = Image.fromarray(img)
                    
                    img.save(os.path.join(save_paths[i], f"{j}_lam_{lam}.jpg"))
                    
                    mask = mask.permute(1, 2, 0)
                    mask = mask.detach().cpu().numpy()
                    mask = Image.fromarray(mask)
                    mask.save(os.path.join(mask_save_paths[i], f"{j}_lam_{lam}.png"))
                    
                    pbar.update(1)
                except ValueError as e:
                    print(e)
                    bias += 1
                
                j += 1
                
                
    
if __name__ == "__main__":
    main()
    