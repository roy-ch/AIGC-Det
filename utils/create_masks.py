from PIL import Image
from tqdm import tqdm

import os
import numpy as np

subset_name = "stable-diffusion-2-inpainting"

train_path = "/root/autodl-tmp/AIGC_data/DRCT-2M/{}/train2017".format(subset_name)
val_path = "/root/autodl-tmp/AIGC_data/DRCT-2M/{}/val2017".format(subset_name)

mask_train_path = "/root/autodl-tmp/AIGC_data/DRCT-2M/{}/masks/train2017".format(subset_name)
mask_val_path = "/root/autodl-tmp/AIGC_data/DRCT-2M/{}/masks/val2017".format(subset_name)

# train_path = "/root/autodl-tmp/AIGC_data/MSCOCO/train2017"
# val_path = "/root/autodl-tmp/AIGC_data/MSCOCO/val2017"

# mask_train_path = "/root/autodl-tmp/AIGC_data/MSCOCO/masks/train2017"
# mask_val_path = "/root/autodl-tmp/AIGC_data/MSCOCO/masks/val2017"

real = False
if real:
    label = 0
else:
    label = 255

def create_single_mask(image_path):
    image = np.array(Image.open(image_path))
    try:
        h, w, c = image.shape
    except:
        h, w = image.shape
    mask = Image.new("RGB", (w, h), color=(label,label,label))
    
    return mask

def create_masks(img_folder, mask_folder):
    os.makedirs(mask_folder, exist_ok=True)
    
    img_list = os.listdir(img_folder)
    with tqdm(total=len(img_list)) as pbar:
        for img in img_list:
            img_path = os.path.join(img_folder, img)
            img_name = img.split(".")[0]

            mask = create_single_mask(img_path)
            mask.save(os.path.join(mask_folder, img_name+".png"))
            
            pbar.update(1)
            
create_masks(train_path, mask_train_path)
create_masks(val_path, mask_val_path)