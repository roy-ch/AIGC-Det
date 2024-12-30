from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2

def albumentations_transform(image, mean, std, size):
    """
    DRCT augmentation implemented bu Albumentations
    """
    resize_fuc = A.RandomCrop(height=size, width=size)
    transform = A.Compose([
        A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
        A.RandomScale(scale_limit=(-0.5, 0.5), p=0.2),  # 23/11/04 add
        A.HorizontalFlip(),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(p=0.1),
        A.RandomRotate90(),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        resize_fuc,
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.5),
        A.OneOf([A.CoarseDropout(), A.GridDropout()], p=0.5),
        A.ToGray(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    return transform(image=image)["image"]

# 转换器：PIL -> NumPy
class PILToAlbumentations:
    def __init__(self, albumentations_transforms, mean, std, size):
        self.albumentations_transform = albumentations_transforms
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, img):
        # Convert PIL.Image to NumPy array
        img = np.array(img)
        # Ensure image is in HWC format and 3 channels
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # Apply Albumentations transform
        img = self.albumentations_transform(img, self.mean, self.std, self.size)
        return img
    
# 转换器：NumPy -> PIL
class AlbumentationsToPIL:
    def __call__(self, tensor):
        # Convert PyTorch tensor to NumPy array
        img = tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        img = (img * 255).astype(np.uint8)  # Denormalize to 0-255
        return Image.fromarray(img)