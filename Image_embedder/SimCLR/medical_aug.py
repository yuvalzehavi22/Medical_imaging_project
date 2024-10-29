import argparse
import torch
from torchvision import models

from torchvision.transforms import transforms
from torchvision import transforms, datasets

def medical_aug(img_t, size=224, s=1):
    # Adjust contrast and brightness slightly
    color_jitter = transforms.ColorJitter(brightness=0.1 * s, contrast=0.1 * s)
    
    data_transforms = transforms.Compose([
        transforms.Resize(size=(size, size)),  # Resize the image to avoid losing critical details
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Mild rotation and translation
        transforms.RandomApply([color_jitter], p=0.5),  # Small brightness/contrast changes to simulate imaging variations
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),  # Mild blur to simulate slight image quality variations
        transforms.ToTensor()
    ])
    
    trans = transforms.ToPILImage()
    img = trans(img_t)
    aug_img = data_transforms(img)
    return aug_img