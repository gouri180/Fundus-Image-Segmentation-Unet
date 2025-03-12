
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import imageio
import glob

class fundus_dataset(Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.no_samples = len(image_path)
        
    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) #shape(height, width, 3)
        image = cv2.resize(image, (512,512))
        image = image/255.0 
        image = np.transpose(image, (2,0,1)) #(3, height, width)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        
        mask_path = self.mask_path[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale
        if mask is None:
            raise FileNotFoundError(f"Mask file not found at path: {self.mask_path[index]}")
        
        mask = cv2.resize(mask, (512,512))
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask) 
        
        return image, mask
    def __len__(self):
        return self.no_samples


