import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './././'))
sys.path.append(project_root)

class CustomDataset(Dataset):
    def __init__(self, rgb_image_patches, 
                chm_image_patches, 
                mask_image_patches, 
                source_ids=None, 
                rgb_image_transform=None, 
                chm_image_transform=None, 
                mask_image_transform=None):

        self.rgb_image_patches = rgb_image_patches
        self.chm_image_patches = chm_image_patches
        self.mask_image_patches = mask_image_patches
        self.source_ids = source_ids
        self.rgb_image_transform = rgb_image_transform
        self.chm_image_transform = chm_image_transform
        self.mask_image_transform = mask_image_transform


    def __len__(self):
        return len(self.rgb_image_patches)
        
    def __getitem__(self, idx):
        x = self.rgb_image_patches[idx]
        y = self.chm_image_patches[idx]
        mask = self.mask_image_patches[idx] if self.mask_image_patches is not None else None
        source_id = self.source_ids[idx] if self.source_ids is not None else None

        if x.ndim == 3:
            x = Image.fromarray(x)
        else:
            raise ValueError("RGB image must be a 3D array.")

        if y.ndim == 2:
            if y.dtype == np.float16:
                y = y.astype(np.float32)
            y = Image.fromarray(y)
        else:
            raise ValueError("CHM image must be a 2D array.")
        
        if mask is not None:
            if mask.ndim == 2:
                mask = Image.fromarray(mask)
            else:
                raise ValueError("Mask must be a 2D array.")
            
        if self.rgb_image_transform:
            x = self.rgb_image_transform(x)
        if self.chm_image_transform:
            y = self.chm_image_transform(y)
        if mask is not None and self.mask_image_transform is not None:
            mask = self.mask_image_transform(mask)
        if mask is not None:
            return x, y, mask, source_id
        else:
            return x, y, source_id
