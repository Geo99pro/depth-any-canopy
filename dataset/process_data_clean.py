import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './././'))
sys.path.append(project_root)
import numpy as np
from typing import Callable
from torch.utils.data import DataLoader, ConcatDataset
from custom_dataset import CustomDataset
from utils.utilities import Utils


class ProcessData:
    def __init__(self,
                 images_folder_path: str,
                 chm_folder_path: str,
                 split_ratio: float = 0.8,
                 mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: tuple[float, float, float] = (0.229, 0.224, 0.225),
                 batch_size: int = 32,
                 num_workers: int = 4,
                 patch_size: int = 256,
                 overlap_ratio: float = 0.1,
                 **args):
        """
        This class processes the input image and CHM (Canopy Height Model) data,
        extracts patches, and prepares DataLoaders for training, validation, and testing.
        
        Parameters:
            images_folder_path (str): Path to the folder containing RGB or RGB+NIR images.
            chm_folder_path (str): Path to the folder containing CHM images.
            split_ratio (float): Ratio to split the dataset into training and validation sets.
            mean (tuple): Mean values for normalization.
            std (tuple): Standard deviation values for normalization.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            patch_size (int): Size of the patches to extract from the images.
            overlap_ratio (float): Overlap ratio for patch extraction.
            args (dict): Additional arguments for processing, such as NDVI threshold, output path, etc.

        Raises:
            ValueError: If the image format is unsupported (not RGB or RGB+NIR).
        """
        
        self.args = args
        self.images_folder_path = images_folder_path
        self.chm_folder_path = chm_folder_path
        self.split_ratio = split_ratio
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio

        self.input_transform, self.target_transform, self.mask_transform = self._prepare_transforms()
        self.rgb_images, self.chm_images = self._get_data()
        
        self.depth_bounds = []
        self.source_ids = []
        self.data = self._gather_all_patches()
        self.train_loader, self.val_loader = self._make_loaders(self.data)
        if self.args.get('visualize_patches', True):
            self._visualize_patches(self.train_loader, self.val_loader, None, np.random.randint(0, 10))
    
    def _get_data(self):
        rgb_images, depth_images = [], []
        for rgb, depth in Utils.read_tif_image(self.images_folder_path, self.chm_folder_path):
            rgb_images.append(rgb)
            depth_images.append(depth)
        return rgb_images, depth_images

    def _prepare_transforms(self):
        input_transform = Utils.apply_transformation(mean=self.mean, std=self.std, is_input=True) #images
        target_transform = Utils.apply_transformation(mean=self.mean, std=self.std, is_input=False, is_mask=False) #chm
        mask_transform = Utils.apply_transformation(mean=self.mean, std=self.std, is_input=False, is_mask=True) #mask
        return  input_transform, target_transform, mask_transform   

    def _gather_all_patches(self):
        """
        Gathers all patches from the RGB and CHM images, extracts binary masks,
        and computes depth bounds for each image pair.
        
        Returns:
            dict: A dictionary containing concatenated RGB patches, CHM patches, binary masks, and source IDs."""
        all_rgb, all_chm, all_mask, all_source_ids = [], [], [], []
        
        for i, (img, chm) in enumerate(zip(self.rgb_images, self.chm_images)):
            print(f"\n📂 Processing image pair {i+1}/{len(self.rgb_images)}")

            if img.shape[-1] == 3:
                rgb = img
                chm_image = chm
                binary_mask = np.where(chm_image >= 5, 1, 0).astype(np.uint8)
                Utils.visualize_histogram(chm_image, bins=100, title="CHM Distribution", xlabel="Height", ylabel="Freq", 
                                  path_to_save=self.args.get('output_path'), name=f"chm_histogram_{i}")
                Utils.visualize_histogram(binary_mask, bins=2, title="Binary Mask Distribution", xlabel="Mask Value", ylabel="Freq",
                                  path_to_save=self.args.get('output_path'), name=f"binary_mask_histogram_{i}")
            elif img.shape[-1] == 4:
                rgb, nir = img[:, :, :3], img[:, :, 3]
                chm_image = chm
                binary_mask, _ = Utils.get_binary_mask(nir, rgb, self.args.get('ndvi_thr', 0.1), self.args.get('output_path'))
            else:
                raise ValueError("Unsupported image format. Expected RGB or RGB+NIR.")

            patches = self._extract_patches({
                'rgb_image': rgb.astype(np.uint8),
                'chm_image': chm_image.astype(np.float16),
                'binary_mask': binary_mask
            })
            num_patches = patches['rgb'].shape[0]
            all_rgb.append(patches['rgb'])
            all_chm.append(patches['chm'])
            all_mask.append(patches['mask'])
            all_source_ids.append(np.full((num_patches,), i))

            min_depth, max_depth, real_min, real_max = Utils.get_min_max_depth(binary_mask, chm_image)
            self.depth_bounds.append((min_depth, max_depth, real_min, real_max))

        all_rgb = np.concatenate(all_rgb, axis=0)
        all_chm = np.concatenate(all_chm, axis=0)
        all_mask = np.concatenate(all_mask, axis=0)
        all_source_ids = np.concatenate(all_source_ids, axis=0)

        return {
            'rgb': all_rgb,
            'chm': all_chm,
            'mask': all_mask,
            'source_ids': all_source_ids
        }
    
    def _extract_patches(self, data):
        """
        Extracts patches from the provided data.
        Args:
            data (dict): Dictionary containing 'rgb_image', 'chm_image', and 'binary_mask'.
        
        Returns:
            dict: Dictionary containing extracted patches for RGB, CHM, and mask.
        """
        rgb_patches, chm_patches, mask_patches = Utils.extract_images_patches(
            rgb_image=data['rgb_image'].astype(np.uint8),
            reference_image=data['chm_image'].astype(np.float16),
            binary_mask_image=data['binary_mask'].astype(np.uint8),
            patch_size=self.patch_size,
            stride_ratio=self.overlap_ratio
        )
        return {
            'rgb': rgb_patches,
            'chm': chm_patches,
            'mask': mask_patches
        }

    def _make_loaders(self, data):
        """
        Creates DataLoaders for training, validation, and testing datasets.
        Args:
            train_data (dict): Dictionary containing training data with keys 'rgb', 'chm', and 'mask'.
            test_data (dict or None): Dictionary containing testing data with keys 'rgb', 'chm', and 'mask'. If None, no test loader is created.

        Returns:
            tuple: A tuple containing train_loader, val_loader, and test_loader.
        """
        indices = list(range(len(data['rgb'])))
        train_idx, val_idx = Utils.split_dataset_indices(indices, test_size=self.split_ratio)

        def subset(idx):
            return {
                'rgb_image_patches': data['rgb'][idx],
                'chm_image_patches': data['chm'][idx],
                'mask_image_patches': data['mask'][idx],
                'source_ids': data['source_ids'][idx],
                'rgb_image_transform': self.input_transform,
                'chm_image_transform': self.target_transform,
                'mask_image_transform': self.mask_transform
            }

        train_dataset = CustomDataset(**subset(train_idx))
        val_dataset = CustomDataset(**subset(val_idx))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader
    
    def get_loader_and_depth(self):
        return self.train_loader, self.val_loader, self.get_min_depth(), self.get_max_depth()

    def get_min_depth(self):
        return min(depth[0] for depth in self.depth_bounds) if self.depth_bounds else None
    
    def get_max_depth(self):
        return max(depth[1] for depth in self.depth_bounds) if self.depth_bounds else None
    
    def _visualize_patches(self, train_loader, val_loader, test_loader, idx):
        """Visualizes patches from the training, validation, and test datasets.
        This method uses the Utils class to visualize patches and save them to the specified output path.
        """
        print("🟢 Visualizing patches from train, val, and test datasets.")

        for loader_name, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
            if loader is not None:
                Utils.visualize_patches(loader,how_many_patches=4,
                                        path_to_save=self.args.get('output_path'),
                                        name=f"{loader_name}_{idx}_patches")

if __name__ == "__main__":
    # Example usage
    process_data_instance = ProcessData(
        images_folder_path="/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/rbg_images",
        chm_folder_path="/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/targets",
        split_ratio=0.2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        batch_size=32,
        num_workers=4,
        patch_size=518,
        overlap_ratio=0.15,
        visualize_patches=True,
        output_path='/home/loick/DL-FOLDER/testing'
    )
    train_loader, val_loader, min_depth, max_depth = process_data_instance.get_loader_and_depth()
    for batch in train_loader:
        rgb_images, chm_images, masks, source_ids = batch
        print(f"Batch RGB shape: {rgb_images.shape}, CHM shape: {chm_images.shape}, Masks shape: {masks.shape}, Source IDs shape: {source_ids.shape}")
        break
    real_min = [process_data_instance.depth_bounds[s][2] for s in source_ids]
    real_max = [process_data_instance.depth_bounds[s][3] for s in source_ids]

    print(f"Min Depth: {min_depth}, Max Depth: {max_depth}")
    print(f"Real Min Depths: {real_min}, Real Max Depths: {real_max}")