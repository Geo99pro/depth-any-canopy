import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './././'))
sys.path.append(project_root)
import numpy as np
from typing import Callable
from torch.utils.data import DataLoader
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

        self.glob_train_loader, self.glob_val_loader = [], []
        for i, (img, chm) in enumerate(zip(self.rgb_images, self.chm_images)):
            #focus only on shape = (H, W, 3) for now
            if img.shape[-1] ==3:
                self.image = img
                self.chm_image = chm
                self._process_rgb_only()
                self.glob_train_loader.append(self.train_loader)
                self.glob_val_loader.append(self.val_loader)
                

        if self.image.shape[-1] == 4:
            self._process_with_nir()
        elif self.image.shape[-1] == 3:
            self._process_rgb_only()
        else:
            raise ValueError("Unsupported image format. Expected RGB or RGB+NIR.")
        
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
    
    def _process_with_nir(self):
        print("🟢 NIR band found. Using RGB, NIR, and CHM.")
        rgb, nir = self.image[:, :, :3], self.image[:, :, 3]
        binary_mask, ndvi_mask = Utils.get_binary_mask(nir, rgb, self.args.get('ndvi_thr', 0.1), self.args.get('output_path'))
        metadata = Utils.split_images(rgb, self.chm_image, ndvi_mask, binary_mask, self.args.get('split_type', 'horizontal'), self.args.get('output_path'))

        self._visualize_histograms(metadata)

        top_patches = self._extract_patches(metadata['top'])
        bottom_patches = self._extract_patches(metadata['bottom'])

        if self.args.get('top_side_as_train', True):
            self.train_loader, self.val_loader, self.test_loader = self._make_loaders(top_patches, bottom_patches)
            self.min_depth, self.max_depth = metadata['top']['min_depth'], metadata['top']['max_depth']
        else:
            self.train_loader, self.val_loader, self.test_loader = self._make_loaders(bottom_patches, top_patches)
            self.min_depth, self.max_depth = metadata['bottom']['min_depth'], metadata['bottom']['max_depth']

        if self.args.get('visualize_patches', True):
            self._visualize_patches()

    def _process_rgb_only(self):
        print("🟢 RGB only. Using RGB and CHM.")
        Utils.visualize_histogram(self.chm_image, bins=100, title="CHM Distribution", xlabel="Height", ylabel="Freq", path_to_save=self.args.get('output_path'), name="chm_histogram")
        binary_mask = np.where(self.chm_image > 0, 1, 0).astype(np.uint8)
        min_depth, max_depth = Utils.get_min_max_depth(binary_mask, self.chm_image)

        patches = self._extract_patches({
            'rgb_image': self.image.astype(np.uint8),
            'chm_image': self.chm_image.astype(np.float16),
            'binary_mask': binary_mask
        })

        self.train_loader, self.val_loader, self.test_loader = self._make_loaders(patches, None)
        self.min_depth, self.max_depth = min_depth, max_depth
        if self.args.get('visualize_patches', True):
            self._visualize_patches()

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
    
    def _make_loaders(self, train_data, test_data):
        """
        Creates DataLoaders for training, validation, and testing datasets.
        Args:
            train_data (dict): Dictionary containing training data with keys 'rgb', 'chm', and 'mask'.
            test_data (dict or None): Dictionary containing testing data with keys 'rgb', 'chm', and 'mask'. If None, no test loader is created.

        Returns:
            tuple: A tuple containing train_loader, val_loader, and test_loader.
        """
        indices = list(range(len(train_data['rgb'])))
        train_idx, val_idx = Utils.split_dataset_indices(indices, test_size=self.split_ratio)

        def subset(data, idx):
            return {
                'rgb_image_patches': data['rgb'][idx],
                'chm_image_patches': data['chm'][idx],
                'mask_image_patches': data['mask'][idx],
                'rgb_image_transform': self.input_transform,
                'chm_image_transform': self.target_transform,
                'mask_image_transform': self.mask_transform
            }

        train_dataset = CustomDataset(**subset(train_data, train_idx))
        val_dataset = CustomDataset(**subset(train_data, val_idx))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        test_loader = None
        if test_data:
            test_dataset = CustomDataset(**subset(test_data, slice(None)))
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
    
    def _visualize_histograms(self, metadata):
        """Visualizes histograms for CHM distributions in the top and bottom sides.
        Args:
            metadata (dict): Dictionary containing metadata for top and bottom sides.

        Returns:
            None. Howerever, it saves the histograms as images in the specified output path.
        """
        for side in ['top', 'bottom']:
            Utils.visualize_histogram(image=metadata[side]['chm_image'], bins=100,
                                      title=f"{side.title()} CHM Distribution", xlabel="Depth", ylabel="Freq",
                                      path_to_save=self.args.get("output_path"), name=f"{side}_CHM_Distribution")
    
    def _visualize_patches(self):
        """Visualizes patches from the training, validation, and test datasets.
        This method uses the Utils class to visualize patches and save them to the specified output path.
        """
        print("🟢 Visualizing patches from train, val, and test datasets.")
        for loader_name, loader in zip(['train', 'val', 'test'], [self.train_loader, self.val_loader, self.test_loader]):
            if loader is not None:
                Utils.visualize_patches(loader, how_many_patches=4, path_to_save=self.args.get('output_path'), name=loader_name)

    def get_loader_and_depth(self):
        """
        Returns the DataLoaders for training, validation, and testing datasets,
        along with the minimum and maximum depth values from the CHM data.
        Returns:
            tuple: A tuple containing train_loader, val_loader, test_loader, min_depth, and max_depth.
        """
        return self.train_loader, self.val_loader, self.test_loader, self.min_depth, self.max_depth
    
    def get_test_loader(self, source_tiff_path: str, reference_tiff_path: str):
        """
        Prepares a DataLoader from a single RGB + CHM image pair, used for testing/inference.

        Args:
            source_tiff_path (str): Path to the RGB (or RGB+NIR) TIFF file.
            reference_tiff_path (str): Path to the CHM file.

        Returns:
            tuple: (DataLoader, min_depth, max_depth)
        """
        print(f"🟢 Loading test dataset from {source_tiff_path} and {reference_tiff_path}")
        img = Utils.read_tiff(source_tiff_path)
        chm_image = np.squeeze(Utils.read_tiff(reference_tiff_path))

        binary_mask = np.where(chm_image > 0, 1, 0).astype(np.uint8)
        min_depth, max_depth = Utils.get_min_max_depth(binary_mask, chm_image)

        rgb_patches, chm_patches, mask_patches = Utils.extract_images_patches(
            rgb_image=img.astype(np.uint8),
            reference_image=chm_image.astype(np.float16),
            binary_mask_image=binary_mask,
            patch_size=self.patch_size,
            stride_ratio=self.overlap_ratio
        )

        test_dataset = CustomDataset(rgb_patches, chm_patches, mask_patches,
            self.input_transform, self.target_transform, self.mask_transform
        )
        print(f"🟢 Test dataset created with {len(test_dataset)} patches.")
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader, min_depth, max_depth


# if __name__ == "__main__":
#     # Example usage
#     process_data_instance = ProcessData(
#         image_path='/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/rgb_LIDAR/RGBNIR.tif',
#         chm_path='/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/rgb_LIDAR/CHM.tif',
#         split_ratio=0.2,
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#         batch_size=32,
#         num_workers=4,
#         patch_size=518,
#         overlap_ratio=0.15,
#         ndvi_thr=0.1,
#         split_type='horizontal',
#         top_side_as_train=True,
#         visualize_patches=True,
#         output_path='/home/loick/DL-FOLDER/testing'
#     )
#    train_loader, val_loader, test_loader, min_depth, max_depth = process_data_instance.get_loader_and_depth()
#    print(f"Train Loader: {train_loader}, Val Loader: {val_loader}, Test Loader: {test_loader}, Min Depth: {min_depth}, Max Depth: {max_depth}")