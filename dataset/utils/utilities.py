import os
import torch
import kornia
import rasterio
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from osgeo import gdal
gdal.DontUseExceptions()
from typing import Callable
from torchvision import transforms as T
from torch.utils.data import DataLoader
from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split
from kornia.augmentation import AugmentationSequential

class Utils:
    @staticmethod
    def read_tiff(file_path):
        data = gdal.Open(file_path).ReadAsArray()
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0)) # Transpose to (height, width, channels)
            data = np.array(data, dtype=np.uint8)
        elif data.ndim == 2:
            data = np.array(data, dtype=np.uint8)
        return data
    
    @staticmethod
    def apply_transformation(mean: tuple[float, float, float] = (0.420, 0.411, 0.296),
                            std: tuple[float, float, float] = (0.213, 0.156, 0.143),
                            size: tuple[int, int] = (518, 518),
                            is_input: bool = True) -> Callable:
        if is_input:
            return T.Compose([T.Resize(size),
                            T.ToTensor(),
                            T.Normalize(mean=mean, std=std)])
        else:
            return T.Compose([T.Resize(size),
                              T.ToTensor()])

    @staticmethod
    def denormalize(tensor: torch.Tensor, 
                mean: tuple[float, float, float] = (0.420, 0.411, 0.296),
                std: tuple[float, float, float] = (0.213, 0.156, 0.143)) -> torch.Tensor:
        """
        Denormalize a tensor using the provided mean and standard deviation.

        Args:
            tensor (torch.Tensor): The input tensor to denormalize.
            mean (tuple[float, float, float]): The mean values for each channel.
            std (tuple[float, float, float]): The standard deviation values for each channel.
        Returns:
            torch.Tensor: The denormalized tensor.
        """
        mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
        return tensor * std + mean

    @staticmethod
    def get_augmentation_pipeline(mean: tuple[float, float, float] = (0.420, 0.411, 0.296),
                                  std: tuple[float, float, float] = (0.213, 0.156, 0.143)) -> Callable:
        return AugmentationSequential(
            kornia.enhance.Normalize(mean=mean, std=std),
            data_keys=["input"])

    @staticmethod
    def visualize_dataset(image,
                        figure_size=(8, 8),
                        path_to_save: str = None,
                        **args):
        """
        Visualize an image with RGB and NIR channels if available.
        This function checks the number of channels in the image and visualizes it accordingly.
        If the image has 3 channels, it is assumed to be an RGB image. If it has 4 channels, the last channel is assumed to be NIR (In this work context).

        Args:
                image (numpy.ndarray): The image to visualize.
                figure_size (tuple): Size of the figure for visualization.
        Returns:
                None. However, it displays the image.
        """

        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")
        
        image_shape = image.shape
        if len(image_shape) > 2:
            if image_shape[-1] == 3:
                rgb_image = image[:, :, :3]
                if np.max(rgb_image) > 1:
                    rgb_image = rgb_image / 255.0
                plt.imshow(rgb_image)
                plt.title('RGB Image')
                plt.axis('off')
                if path_to_save:
                    fig_path = os.path.join(path_to_save, 'rgb_image.png')
                    plt.savefig(fig_path)
                plt.close()

            elif image_shape[-1] == 4:
                rgb_image = image[:, :, :3]
                nir_image = image[:, :, 3]
                #create a subplot with 2 images
                if np.max(rgb_image) > 1:
                    rgb_image = rgb_image / 255.0
                if np.max(nir_image) > 1:
                    nir_image = nir_image / 255.0
                
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
                ax1.imshow(rgb_image)
                ax1.set_title('Ortofoto RGB Image')
                ax1.axis('off')
                ax2.imshow(nir_image, cmap=args.get('cmap', 'gray'))
                ax2.set_title('Ortofoto NIR Image')
                ax2.axis('off')
                plt.tight_layout()
                if path_to_save:
                    fig_path = os.path.join(path_to_save, 'rgb_nir_image.png')
                    plt.savefig(fig_path)
                plt.close()
                
        elif len(image_shape) == 2:
            rgb_image = image
            if np.max(rgb_image) > 1:
                rgb_image = rgb_image / 255.0
            plt.imshow(rgb_image, cmap = args.get('cmap', 'gray'))
            plt.title(args.get('title', 'Canopy Height Map'))
            plt.axis('off')
            if path_to_save:
                fig_path = os.path.join(path_to_save, 'canopy_height_map.png')
                plt.savefig(fig_path)
            plt.close()
        else:
            raise ValueError("Input image must be a 2D or 3D numpy array.")

    @staticmethod
    def visualize_patches(loader: DataLoader, 
                          how_many_patches: int = 4,
                          path_to_save: str = None):
        """
        Visualize patches from a given DataLoader.

        Parameters:
            loader (DataLoader): DataLoader to visualize patches from. It contains batches of rgb images, labels and masks.
            how_many_patches (int): Number of patches to visualize.
            path_to_save (str): Optional path to save the visualized patches.
        """

        images, labels, masks = [], [], []
        for x, y, mask in loader:
            for i in range(len(x)):
                images.append(Utils.denormalize(x[i]).numpy())
                labels.append(y[i].numpy())
                masks.append(mask[i].numpy())
                if len(images) == how_many_patches:
                    break
            if len(images) == how_many_patches:
                break

        ncols = min(how_many_patches, 4)
        nrows = (how_many_patches + ncols - 1) // ncols
        _, axs = plt.subplots(nrows=nrows * 3, ncols=ncols, figsize=(ncols * 4, nrows * 4 * 1.5))
        axs = np.array(axs).reshape(nrows * 3, ncols)

        for i in range(how_many_patches):
            row = (i // ncols) * 3
            col = i % ncols

            axs[row, col].imshow(np.clip(images[i].transpose(1, 2, 0), 0, 255)) # CHW to HWC
            axs[row, col].axis('off')
            axs[row, col].set_title('Satellite RGB Image')

            axs[row + 1, col].imshow(labels[i].squeeze(0), cmap='plasma') #label image comes with (1, H, W) shape
            axs[row + 1, col].axis('off')
            axs[row + 1, col].set_title('CHM Image')

            axs[row + 2, col].imshow(mask[i].squeeze(0), cmap='gray', alpha=0.5) #mask image comes with (1, H, W) shape
            axs[row + 2, col].axis('off')
            axs[row + 2, col].set_title('NDVI Binary Mask Overlay')


        plt.tight_layout()
        if path_to_save:
            fig_path = os.path.join(path_to_save, f'{how_many_patches}patches_visualization.png')
            plt.savefig(fig_path)
            print(f"🖼️ Patches saved under: {fig_path}")
        plt.close()

    @staticmethod
    def extract_images_patches(rgb_image,
                            reference_image,
                            binary_mask_image: np.ndarray,
                            patch_len: int = 256,
                            stride_ratio: float = 0.5):
        """
        Extract patches from the input image using a sliding window approach.
        
        Parameters:
            channel (numpy.ndarray): Input image channel.
            patch_len (int): Length of the patches to be extracted.
        """
        channel_n = 3
        train_step = int(patch_len * stride_ratio)


        print(f'Image shape: {rgb_image.shape}, CHM: {reference_image.shape}, Mask: {binary_mask_image.shape}')
        print(f'Patch size: {patch_len}, Step: {train_step} → Overlap: {100 - stride_ratio * 100:.0f}%')
        
        rgb_image_patch = view_as_windows(rgb_image, (patch_len, patch_len, channel_n), step=train_step)
        reference_image_patch = view_as_windows(reference_image, (patch_len, patch_len), step=train_step)
        mask_image_patch = view_as_windows(binary_mask_image, (patch_len, patch_len), step=train_step)

        rgb_image_patch = rgb_image_patch.reshape(-1, patch_len, patch_len, channel_n)
        reference_image_patch = reference_image_patch.reshape(-1, patch_len, patch_len)
        mask_image_patch = mask_image_patch.reshape(-1, patch_len, patch_len)
        print(f"🤖 The number of patches obtained is: {rgb_image_patch.shape[0]}")
        return rgb_image_patch, reference_image_patch, mask_image_patch

    @staticmethod
    def split_dataset(input: np.ndarray, 
            reference: np.ndarray,
            test_size: float = 0.2, 
            random_state: int = 42):
        """
        Split the dataset into training and testing sets.

        Parameters:
            input (numpy.ndarray): Input images.
            reference (numpy.ndarray): Reference images.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: Training and testing sets.
        """
        return train_test_split(input, reference, test_size=test_size, random_state=random_state)

    @staticmethod
    def split_dataset_indices(all_indices: list,
                            test_size: float = 0.2, 
                            random_state: int = 42):
        """
        Split the dataset indices into training and testing sets.

        Parameters:
            all_indices (list): List of all indices.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[list, list]: Training and testing indices.
        """
        return train_test_split(all_indices, test_size=test_size, random_state=random_state)
    
    @staticmethod
    def get_canopy_height_pixel_resolution(canopy_height_map_path: str) -> float:
        """
        Get the pixel resolution of the canopy height map.

        Parameters:
            canopy_height_map_path (str): Path to the canopy height map.

        Returns:
            float: Pixel resolution of the canopy height map.
        """
        with rasterio.open(canopy_height_map_path) as src:
            #print(src.profile)
            resolution = src.res
        print(f"Resolution (meters/pixel): {resolution}")

    @staticmethod
    def get_canopy_height_bounds(image: np.ndarray) -> tuple[float, float]:
        """
        Get the minimum and maximum canopy height from the image.

        Parameters:
            image (numpy.ndarray): Input image.
            
        Returns:
            tuple[float, float]: Minimum and maximum canopy height.
        """
        min_canopy_height = np.min(image)
        max_canopy_height = np.max(image)
        print(f"Minimum canopy height: {min_canopy_height}\nMaximum canopy height: {max_canopy_height}")

    @staticmethod
    def get_binary_mask(nir_image: np.ndarray,
                        rgb_image: np.ndarray,
                        ndvi_threshold: float = 0.5,
                        figure_size: tuple[int, int] = (8, 8),
                        path_to_save: str = None) -> np.ndarray:
        """
        Get a binary mask from the NIR image using a threshold.
        
        Parameters:
            nir_image (numpy.ndarray): NIR image.
            rgb_image (numpy.ndarray): RGB image.
            ndvi_threshold (float): Threshold for binary mask.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Binary mask and NDVI.
        """
        red_channel = rgb_image[:, :, 0]
        if np.max(red_channel) > 1:
            red_channel = red_channel / 255.0
        if np.max(nir_image) > 1:
            nir_image = nir_image / 255.0
        ndvi  = (nir_image - red_channel) / (nir_image + red_channel + 1e-8)
        binary_mask = np.where(ndvi  > ndvi_threshold, 1, 0).astype(np.uint8)
        if path_to_save:
            Utils.visualize_dataset(binary_mask, figure_size=figure_size, path_to_save=path_to_save, title='Binary Mask', cmap='gray')
            plt.figure(figsize=figure_size)
            plt.imshow(rgb_image)
            plt.imshow(binary_mask, alpha=0.5, cmap='gray')
            plt.axis('off')
            plt.title('Binary Mask Overlay')
            plt.savefig(os.path.join(path_to_save, 'binary_mask_overlay.png'))
            plt.close()
        return binary_mask, ndvi