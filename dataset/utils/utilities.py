import os
import torch
import kornia
import rasterio
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from glob import glob
from typing import Callable, Optional
from torchvision import transforms as T
from torch.utils.data import DataLoader
from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split
from kornia.augmentation import AugmentationSequential

class Utils:
    @staticmethod
    def read_tif_image(rbg_img_folder_path: str,
                  depth_img_folder_path: str):
        """
        Read TIFF files from the specified folders and return the data as numpy arrays.
        Args:
            rbg_img_folder_path (str): Path to the folder containing RGB TIFF files.
            depth_img_folder_path (str): Path to the folder containing depth TIFF files.
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: RGB and depth data as numpy arrays.
        """
        rgb_files = sorted(glob(os.path.join(rbg_img_folder_path, '*.tif')))
        depth_files = sorted(glob(os.path.join(depth_img_folder_path, '*.tif')))

        if not rgb_files or not depth_files:
            raise ValueError("No TIFF files found in the specified folders.")
        
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            rgb_image = np.transpose(rasterio.open(rgb_file).read(), (1, 2, 0))  # Transpose to HWC format
            depth_image = np.squeeze(rasterio.open(depth_file).read())  # Read single band depth image
            print(f"Read RGB image from {rgb_file} with shape {rgb_image.shape}")
            print(f"Read depth image from {depth_file} with shape {depth_image.shape}")
            
            if rgb_image.shape[:2] != depth_image.shape:
                raise ValueError(f"Shape mismatch: RGB image {rgb_image.shape} and depth image {depth_image.shape} do not match.")
            yield rgb_image, depth_image

    @staticmethod
    def apply_transformation(mean: tuple[float, float, float] = (0.420, 0.411, 0.296),
                            std: tuple[float, float, float] = (0.213, 0.156, 0.143),
                            is_input: bool = True,
                            is_mask: bool = False) -> Callable:
        
        """
        Return a transformation pipeline for RGB, label (CHM), or binary mask.
        """
        if is_mask:
            return T.Compose([T.PILToTensor(),  
                    T.Lambda(lambda x: x.float())  
                    ])
        if is_input:
            return T.Compose([T.ToTensor(), 
                            T.Normalize(mean=mean, std=std)])
        else:
            return T.Compose([T.ToTensor(), 
                              T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))])
        
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
                fig_path = os.path.join(path_to_save, args.get('name', 'canopy_height_map.png'))
                plt.savefig(fig_path)
            plt.close()
        else:
            raise ValueError("Input image must be a 2D or 3D numpy array.")

    @staticmethod
    def visualize_patches(loader: DataLoader, 
                      how_many_patches: int = 4,
                      path_to_save: Optional[str] = None,
                      **args):
        """
        Visualize patches from a given DataLoader.

        Parameters:
            loader (DataLoader): Yields batches of (x, y) or (x, y, mask).
            how_many_patches (int): Number of patches to visualize.
            path_to_save (str): Optional path to save the visualized patches.
        """
        images, labels, masks = [], [], []

        for batch in loader:
            if len(batch) == 3:
                x, y, mask = batch
                has_mask = True
            elif len(batch) == 2:
                x, y = batch
                mask = None
                has_mask = False
            elif len(batch) == 4:
                x, y, mask, _ = batch
                has_mask = True
            else:
                raise ValueError("DataLoader must return (x, y) or (x, y, mask)")

            for i in range(len(x)):
                images.append(Utils.denormalize(x[i]).numpy())
                labels.append(y[i].numpy())
                if has_mask:
                    masks.append(mask[i].numpy())
                if len(images) == how_many_patches:
                    break
            if len(images) == how_many_patches:
                break

        ncols = min(how_many_patches, 4)
        nrows = (how_many_patches + ncols - 1) // ncols
        fig_height = nrows * (4 if has_mask else 2) * 1.5
        fig, axs = plt.subplots(nrows=nrows * (4 if has_mask else 2), ncols=ncols, figsize=(ncols * 4, fig_height))
        axs = np.array(axs).reshape(nrows * (4 if has_mask else 2), ncols)

        for i in range(how_many_patches):
            row = (i // ncols) * (4 if has_mask else 2)
            col = i % ncols

            axs[row, col].imshow(np.clip(images[i].transpose(1, 2, 0), 0, 255))
            axs[row, col].axis('off')
            axs[row, col].set_title('RGB Image')

            im=axs[row + 1, col].imshow(labels[i].squeeze(), cmap='Spectral_r', vmin=0, vmax=1)
            axs[row + 1, col].set_xlabel(f'Min: {np.min(labels[i]):.2f}, Max: {np.max(labels[i]):.2f}')
            axs[row + 1, col].axis('off')
            axs[row + 1, col].set_title('CHM Image')
            #fig.colorbar(im, ax=axs[row + 1, col])

            if has_mask:
                im_2= axs[row + 2, col].imshow(masks[i].squeeze(), cmap='gray', alpha=0.5)
                axs[row + 2, col].axis('off')
                axs[row + 2, col].set_title('Binary Mask')
                #fig.colorbar(im_2, ax=axs[row + 2, col])

                axs[row + 3, col].imshow(masks[i].squeeze() * labels[i].squeeze(), cmap='viridis', alpha=0.5)
                axs[row + 3, col].axis('off')
                axs[row + 3, col].set_title('Masked CHM')
                axs[row + 3, col].set_xlabel(f'Patch {i + 1}')
                #fig.colorbar(im_2, ax=axs[row + 3, col])

        plt.tight_layout()
        if path_to_save:
            fig_path = os.path.join(path_to_save, f"{args.get('name')}_{how_many_patches}_patches_visualization.png")
            plt.savefig(fig_path)
            print(f"🖼️ Patches saved under: {fig_path}")
        plt.close()

    @staticmethod
    def visualize_histogram(image: np.ndarray, 
                            bins: int = 100,  
                            title: str = 'Histogram', 
                            xlabel: str = 'Pixel Value', 
                            ylabel: str = 'Frequency',
                            path_to_save: str = None,
                            name: str = 'histogram'):
        """
        Visualize the histogram of an image.
        
        Parameters:
            image (numpy.ndarray): Input image.
            bins (int): Number of bins for the histogram.
            title (str): Title of the histogram plot.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            path_to_save (str): Optional path to save the histogram plot.
            name (str): Name of the histogram plot file.
            
        Returns:
            None. However, it saves the histogram plot if a path is provided.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(image.ravel(), bins=bins, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        if path_to_save:
            figure_path = os.path.join(path_to_save, f'{name}.png')
            plt.savefig(figure_path)
            print(f"🖼️ Histogram saved under: {figure_path}")
        plt.close()

    @staticmethod
    def get_outlier_outlier_mask(reference_image: np.ndarray,
                                 multiplier: float = 1.5) -> np.ndarray:
        valid = reference_image[~np.isnan(reference_image)]
        q1 = np.percentile(valid, 25)
        q3 = np.percentile(valid, 75)
        iqr = q3 - q1
        upper_limit = q3 + multiplier * iqr
        return reference_image > upper_limit

    @staticmethod
    def extract_images_patches(rgb_image,
                            reference_image,
                            patch_size: int = 518,
                            stride_ratio: float = 0.5,
                            **args)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract patches from the input image using a sliding window approach.
        Parameters:
            rgb_image (numpy.ndarray): Input RGB image.
            reference_image (numpy.ndarray): Reference image (e.g., CHM).
            patch_size (int): Length of the patches to be extracted.
            stride_ratio (float): Ratio of the stride length to the patch length. 0.5 means 50% overlap, 0.25 means 75% overlap, 0.15 means 85% overlap.
            args (dict): Additional arguments such as binary_mask_image. Such as: binary_mask_image which is a binary mask image obtained from the NIR image. 
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Patches of RGB image, reference image, and binary mask image.
        """
        channel_n = 3
        train_step = int(patch_size * stride_ratio)

        print(f'Patch size: {patch_size}, Step: {train_step} → Overlap: {100 - stride_ratio * 100:.0f}%')
        
        rgb_image_patch = view_as_windows(rgb_image, (patch_size, patch_size, channel_n), step=train_step)
        reference_image_patch = view_as_windows(reference_image, (patch_size, patch_size), step=train_step)
        binary_mask_image = args.get('binary_mask_image', None)
        if binary_mask_image is not None:
            print(f'Image shape: {rgb_image.shape}, CHM: {reference_image.shape}, Mask: {binary_mask_image.shape}')
            mask_image_patch = view_as_windows(binary_mask_image, (patch_size, patch_size), step=train_step)
        else:
            mask_image_patch = None
        rgb_image_patch = rgb_image_patch.reshape(-1, patch_size, patch_size, channel_n)
        reference_image_patch = reference_image_patch.reshape(-1, patch_size, patch_size)
        if mask_image_patch is not None:
            mask_image_patch = mask_image_patch.reshape(-1, patch_size, patch_size)
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
                        threshold: float = 0.2,
                        figure_size: tuple[int, int] = (8, 8),
                        path_to_save: str = None) -> np.ndarray:
        """
        Get a binary mask from the NIR image using a threshold.
        
        Parameters:
            nir_image (numpy.ndarray): NIR image.
            rgb_image (numpy.ndarray): RGB image.
            threshold (float): Threshold for binary mask.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Binary mask and NDVI.
        """
        red_channel = rgb_image[:, :, 0]
        # if np.max(red_channel) > 1:
        #     red_channel = red_channel / 255.0
        # if np.max(nir_image) > 1:
        #     nir_image = nir_image / 255.0
        ndvi  = (nir_image - red_channel) / (nir_image + red_channel + 1e-8)
        binary_mask = np.where(ndvi  > threshold, 1, 0).astype(np.float32)

        if path_to_save:
            Utils.visualize_dataset(binary_mask, figure_size=figure_size, path_to_save=path_to_save, title='Binary Mask', cmap='gray', name='NDVI_Binary_mask.png')
            plt.figure(figsize=figure_size)
            plt.imshow(rgb_image / 255.0 if np.max(rgb_image) > 1 else rgb_image)
            plt.imshow(binary_mask, alpha=0.5, cmap='gray')
            plt.axis('off')
            plt.title('Binary Mask Overlay')
            plt.savefig(os.path.join(path_to_save, 'binary_mask_overlay.png'))
            plt.close()
        return binary_mask, ndvi
    
    @staticmethod
    def get_min_max_depth(binary_mask: np.ndarray,
                          chm_image: np.ndarray) -> tuple[float, float]:
        """
        Calculate the minimum and maximum depth from the NDVI image, binary mask image, and reference image.
        Parameters:
            binary_mask_image (numpy.ndarray): Binary mask image obtained from the NIR image.
            reference_image (numpy.ndarray): Reference image (e.g., CHM).

        Returns:

            Tuple[float, float]: Minimum and maximum depth.
        """
        #norma_chm_image = chm_image/255 if np.max(chm_image) > 1 else chm_image

        depth_map = binary_mask * chm_image
        norm_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        min_norm_depth_map = np.min(norm_depth_map)
        max_norm_depth_map = np.max(norm_depth_map)
        min_depth_map = np.min(depth_map)
        max_depth_map = np.max(depth_map)
        return min_norm_depth_map, max_norm_depth_map, min_depth_map, max_depth_map
    
    @staticmethod
    def _side_handler(image: np.ndarray, which_half: str = 'top', axis: int = 0) -> np.ndarray:
        """
        Generic side handler for splitting an image along a specified axis.
        """
        split_index = image.shape[axis] // 2
        slicer = [slice(None)] * image.ndim
        if which_half == 'top':
            slicer[axis] = slice(0, split_index)
        elif which_half == 'bottom':
            slicer[axis] = slice(split_index, None)
        else:
            raise ValueError("Invalid 'which_half'. Use 'top' or 'bottom'.")
        return image[tuple(slicer)]

    @staticmethod
    def split_images(rgb_image: np.ndarray,
                    chm_image: np.ndarray,
                    ndvi_mask: np.ndarray,
                    binary_mask: np.ndarray,
                    split_type: str = 'horizontal',
                    path_to_save: str = None) -> dict:
        """
        Split the RGB image into two parts based on the split type.
        
        Parameters:
            rgb_image (numpy.ndarray): Input RGB image.
            chm_image (numpy.ndarray): Reference image (e.g., CHM).
            ndvi_mask (numpy.ndarray): NDVI image.
            binary_mask (numpy.ndarray): Binary mask image obtained from the NIR image.
            split_type (str): Type of split ('horizontal' or 'vertical').
        
        Returns:
            dict: Metadata for the top and bottom halves of the split images.
        """
        for img in [rgb_image, ndvi_mask, binary_mask, chm_image]:
            if not isinstance(img, np.ndarray):
                raise ValueError("All images must be numpy arrays.")
        
        if not (rgb_image.shape[:2] == ndvi_mask.shape[:2] == binary_mask.shape[:2] == chm_image.shape[:2]):
            raise ValueError("All images must have the same spatial dimensions.")
        
        if split_type == 'horizontal':
            axis = 0 # Split along the height
        elif split_type == 'vertical':
            axis = 1 # Split along the width
        else:
            raise ValueError("Invalid split type. Use 'horizontal' or 'vertical'.")
        
        metadata = {}

        for half in ['top', 'bottom']:
            rgb_half = Utils._side_handler(rgb_image, half, axis)
            ndvi_half = Utils._side_handler(ndvi_mask, half, axis)
            binary_mask_half = Utils._side_handler(binary_mask, half, axis)
            chm_half = Utils._side_handler(chm_image, half, axis)

            min_depth, max_depth = Utils.get_min_max_depth(binary_mask_half, chm_half)
            metadata[half] = {
                'rgb_image': rgb_half,
                'chm_image': chm_half,
                'ndvi_mask': ndvi_half,
                'binary_mask': binary_mask_half,
                'min_depth': min_depth,
                'max_depth': max_depth}

            if path_to_save:
                Utils.visualize_dataset(rgb_half, 
                                        path_to_save=path_to_save, 
                                        title=f'RGB Image - {half.capitalize()} Half')
                Utils.visualize_dataset(ndvi_half,
                                        path_to_save=path_to_save, 
                                        title=f'NDVI Image - {half.capitalize()} Half')
                Utils.visualize_dataset(binary_mask_half,
                                        path_to_save=path_to_save, 
                                        title=f'Binary Mask - {half.capitalize()} Half',
                                        cmap='gray')
                Utils.visualize_dataset(chm_half, 
                                        path_to_save=path_to_save, 
                                        title=f'Reference Image (CHM) - {half.capitalize()} Half')
                
                #plt.figure(figsize=figure_size)
                _, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(rgb_half/255.0)
                axs[0].set_title(f'RGB Image - {half.capitalize()} Half')
                axs[0].axis('off')
                axs[1].imshow(ndvi_half, cmap='gray')
                axs[1].set_title(f'NDVI Mask - {half.capitalize()} Half')
                axs[1].axis('off')
                axs[2].imshow(binary_mask_half, cmap='gray')
                axs[2].set_title(f'Binary Mask - {half.capitalize()} Half')
                axs[2].axis('off')
                axs[3].imshow(chm_half/255.0, cmap='plasma')
                axs[3].set_title(f'CHM Image (CHM) - {half.capitalize()} Half')
                axs[3].axis('off')
                # axs[4].imshow(binary_mask_half * (chm_half / 255 if np.max(chm_half) > 1 else chm_half), cmap='viridis', alpha=0.5)
                # axs[4].set_title(f'Masked CHM Image - {half.capitalize()} Half')
                # axs[4].axis('off')
                plt.suptitle(f'Min Depth: {min_depth:.2f}, Max Depth: {max_depth:.2f}', fontsize=16)
                plt.subplots_adjust(top=0.85)
                plt.tight_layout()
                fig_path = os.path.join(path_to_save, f'{half}_half_images.png')
                plt.savefig(fig_path)
                print(f"🖼️ {half.capitalize()} half images saved under: {fig_path}")
                plt.close()
  
        return metadata