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
    def load_dataset_from_tif(rbg_img_folder_path: str,
                depth_img_folder_path: str,
                binary_mask_folder: str = None):
        """
        Read TIFF files (RGB, depth, and optionally mask) from the specified folders.

        Args:
            rbg_img_folder_path (str): Folder path for RGB TIFF files.
            depth_img_folder_path (str): Folder path for depth TIFF files.
            binary_mask_folder (str, optional): Folder path for binary mask TIFF files.
            NB (⚠️): If binary_mask_folder is provided, it must contain the same number of files as the RGB and depth folders.
        Yields:
            Tuple: (rgb, depth) if mask not provided, else (rgb, depth, mask)
        """
        rgb_files = sorted(glob(os.path.join(rbg_img_folder_path, '*.tif')))
        depth_files = sorted(glob(os.path.join(depth_img_folder_path, '*.tif')))
        mask_files = sorted(glob(os.path.join(binary_mask_folder, '*.tif'))) if binary_mask_folder else None

        if not rgb_files or not depth_files:
            raise ValueError("No TIFF files found in the specified folders.")

        if mask_files and (len(rgb_files) != len(depth_files) or len(rgb_files) != len(mask_files)):
            raise ValueError("Mismatch in number of RGB, depth, and mask files.")

        for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
            rgb = np.transpose(rasterio.open(rgb_file).read(), (1, 2, 0))
            depth = rasterio.open(depth_file).read(1)
            print(f"ℹ️ Reading RGB image from {rgb_file} with shape {rgb.shape}")
            print(f"ℹ️ Reading depth image from {depth_file} with shape {depth.shape}")

            if rgb.shape[:2] != depth.shape:
                raise ValueError(f"Shape mismatch at index {i}: RGB {rgb.shape} vs Depth {depth.shape}")

            if mask_files:
                mask_file = mask_files[i]
                mask = rasterio.open(mask_file).read(1)
                print(f"ℹ️ Reading Mask image from {mask_file} with shape {mask.shape}")
                if rgb.shape[:2] != mask.shape:
                    raise ValueError(f"Shape mismatch at index {i}: Mask {mask.shape} does not match RGB {rgb.shape}")
                yield rgb, depth, mask
            else:
                mask = None
                yield rgb, depth, mask

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
            return T.Compose([T.ToTensor()])

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
                has_mask = False
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
        _, axs = plt.subplots(nrows=nrows * (4 if has_mask else 2), ncols=ncols, figsize=(ncols * 4, fig_height))
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

        print(f'➡️ Patch size: {patch_size}, Step: {train_step} → Overlap: {100 - stride_ratio * 100:.0f}%')
        
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
    def get_min_max_depth(chm_image: np.ndarray,
                        binary_mask: np.ndarray = None) -> tuple[float, float]:
        """
        Calculate the minimum and maximum depth from the NDVI image, binary mask image, and reference image.
        Parameters:
            binary_mask_image (numpy.ndarray): Binary mask image obtained from the NIR image.
            chm_image (numpy.ndarray): Reference image (e.g., CHM).

        Returns:

            Tuple[float, float]: Minimum and maximum depth.
        """

        if binary_mask is not None:
            depth_map = binary_mask * chm_image
            valid_pixels = depth_map[depth_map > 1e-8]

            if valid_pixels.size == 0:
                raise ValueError("No valid pixels found in CHM after applying the binary mask.")

            min_depth = np.min(valid_pixels)
            max_depth = np.max(valid_pixels)

            norm_depth_map = (valid_pixels - min_depth) / (max_depth - min_depth + 1e-8)

            min_norm_depth = float(np.min(norm_depth_map))
            max_norm_depth = float(np.max(norm_depth_map))
            return min_norm_depth, max_norm_depth, float(min_depth), float(max_depth)

        else:
            min_depth = np.min(chm_image)
            max_depth = np.max(chm_image)

            norm_depth_map = (chm_image - min_depth) / (max_depth - min_depth + 1e-8)

            min_norm_depth = float(np.min(norm_depth_map))
            max_norm_depth = float(np.max(norm_depth_map))

            return min_norm_depth, max_norm_depth, float(min_depth), float(max_depth)        

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
                    ndvi_mask: np.ndarray = None,
                    binary_mask: np.ndarray = None,
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

        images = {'RGB': rgb_image, 'CHM': chm_image, 'NDVI': ndvi_mask, 'Binary Mask': binary_mask}

        for name, img in images.items():
            if img is not None and not isinstance(img, np.ndarray):
                raise ValueError(f"{name} must be a numpy array.")

        shapes = [img.shape[:2] for img in images.values() if img is not None]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("All input images must have the same spatial dimensions.")

        axis = {'horizontal': 0, 'vertical': 1}.get(split_type)
        if axis is None:
            raise ValueError("Invalid split type. Use 'horizontal' or 'vertical'.")

        metadata = {}
        for half in ['top', 'bottom']:
            rgb_half = Utils._side_handler(rgb_image, half, axis)
            chm_half = Utils._side_handler(chm_image, half, axis)
            ndvi_half = Utils._side_handler(ndvi_mask, half, axis) if ndvi_mask is not None else None
            binary_half = Utils._side_handler(binary_mask, half, axis) if binary_mask is not None else None

            #norm_min_depth, norm_max_depth, real_min_depth, real_max_depth = Utils.get_min_max_depth(binary_half, chm_half)

            metadata[half] = {
                'rgb_image': rgb_half,
                'chm_image': chm_half,
                'ndvi_mask': ndvi_half,
                'binary_mask': binary_half}
            # ,
            #     'min_depth': norm_min_depth,
            #     'max_depth': norm_max_depth,
            #     'real_min_depth': real_min_depth,
            #     'real_max_depth': real_max_depth}

            if path_to_save:
                Utils.visualize_dataset(rgb_half, 
                                        path_to_save=path_to_save, 
                                        title=f'RGB Image - {half.capitalize()} Half')

                if ndvi_half is not None:
                    Utils.visualize_dataset(ndvi_half,
                                        path_to_save=path_to_save, 
                                        title=f'NDVI Image - {half.capitalize()} Half') if ndvi_mask is not None else None

                if binary_half is not None:
                    Utils.visualize_dataset(binary_half,
                                        path_to_save=path_to_save, 
                                        title=f'Binary Mask - {half.capitalize()} Half',
                                        cmap='gray') if binary_mask is not None else None

                Utils.visualize_dataset(chm_half, 
                                        path_to_save=path_to_save, 
                                        title=f'Reference Image (CHM) - {half.capitalize()} Half')

                n_cols = 2 + sum([ndvi_half is not None, binary_half is not None])
                _, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

                idx = 0
                axs[idx].imshow(rgb_half / 255.0)
                axs[idx].set_title(f'RGB - {half.capitalize()}')
                axs[idx].axis('off')
                idx += 1

                if ndvi_half is not None:
                    axs[idx].imshow(ndvi_half, cmap='gray')
                    axs[idx].set_title(f'NDVI - {half.capitalize()}')
                    axs[idx].axis('off')
                    idx += 1

                if binary_half is not None:
                    axs[idx].imshow(binary_half, cmap='gray')
                    axs[idx].set_title(f'Binary Mask - {half.capitalize()}')
                    axs[idx].axis('off')
                    idx += 1

                axs[idx].imshow(chm_half / 255.0, cmap='plasma')
                axs[idx].set_title(f'CHM - {half.capitalize()}')
                axs[idx].axis('off')

                plt.suptitle(f'{half.capitalize()} Half | Min Depth: {chm_half.min():.2f}, Max Depth: {chm_half.max():.2f}')
                plt.tight_layout()
                fig_path = os.path.join(path_to_save, f'{half}_half_images.png')
                plt.savefig(fig_path)
                print(f"🖼️ {half.capitalize()} half images saved at: {fig_path}")
                plt.close()

        return metadata

    @staticmethod
    def save_inference_plot(rgb, depth_gt, depth_pred, save_path, mask=None, title=None):

        image = np.transpose(rgb, (1, 2, 0))
        image_min, image_max = image.min(), image.max()
        image = (image - image_min) / (image_max - image_min) if image_max > image_min else torch.zeros_like(image)

        ncols = 4 if mask is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        axes[0].imshow(image)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')
        vmin=0
        vmax = max(depth_gt.max(), depth_pred.max())
        im1 = axes[1].imshow(depth_gt, cmap="Spectral_r", vmin=vmin, vmax=vmax)
        axes[1].set_title("Depth Ground Truth")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(depth_pred, cmap="Spectral_r",vmin=vmin, vmax=vmax)
        axes[2].set_title("Depth Prediction")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])

        if mask is not None:
            axes[3].imshow(mask, cmap="gray")
            axes[3].set_title("Mask")
            axes[3].axis('off')

        if title:
            fig.suptitle(title, fontsize=16)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot(image, 
            depth, 
            prediction=None, 
            mask=None,
            depth_bounds=None,
            source_id=None, show_titles=True):
        """
        Plots RGB image, depth map, NDVI mask, and optionally the prediction and difference.
        Also rescales using the real depth values based on source_id.

        Args:
            image (Tensor): RGB image (3, H, W)
            depth (Tensor): Ground truth CHM normalized (1, H, W)
            prediction (Tensor): Optional predicted depth normalized (1, H, W)
            mask (Tensor): Optional NDVI binary mask (1, H, W)
            source_id (int): Index of the original image to retrieve real min/max
            show_titles (bool): Whether to show plot titles
        Returns:
            fig, scatter_fig: matplotlib figures
        """

        # Normalize image for display
        image = image.float()
        image_min, image_max = image.min(), image.max()
        image = (image - image_min) / (image_max - image_min) if image_max > image_min else torch.zeros_like(image)

        depth = depth.float()
        if mask is not None:
            mask = mask.float()

        # === Get real depth range from source_id ===
        if source_id is not None:
            real_min, real_max = depth_bounds[source_id][2], depth_bounds[source_id][3]
            # Rescale depth and prediction
            depth = depth * (real_max - real_min) + real_min
            if prediction is not None:
                prediction = prediction * (real_max - real_min) + real_min
            vmax = real_max
        else:
            vmax = 1  # fallback if no source_id

        # === Number of columns ===
        ncols = 2  # RGB + Depth
        if mask is not None:
            ncols += 2
        if prediction is not None:
            ncols += 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        idx = 0
        axs[idx].imshow(image.permute(1, 2, 0).cpu())
        axs[idx].axis("off")
        if show_titles:
            axs[idx].set_title("RGB Image")
        idx += 1

        im_depth = axs[idx].imshow(depth.squeeze().cpu(), cmap="Spectral_r", vmin=0, vmax=vmax)
        axs[idx].axis("off")
        if show_titles:
            axs[idx].set_title("Depth Map")
        fig.colorbar(im_depth, ax=axs[idx])
        idx += 1

        if mask is not None:
            im_mask = axs[idx].imshow(mask.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)

            axs[idx].axis("off")
            if show_titles:
                axs[idx].set_title("Binary Mask")
            fig.colorbar(im_mask, ax=axs[idx])
            idx += 1

            depth_masked = depth * mask
            im_depth_masked = axs[idx].imshow(depth_masked.squeeze().cpu(), cmap="Spectral_r", vmin=0, vmax=vmax)
            axs[idx].axis("off")
            if show_titles:
                axs[idx].set_title("Depth * Mask")
            fig.colorbar(im_depth_masked, ax=axs[idx])
            idx += 1

        if prediction is not None:
            pred_vis = prediction * mask if mask is not None else prediction

            im_pred = axs[idx].imshow(pred_vis.squeeze().cpu(), cmap="Spectral_r", vmin=0, vmax=vmax)
            axs[idx].axis("off")
            if show_titles:
                axs[idx].set_title("Prediction")
            fig.colorbar(im_pred, ax=axs[idx])
            idx += 1

            diff = torch.abs(prediction - depth).squeeze().cpu()
            im_diff = axs[idx].imshow(diff, cmap="viridis", vmin=0, vmax=(vmax - 0))
            axs[idx].axis("off")
            if show_titles:
                axs[idx].set_title("Difference |Pred - CHM|")
            fig.colorbar(im_diff, ax=axs[idx])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)

        scatter_fig = None
        if prediction is not None:
            scatter_fig, ax = plt.subplots(figsize=(6, 6))
            depth_np = depth.flatten().cpu().numpy()
            pred_np = prediction.flatten().cpu().numpy()

            if mask is not None:
                mask_np = mask.flatten().cpu().numpy()
                valid = mask_np > 0
                depth_np = depth_np[valid]
                pred_np = pred_np[valid]

            hb = ax.hexbin(depth_np, pred_np, gridsize=50, cmap='plasma', bins='log')
            ax.plot([0, vmax], [0, vmax], color='red', linestyle='--', linewidth=2)
            ax.set_xlabel("Ground Truth Depth")
            ax.set_ylabel("Predicted Depth")
            ax.set_title("2D Density: Prediction vs Ground Truth")
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            scatter_fig.colorbar(hb, ax=ax, label='log(count)')
            ax.grid(True)
            plt.tight_layout()

        return fig, scatter_fig

    @staticmethod
    def get_reference_mean_std_dual_band(reference, predicted, max_bin_value=35, output_path="reference_std_bands.png"):
        """
        Plots the mean of reference values per predicted value bin, with two spread bands:
        one for ±1×std and another for ±2×std.
        
        Parameters:
        - reference: array-like ground truth values
        - predicted: array-like predicted values
        - max_bin_value: max predicted value to bin
        - output_path: path to save the plot
        """
        
        # reference = np.array(reference)
        # predicted = np.array(predicted)
        assert reference.shape == predicted.shape, "reference and predicted must have the same shape"

        bins = np.arange(0, max_bin_value + 1, 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(predicted, bins) - 1

        means = []
        std1_lows = []
        std1_highs = []
        std2_lows = []
        std2_highs = []
        centers = []

        for i in range(len(bins) - 1):
            idx = np.where(bin_indices == i)[0]
            if len(idx) == 0:
                continue

            refs = reference[idx]
            mean = np.mean(refs)
            # print("Getting errors...")
            # errors = np.abs(reference[idx] - predicted[idx])
            # mean = np.mean(errors)

            std = np.std(refs, ddof=1)
            # std = np.std(errors, ddof=1)

            means.append(mean)
            std1_lows.append(mean - std)
            std1_highs.append(mean + std)
            std2_lows.append(mean - 2 * std)
            std2_highs.append(mean + 2 * std)
            centers.append(bin_centers[i])

        Utils.plot_reference_mean_std_dual_band(centers, means, std1_lows, std1_highs,
            std2_lows, std2_highs, output_path)

        return centers, means, std1_lows, std1_highs, std2_lows, std2_highs, output_path

    @staticmethod
    def plot_reference_mean_std_dual_band(centers, means, std1_lows, std1_highs,
        std2_lows, std2_highs, output_path):
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(centers, means, marker='o', label='Mean Reference', color='black')
        plt.fill_between(centers, std2_lows, std2_highs, color='orange', alpha=0.25, label='±2×Std')
        plt.fill_between(centers, std1_lows, std1_highs, color='blue', alpha=0.4, label='±1×Std')

        plt.xlabel("Predicted Value Bin Center")
        plt.ylabel("Reference Value")
        plt.title("Reference Mean with ±1×Std and ±2×Std Bands by Predicted Value")
        plt.grid(True, axis='y')
        plt.legend(loc='upper left')
        plt.tight_layout()
        # plt.xlim(bins[0] - 1, bins[-1] + 1)
        # plt.xlim([0,20])
        # plt.ylim([-4,20])
        plt.xlim([0,35])
        plt.ylim([-1,25])

        plt.savefig(output_path)
        plt.close()
        print(f"Dual std band plot saved to {output_path}")