import os
import torch
import numpy as np
import kornia
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from PIL import Image
from osgeo import gdal
gdal.DontUseExceptions()
from typing import Callable
from torchvision import transforms as T
from skimage.util import view_as_windows
from torch.utils.data import Dataset, DataLoader
from kornia.augmentation import AugmentationSequential
from sklearn.model_selection import train_test_split


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
                plt.show()

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
                plt.show()

        elif len(image_shape) == 2:
            rgb_image = image
            if np.max(rgb_image) > 1:
                rgb_image = rgb_image / 255.0
            plt.imshow(rgb_image, cmap = args.get('cmap', 'gray'))
            plt.title(args.get('title', 'Canopy Height Map'))
            plt.axis('off')
            plt.show()

    @staticmethod
    def visualize_patches(loader: DataLoader, 
                          how_many_patches: int = 4,
                          path_to_save: str = None):
        """
        Visualize patches from a given DataLoader.

        Parameters:
            loader (DataLoader): DataLoader to visualize patches from.
            how_many_patches (int): Number of patches to visualize.
            path_to_save (str): Optional path to save the visualized patches.
        """

        images, masks = [], []
        for x, y in loader:
            for i in range(len(x)):
                images.append(Utils.denormalize(x[i]).numpy())
                masks.append(y[i].numpy())
                if len(images) == how_many_patches:
                    break
            if len(images) == how_many_patches:
                break

        ncols = min(how_many_patches, 4)
        nrows = (how_many_patches + ncols - 1) // ncols
        _, axs = plt.subplots(nrows=nrows * 2, ncols=ncols, figsize=(ncols * 4, nrows * 4))
        axs = np.array(axs).reshape(nrows * 2, ncols)

        for i in range(how_many_patches):
            row = (i // ncols) * 2
            col = i % ncols

            axs[row, col].imshow(np.clip(images[i].transpose(1, 2, 0), 0, 255)) # CHW to HWC
            axs[row, col].axis('off')
            axs[row, col].set_title('Satellite RGB Image')

            axs[row + 1, col].imshow(masks[i].squeeze(0), cmap='gray') #mask image comes with (1, H, W) shape
            axs[row + 1, col].axis('off')
            axs[row + 1, col].set_title('CHM Mask Image')

        plt.tight_layout()
        if path_to_save:
            fig_path = os.path.join(path_to_save, f'{how_many_patches}_patches_examples.png')
            plt.savefig(fig_path)
        plt.close()

    @staticmethod
    def extract_images_patches(rgb_image,
                            reference_image,
                            augmentation=None,
                            patch_len=256):
        """
        Extract patches from the input image using a sliding window approach.
        
        Parameters:
            channel (numpy.ndarray): Input image channel.
            patch_len (int): Length of the patches to be extracted.
        """
        channel_n = 3
        train_step = patch_len//2

        print(f'The shape of the input image is: {rgb_image.shape}')
        print(f'The shape of the reference image is: {reference_image.shape}')
        print(f'The patch length is: {patch_len}')
        print(f'The stride is: {train_step}')
        print(f'The number of channels is: {channel_n}')
        
        rgb_image_patch = view_as_windows(rgb_image, (patch_len, patch_len, channel_n), step=train_step)
        reference_image_patch = view_as_windows(reference_image, (patch_len, patch_len), step=train_step)

        rgb_image_patch = rgb_image_patch.reshape(-1, patch_len, patch_len, channel_n)
        reference_image_patch = reference_image_patch.reshape(-1, patch_len, patch_len)
        print(f"🤖 The number of patches obtained is: {rgb_image_patch.shape[0]}")
        return rgb_image_patch, reference_image_patch

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

class CustomDataset(Dataset):
    def __init__(self, 
                 rgb_patches, 
                 chm_patches,
                 input_transform=None,
                 target_transform=None):
        self.rgb_patches = rgb_patches
        self.chm_patches = chm_patches
        self.input_transform =  input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.rgb_patches)

    def __getitem__(self, idx):
        x = self.rgb_patches[idx]
        y = self.chm_patches[idx]
        
        if x.ndim == 3:
            x = Image.fromarray(x)
        else: 
            raise ValueError("Invalid image dimensions. Expected 3D for RGB.")
        if y.ndim == 2:
            y = Image.fromarray(y)
        else:
            raise ValueError("Invalid image dimensions. Expected 2D for CHM.")
        
        if self.input_transform:
            x = self.input_transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        return x, y

class PrepareDataset:
    def __init__(self, 
                 source_tiff_path: str, 
                 reference_tiff_path: str, 
                 split_size: float,
                 mean: tuple[float, float, float] = [0.485, 0.456, 0.406],
                 std: tuple[float, float, float] = [0.229, 0.224, 0.225],
                 resize: list[int] = [518, 518],
                 batch_size: int = 16,
                 num_workers : int = 1,
                 patch_len: int = 256,
                 visualize_patches: bool = False,
                 input_transform: Callable = None,
                 target_transform: Callable = None):
        
        self.source_tiff_path = source_tiff_path
        self.reference_tiff_path = reference_tiff_path
        self.split_size = split_size
        self.mean = mean
        self.std = std
        self.resize = resize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_len = patch_len
        self.visualize_patches = visualize_patches
        self.input_transform = (Utils.apply_transformation(mean=self.mean,
                                                        std= self.std, 
                                                        size=self.resize,
                                                        is_input=True) if input_transform is None else input_transform)

        self.target_transform = (Utils.apply_transformation(mean=self.mean,
                                                        std= self.std,
                                                        size=self.resize,
                                                        is_input=False) if target_transform is None else target_transform)

        self.rgb_image = Utils.read_tiff(self.source_tiff_path)[:, :, :3]
        self.reference_image = Utils.read_tiff(self.reference_tiff_path)

        self.rgb_patches, self.reference_patches = Utils.extract_images_patches(self.rgb_image, self.reference_image, patch_len=self.patch_len) #✅

        self.train_rgb, self.val_rgb, self.train_reference, self.val_rgb_reference = Utils.split_dataset(input=self.rgb_patches, 
                                                                                                        reference=self.reference_patches, 
                                                                                                        test_size=self.split_size)

        train_dataset = CustomDataset(self.train_rgb, self.train_reference, input_transform=self.input_transform, target_transform=self.target_transform)
        val_dataset = CustomDataset(self.val_rgb, self.val_rgb_reference, input_transform=self.input_transform, target_transform=self.target_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if self.visualize_patches:
            print("Visualizing patches...")
            Utils.visualize_patches(loader=self.train_loader, how_many_patches=8, path_to_save=os.path.dirname(self.source_tiff_path))
    
    def get_train_val_loaders(self):
        return self.train_loader, self.val_loader
     
# if __name__ == "__main__":
#     source_tiff_path = "D:/meus_codigos_doutourado/Depth-any-canopy/rgb_LIDAR/RGBNIR.tif"
#     reference_tiff_path = "D:/meus_codigos_doutourado/Depth-any-canopy/rgb_LIDAR/CHM.tif"
#     split_size = 0.2
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     resize = [518, 518]
#     batch_size = 16
#     num_workers = 1
#     patch_len = 256
#     visualize_patches = True

#     dataset_preparer = PrepareDataset(source_tiff_path=source_tiff_path, 
#                                     reference_tiff_path=reference_tiff_path, 
#                                     split_size=split_size,
#                                     mean=mean,
#                                     std=std,
#                                     resize=resize,
#                                     batch_size=batch_size,
#                                     num_workers=num_workers,
#                                     patch_len=patch_len,
#                                     visualize_patches=visualize_patches)
#     train_loader, val_loader = dataset_preparer.get_train_val_loaders()

#     for batch in train_loader:
#         print(batch[0].shape, batch[1].shape)
#         if visualize_patches:
#             Utils.visualize_patches(train_loader, 
#                                     how_many_patches=8,
#                                     path_to_save=os.path.dirname(source_tiff_path))
            
#         break