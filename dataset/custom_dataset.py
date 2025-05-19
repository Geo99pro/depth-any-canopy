import os
from PIL import Image
from utils.utilities import Utils
from typing import Callable
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, 
                 rgb_patches, 
                 chm_patches,
                 mask_patches,
                 input_transform=None,
                 target_transform=None,
                 mask_transform=None):
        self.rgb_patches = rgb_patches
        self.chm_patches = chm_patches
        self.mask_patches = mask_patches
        self.input_transform =  input_transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.rgb_patches)

    def __getitem__(self, idx):
        x = self.rgb_patches[idx]
        y = self.chm_patches[idx]
        mask = self.mask_patches[idx]
        
        if x.ndim == 3:
            x = Image.fromarray(x)
        else: 
            raise ValueError("Invalid image dimensions. Expected 3D for RGB.")
        if y.ndim == 2:
            y = Image.fromarray(y)
        else:
            raise ValueError("Invalid image dimensions. Expected 2D for CHM.")
        if mask.ndim == 2:
            mask = Image.fromarray(mask)
        else:
            raise ValueError("Invalid image dimensions. Expected 2D for mask.")
        
        if self.input_transform:
            x = self.input_transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return x, y, mask

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
                 ndvi_threshold: float = 0.5,
                 patch_len: int = 256,
                 visualize_patches: bool = False,
                 input_transform: Callable = None,
                 target_transform: Callable = None,
                 mask_transform: Callable = None):
        """ 
        Args:
            source_tiff_path (str): Path to the source TIFF file.
            reference_tiff_path (str): Path to the reference TIFF file.
            split_size (float): Fraction of the dataset to be used for validation.
            mean (tuple[float, float, float]): Mean values for normalization.
            std (tuple[float, float, float]): Standard deviation values for normalization.
            resize (list[int]): Size to resize the images to.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            patch_len (int): Length of the patches to be extracted from the images.
            visualize_patches (bool): Whether to visualize the patches or not.
            input_transform (Callable): Transformations to be applied to the input images.
            target_transform (Callable): Transformations to be applied to the target images.
            mask_transform (Callable): Transformations to be applied to the masks.
        """
        
        self.source_tiff_path = source_tiff_path
        self.reference_tiff_path = reference_tiff_path
        self.split_size = split_size
        self.mean = mean
        self.std = std
        self.resize = resize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ndvi_threshold = ndvi_threshold
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

        self.mask_transform = (Utils.apply_transformation(mean=self.mean,
                                                        std= self.std,
                                                        size=self.resize,
                                                        is_input=False) if mask_transform is None else mask_transform)
        
        self.image = Utils.read_tiff(self.source_tiff_path)
        self.rgb_image = self.image[:, :, :3]
        self.nir_image = self.image[:, :, 3]
        self.reference_image = Utils.read_tiff(self.reference_tiff_path)
        self.binary_mask, _ = Utils.get_binary_mask(nir_image=self.nir_image,
                                                 rgb_image=self.rgb_image,
                                                 ndvi_threshold=self.ndvi_threshold,
                                                 figure_size=(10, 10),
                                                 path_to_save=os.path.dirname(self.source_tiff_path))

        self.rgb_patches, self.reference_patches, self.mask_patches = Utils.extract_images_patches(self.rgb_image,
                                                                                                   self.reference_image,self.binary_mask,
                                                                                                   patch_len=self.patch_len,
                                                                                                   stride_ratio=0.25) #✅
        all_indices = list(range(len(self.rgb_patches)))
        train_indices, val_indices = Utils.split_dataset_indices(all_indices, self.split_size)

        self.train_rgb = self.rgb_patches[train_indices]
        self.val_rgb = self.rgb_patches[val_indices]
        self.train_reference = self.reference_patches[train_indices]
        self.val_reference = self.reference_patches[val_indices]
        self.train_mask = self.mask_patches[train_indices]
        self.val_mask = self.mask_patches[val_indices]

        train_dataset = CustomDataset(self.train_rgb, self.train_reference, self.train_mask, input_transform=self.input_transform, target_transform=self.target_transform, mask_transform=self.mask_transform)
        val_dataset = CustomDataset(self.val_rgb, self.val_reference, self.val_mask, input_transform=self.input_transform, target_transform=self.target_transform, mask_transform=self.mask_transform)

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
#     num_workers = 0
#     ndvi_threshold = 0.4
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
#                                     ndvi_threshold=ndvi_threshold,
#                                     patch_len=patch_len,
#                                     visualize_patches=visualize_patches)
#     train_loader, val_loader = dataset_preparer.get_train_val_loaders()

#     for batch in train_loader:
#         print(batch[0].shape, batch[1].shape, batch[2].shape)
#         if visualize_patches:
#             Utils.visualize_patches(train_loader, 
#                                     how_many_patches=8,
#                                     path_to_save=os.path.dirname(source_tiff_path))
            
#         break