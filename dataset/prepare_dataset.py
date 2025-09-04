import os
import sys
import numpy as np
import shutil as sh
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './././'))
sys.path.append(project_root)

from typing import Callable
from utils.utilities import Utils
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset

class PrepareDataset:
    def __init__(self, 
                source_tiff_path: str,
                reference_tiff_path: str,
                split_ratio: float = 0.8,
                mean: tuple[float, float, float] = [0.485, 0.456, 0.406],
                std: tuple[float, float, float] = [0.229, 0.224, 0.225],
                batch_size: int = 32,
                num_workers: int = 4,
                patch_size: int = 518,
                overlap_ratio: float = 0.25,
                visualize_patches: bool = False,
                transform_rgb: Callable = None,
                transform_chm: Callable = None,
                **args):
        
        self.source_tiff_path = source_tiff_path
        self.reference_tiff_path = reference_tiff_path
        self.split_ratio = split_ratio
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.visualize_patches = visualize_patches
        self.transform_rgb = (Utils.apply_transformation(mean=self.mean,
                                                    std=self.std,
                                                    is_input=True,
                                                    is_mask=False) if transform_rgb is None else transform_rgb)
        self.transform_chm = (Utils.apply_transformation(mean=self.mean,
                                                    std=self.std,
                                                    is_input=False,
                                                    is_mask=False) if transform_chm is None else transform_chm)
        self.args = args

        self.img = Utils.read_tiff(source_tiff_path)
        self.chm = np.squeeze(Utils.read_tiff(reference_tiff_path))

        if self.img.shape[-1] == 4:
            self.transform_mask = (Utils.apply_transformation(mean=self.mean,
                                            std=self.std,
                                            is_input=False) if self.args.get('transform_mask', None) is None else self.args['transform_mask'])
            self.rgb_img = self.img[:, :, :3]
            self.nir = self.img[:, :, 3]
            
            self.binary_mask, self.ndvi_mask = Utils.get_binary_mask(nir_image=self.nir,
                                                                     rgb_image=self.rgb_img,
                                                                     threshold=self.args.get('ndvi_threshold', 0.2),
                                                                     figure_size= self.args.get('figure_size', (10, 10)),
                                                                     path_to_save=args.get('output_path'))

            self.metadata = Utils.split_images(rgb_image=self.rgb_img,
                                            ndvi_mask_image= self.ndvi_mask,
                                            binary_mask_image=self.binary_mask,
                                            reference_image=self.chm,
                                            split_type=self.args.get('split_type', 'horizontal'),
                                            path_to_save= os.path.dirname(self.source_tiff_path),
                                            figure_size=self.args.get('figure_size', (10, 10)))
            
            Utils.visualize_histogram(image=self.metadata["top"]["masked_ndvi_chm_image"],
                                      bins=100,
                                      title=self.args.get('histogram_title', 'Masked NDVI CHM Histogram top'),
                                      xlabel=self.args.get('histogram_xlabel', 'Depth Trees'),
                                      ylabel=self.args.get('histogram_ylabel', 'Frequency'),
                                      path_to_save=os.path.dirname(self.source_tiff_path),
                                      name="masked_ndvi_histogram_top")
            
            Utils.visualize_histogram(image=self.metadata["bottom"]["masked_ndvi_chm_image"],
                                      bins=100,
                                      title=self.args.get('histogram_title', 'Masked NDVI CHM Histogram bottom'),
                                      xlabel=self.args.get('histogram_xlabel', 'Depth Trees'),
                                      ylabel=self.args.get('histogram_ylabel', 'Frequency'),
                                      path_to_save=os.path.dirname(self.source_tiff_path),
                                      name="masked_ndvi_histogram_bottom")
            
            self.rgb_image_split = self.metadata["top"]['rgb_image']
            self.reference_image_split = self.metadata["top"]['reference_image']
            self.binary_mask_split = self.metadata["top"]['binary_mask_image']
            self.rgb_patches, self.reference_patches, self.mask_patches = Utils.extract_images_patches(rgb_image=self.rgb_image_split,
                                                                                                   reference_image=self.reference_image_split,
                                                                                                   binary_mask_image=self.binary_mask_split,
                                                                                                   patch_size=self.patch_size,
                                                                                                   stride_ratio=self.overlap_ratio)
            
            all_indices = list(range(len(self.rgb_patches)))
            train_indices, val_indices = Utils.split_dataset_indices(all_indices, self.split_ratio)
            print(f"🕹️ Training contains {len(train_indices)} patches and validation contains {len(val_indices)} patches.")

            self.train_rgb = self.rgb_patches[train_indices]
            self.val_rgb = self.rgb_patches[val_indices]
            self.train_reference = self.reference_patches[train_indices]
            self.val_reference = self.reference_patches[val_indices]
            self.train_mask = self.mask_patches[train_indices]
            self.val_mask = self.mask_patches[val_indices]
        
            train_dataset = CustomDataset(rgb_image_patches=self.train_rgb, 
                                          chm_image_patches=self.train_reference, 
                                          mask_image_patches=self.train_mask, 
                                          rgb_image_transform=self.transform_rgb , 
                                          chm_image_transform=self.transform_chm, 
                                          mask_image_transform=self.transform_mask)
            
            val_dataset = CustomDataset(rgb_image_patches=self.val_rgb, 
                                        chm_image_patches=self.val_reference, 
                                        mask_image_patches=self.val_mask, 
                                        rgb_image_transform=self.transform_rgb , 
                                        chm_image_transform=self.transform_chm, 
                                        mask_image_transform=self.transform_mask)
            
            self.train_dataloader = DataLoader(train_dataset, 
                                               batch_size=self.batch_size, 
                                               shuffle=True, 
                                               num_workers=self.num_workers)
            
            self.val_dataloader = DataLoader(val_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=False, 
                                            num_workers=self.num_workers)
        
            if self.visualize_patches:
                print("Visualizing patches...")
                Utils.visualize_patches(loader=self.train_dataloader,
                                        how_many_patches=4, 
                                        path_to_save=args.get('output_path'))
        
        else:
            self.transform_mask = (Utils.apply_transformation(mean=self.mean,
                                            std=self.std,
                                            is_input=False,
                                            is_mask=True) if self.args.get('transform_mask', None) is None else self.args['transform_mask'])
            self.nir = None
            self.binary_mask = None
            self.ndvi_mask = None
            
            Utils.visualize_histogram(image=self.chm,
                                      bins=100,
                                      title="Distribution of Canopy Height Map Values (before clipping)",
                                      xlabel="Canopy Height",
                                      ylabel="Frequency",
                                      path_to_save=args.get('output_path'),
                                      name="chm_histogram_before_clipping")
            
            clipped_chm = Utils.clip_outliers(reference_image=self.chm,
                                              multiplier=1.5,
                                              path_to_save=args.get('output_path'))
            
            Utils.visualize_histogram(image=clipped_chm,
                                      bins=100,
                                      title="Distribution of Canopy Height Map Values (after clipping)",
                                      xlabel="Canopy Height",
                                      ylabel="Frequency",
                                      path_to_save=args.get('output_path'),
                                      name="chm_histogram_after_clipping")
                                              
            self.binary_mask = np.where(clipped_chm > 0, 1, 0).astype(np.uint8)
            self.rgb_images_patches, self.reference_images_patches, self.mask_patches = Utils.extract_images_patches(rgb_image=self.img,
                                                                                                       reference_image=clipped_chm,
                                                                                                       patch_size=self.patch_size,
                                                                                                       stride_ratio=self.overlap_ratio,
                                                                                                       binary_mask_image=self.binary_mask)

            all_indices = list(range(len(self.rgb_images_patches)))
            train_indices, val_indices = Utils.split_dataset_indices(all_indices, self.split_ratio)
            print(f"🕹️ Training contains {len(train_indices)} patches and validation contains {len(val_indices)} patches.")
            
            self.train_rgb = self.rgb_images_patches[train_indices]
            self.train_reference = self.reference_images_patches[train_indices]

            self.val_rgb = self.rgb_images_patches[val_indices]
            self.val_reference = self.reference_images_patches[val_indices]

            self.train_mask = self.mask_patches[train_indices]
            self.val_mask = self.mask_patches[val_indices]

            train_dataset = CustomDataset(rgb_image_patches=self.train_rgb, 
                                        chm_image_patches=self.train_reference,
                                        rgb_image_transform=self.transform_rgb, 
                                        chm_image_transform=self.transform_chm, 
                                        mask_image_patches=self.train_mask,
                                        mask_image_transform= self.transform_mask)
            
            val_dataset = CustomDataset(rgb_image_patches=self.val_rgb, 
                                        chm_image_patches=self.val_reference,
                                        rgb_image_transform=self.transform_rgb, 
                                        chm_image_transform=self.transform_chm,
                                        mask_image_patches=self.val_mask,
                                        mask_image_transform=self.transform_mask)

            self.train_dataloader = DataLoader(train_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=True, 
                                            num_workers=self.num_workers)
            
            self.val_dataloader = DataLoader(val_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        num_workers=self.num_workers)
            
            if self.visualize_patches:
                print("Visualizing patches...")
                Utils.visualize_patches(loader=self.train_dataloader,
                                        how_many_patches=4, 
                                        path_to_save=args.get('output_path'))
    
    def get_train_val_loader(self):
        return self.train_dataloader, self.val_dataloader
    
    @staticmethod
    def get_test_loader(source_tiff_path: str, 
                        reference_tiff_path: str, 
                        batch_size: int = 32, 
                        num_workers: int = 4,
                        patch_size: int = 518,
                        overlap_ratio: float = 0.25,
                        mean: tuple[float, float, float] = [0.485, 0.456, 0.406], 
                        std: tuple[float, float, float] = [0.229, 0.224, 0.225],
                        transform_rgb: Callable = None,
                        transform_chm: Callable = None):
        _transform_rgb = (Utils.apply_transformation(mean=mean,
                                                    std=std,
                                                    is_input=True) if transform_rgb is None else transform_rgb)
        _transform_chm = (Utils.apply_transformation(mean=mean,
                                                    std=std,
                                                    is_input=False) if transform_chm is None else transform_chm)
        print(f"Loading test dataset from {source_tiff_path} and {reference_tiff_path} with patch size {patch_size} and overlap ratio {overlap_ratio}")
        img = Utils.read_tiff(source_tiff_path)
        chm = np.squeeze(Utils.read_tiff(reference_tiff_path))
        
        rgb_images_patches, reference_images_patches, _ = Utils.extract_images_patches(rgb_image=img,
                                                                                    reference_image=chm,
                                                                                    patch_size=patch_size,
                                                                                    stride_ratio=overlap_ratio)
        
        test_dataset = CustomDataset(rgb_image_patches=rgb_images_patches,
                                     chm_image_patches=reference_images_patches,
                                     rgb_image_transform=_transform_rgb,
                                     chm_image_transform=_transform_chm)
        
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=False, 
                                     num_workers=num_workers)
        
        return test_dataloader
    

# if __name__ == "__main__":
#     source_tiff_path = "/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/images/285E21_GeoTiff_cut.tif"
#     reference_tiff_path = "/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/images/285E21_CHM_015.tif"
#     split_ratio = 0.2
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     ndvi_threshold = 0.1
#     batch_size = 32
#     num_workers = 4
#     patch_size = 518
#     overlap_ratio = 0.15
#     visualize_patches = True
#     output_path = '/home/loick/DL-FOLDER/testing'


#     test_loader = PrepareDataset.get_test_loader(source_tiff_path=source_tiff_path,
#                                                 reference_tiff_path=reference_tiff_path,
#                                                 batch_size=batch_size,
#                                                 num_workers=num_workers,
#                                                 patch_size=patch_size,
#                                                 overlap_ratio=overlap_ratio,
#                                                 mean=mean,
#                                                 std=std)
#     for batch in test_loader:
#         if len(batch) == 2:
#             rgb_images, reference_images = batch
#             print(f"RGB Images: {rgb_images.shape}, Reference Images: {reference_images.shape}")
#         else:
#             rgb_images, reference_images, mask_images = batch
#             print(f"RGB Images: {rgb_images.shape}, Reference Images: {reference_images.shape}, Mask Images: {mask_images.shape}")
#         break
# prepare_dataset = PrepareDataset(source_tiff_path=source_tiff_path,
#                                 reference_tiff_path=reference_tiff_path,
#                                 split_ratio=split_ratio,
#                                 mean=mean,
#                                 std=std,
#                                 ndvi_threshold=ndvi_threshold,
#                                 batch_size=batch_size,
#                                 num_workers=num_workers,
#                                 patch_size=patch_size,
#                                 overlap_ratio=overlap_ratio,
#                                 visualize_patches=visualize_patches,
#                                 output_path=output_path)

# train_loader, val_loader = prepare_dataset.get_train_val_loader()
# for batch in train_loader:
#     if len (batch) == 3:
#         rgb_images, reference_images, mask_images = batch
#         print(f"RGB Images: {rgb_images.shape}, Reference Images: {reference_images.shape}, Mask Images: {mask_images.shape}")
#     else:
#         rgb_images, reference_images = batch
#         print(f"RGB Images: {rgb_images.shape}, Reference Images: {reference_images.shape}")
#     break