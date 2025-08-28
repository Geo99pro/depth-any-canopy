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

class ProcessData:
    def __init__(self,
                 image_path: str,
                 chm_path: str,
                 split_ratio: float = 0.8,
                 mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: tuple[float, float, float] = (0.229, 0.224, 0.225),
                 batch_size: int = 32,
                 num_workers: int = 4,
                 patch_size: int = 256,
                 overlap_ratio: float = 0.1,
                 input_transform: Callable = None,
                 target_transform: Callable = None,
                 **args):
        
        self.image_path = image_path
        self.chm_path = chm_path
        self.split_ratio = split_ratio
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.args = args

        
        self.image = Utils.read_tiff(self.image_path)
        self.chm_image = np.squeeze(Utils.read_tiff(self.chm_path))

        #prepare transformations
        self.input_transform = Utils.apply_transformation(mean=self.mean,
                                                        std=self.std,
                                                        is_input=True) #images
        self.target_transform = Utils.apply_transformation(mean=self.mean,
                                                        std=self.std,
                                                        is_input=False,
                                                        is_mask=False) #chm
        self.mask_transform = Utils.apply_transformation(mean=self.mean,
                                                        std=self.std,
                                                        is_input=False,
                                                        is_mask=True) #mask

        if self.image.shape[-1] == 4:
            print("🟢 NIR band found in the input image. Proceeding with RGB NIR and CHM.")
            self.rgb_image = self.image[:, :, :3]
            self.nir = self.image[:, :, 3]
        
            #get binary mask and ndvi_mask
            self.binary_mask, self.ndvi_mask = Utils.get_binary_mask(nir_image=self.nir,
                                                                    rgb_image=self.rgb_image,
                                                                    threshold=self.args.get('ndvi_thr', 0.2 ),
                                                                    path_to_save=self.args.get('output_path'))
            
            #split the image either horizontally of vertically.
            self.meta_data = Utils.split_images(rgb_image=self.rgb_image,
                                                chm_image=self.chm_image,
                                                ndvi_mask=self.ndvi_mask,
                                                binary_mask=self.binary_mask,
                                                split_type=self.args.get('split_type', 'horizontal'),
                                                path_to_save=self.args.get('output_path'))
            
            #visualize the histograms
            Utils.visualize_histogram(image=self.meta_data['top']["chm_image"],
                                    bins=100,
                                    title="Top CHM Distribution",
                                    xlabel="Depth",
                                    ylabel="Frequency",
                                    path_to_save=self.args.get("output_path"),
                                    name='Top_CHM_Distribution')
            #visualize the histograms
            Utils.visualize_histogram(image=self.meta_data['bottom']["chm_image"],
                                    bins=100,
                                    title="Bottom CHM Distribution",
                                    xlabel="Depth",
                                    ylabel="Frequency",
                                    path_to_save=self.args.get("output_path"),
                                    name='Bottom_CHM_Distribution')
            
            #extract the images from the meta_data
            self.rgb_image_top = self.meta_data['top']['rgb_image'].astype(np.uint8)
            self.rgb_image_bottom = self.meta_data['bottom']['rgb_image'].astype(np.uint8)

            self.chm_image_top = self.meta_data['top']['chm_image'].astype(np.float16)
            self.chm_image_bottom = self.meta_data['bottom']['chm_image'].astype(np.float16)
            
            self.binary_mask_top = self.meta_data['top']['binary_mask'].astype(np.uint8)
            self.binary_mask_bottom = self.meta_data['bottom']['binary_mask'].astype(np.uint8)

            self.top_min_depth, self.top_max_depth = self.meta_data['top']['min_depth'], self.meta_data['top']['max_depth'] #comes already normalized
            self.bottom_min_depth, self.bottom_max_depth = self.meta_data['bottom']['min_depth'], self.meta_data['bottom']['max_depth'] #comes already normalized

            self.rgb_image_top_patches, self.chm_image_top_patches, self.binary_mask_top_patches = Utils.extract_images_patches(rgb_image=self.rgb_image_top,
                                                                                                                                reference_image=self.chm_image_top,
                                                                                                                                binary_mask_image=self.binary_mask_top,
                                                                                                                                patch_size=self.patch_size,
                                                                                                                                stride_ratio=self.overlap_ratio)
            
            self.rgb_image_bottom_patches, self.chm_image_bottom_patches, self.binary_mask_bottom_patches = Utils.extract_images_patches(rgb_image=self.rgb_image_bottom,
                                                                                                                                reference_image=self.chm_image_bottom,
                                                                                                                                binary_mask_image=self.binary_mask_bottom,
                                                                                                                                patch_size=self.patch_size,
                                                                                                                                stride_ratio=self.overlap_ratio)


            if self.args.get('top_side_as_train'):
                #use top as training and bottom as test
                print("⬆️ Using top side image for trainig and bottom side to test.")
                self.train_loader, self.val_loader, self.test_loader = ProcessData.make_data_ready(batch_size=self.batch_size,
                                                           input_transform=self.input_transform,
                                                           target_transform=self.target_transform,
                                                           mask_transform=self.mask_transform,
                                                           side_rgb_images_patches=self.rgb_image_top_patches,
                                                           side_chm_image_patches=self.chm_image_top_patches,
                                                           side_binary_mask_patches=self.binary_mask_top_patches,
                                                           num_workers=self.num_workers,
                                                           side_test_rgb_images_patches=self.rgb_image_bottom_patches,
                                                           side_test_chm_image_patches=self.chm_image_bottom_patches,
                                                           side_test_binary_mask_patches=self.binary_mask_bottom_patches)
                
            
            elif not self.args.get('top_side_as_train'):                
                print("⬇️ Using bottom side image for trainig and top side to train.")
                self.train_loader, self.val_loader, self.test_loader = ProcessData.make_data_ready(batch_size=self.batch_size,
                                                            input_transform=self.input_transform,
                                                            target_transform=self.target_transform,
                                                            mask_transform=self.mask_transform,
                                                            side_rgb_images_patches=self.rgb_image_bottom_patches,
                                                            side_chm_image_patches=self.chm_image_bottom_patches,
                                                            side_binary_mask_patches=self.binary_mask_bottom_patches,
                                                            num_workers=self.num_workers,
                                                            side_test_rgb_images_patches=self.rgb_image_top_patches,
                                                            side_test_chm_image_patches=self.chm_image_top_patches,
                                                            side_test_binary_mask_patches=self.binary_mask_top_patches)
                
            
            if self.args.get('visualize_patches', True):
                print("Visualizing patches...")
                Utils.visualize_patches(self.train_loader,
                                        how_many_patches=4,
                                        path_to_save=self.args.get('output_path'),
                                        name='train')
                
                Utils.visualize_patches(self.val_loader,
                                        how_many_patches=4,
                                        path_to_save=self.args.get('output_path'),
                                        name='val')
                
                Utils.visualize_patches(self.test_loader,
                                        how_many_patches=4,
                                        path_to_save=self.args.get('output_path'),
                                        name='test')
        
        elif self.image.shape[-1] == 3:
            print("🟢 No NIR band found in the input image. Proceeding with RGB and CHM.")

            #visualize the histograms
            Utils.visualize_histogram(image=self.chm_image,
                                      bins=100,
                                      title="Distribution of Canopy Height Map Values",
                                      xlabel="Canopy Height",
                                      ylabel="Frequency",
                                      path_to_save=args.get('output_path'),
                                      name="chm_histogram")
            
            #get binary mask
            self.chm_image = self.chm_image.astype(np.float16)
            self.binary_mask = np.where(self.chm_image > 0, 1, 0).astype(np.uint8)
            self.min_depth, self.max_depth = Utils.get_min_max_depth(self.binary_mask, self.chm_image)

            self.rgb_image_patches, self.chm_image_patches, self.mask_image_patches= Utils.extract_images_patches(rgb_image=self.image,
                                                                                                                        reference_image=self.chm_image,
                                                                                                                        patch_size=self.patch_size,
                                                                                                                        stride_ratio=self.overlap_ratio,
                                                                                                                        binary_mask_image=self.binary_mask)
            self.train_loader, self.val_loader, _ = ProcessData.make_data_ready(batch_size=self.batch_size,
                                                                      input_transform=self.input_transform,
                                                                      target_transform=self.target_transform,
                                                                      mask_transform=self.mask_transform,
                                                                      side_rgb_images_patches=self.rgb_image_patches,
                                                                      side_chm_image_patches=self.chm_image_patches,
                                                                      side_binary_mask_patches=self.mask_image_patches,
                                                                      num_workers=self.num_workers)
            
    
    def get_loader_and_depth(self):
        """return loaders and min-max depth"""
        if self.image.shape[-1] == 4:
            if self.args.get('top_side_as_train'):
                return self.train_loader, self.val_loader, self.test_loader, self.top_min_depth, self.top_max_depth
            else:
                return self.train_loader, self.val_loader, self.test_loader, self.bottom_min_depth, self.bottom_max_depth
        else:
            return self.train_loader, self.val_loader, None, self.min_depth, self.max_depth

    @staticmethod
    def make_data_ready(batch_size, 
                        input_transform, 
                        target_transform, 
                        mask_transform, 
                        side_rgb_images_patches, 
                        side_chm_image_patches, 
                        side_binary_mask_patches, 
                        num_workers,
                        **args):

        
        all_indices = list(range(len(side_rgb_images_patches)))
        train_indices, val_indices = Utils.split_dataset_indices(all_indices=all_indices)
        print(f"🕹️ Training contains {len(train_indices)} patches and validation contains {len(val_indices)} patches.")

    
        train_image_rgb = side_rgb_images_patches[train_indices]
        val_image_rgb = side_rgb_images_patches[val_indices]

        train_chm = side_chm_image_patches[train_indices]
        val_chm = side_chm_image_patches[val_indices]

        train_mask = side_binary_mask_patches[train_indices]
        val_mask = side_binary_mask_patches[val_indices]


        train_dataset = CustomDataset(rgb_image_patches=train_image_rgb,
                                    chm_image_patches=train_chm,
                                    mask_image_patches=train_mask,
                                    rgb_image_transform=input_transform,
                                    chm_image_transform=target_transform,
                                    mask_image_transform=mask_transform)
        
        val_dataset = CustomDataset(rgb_image_patches=val_image_rgb,
                                    chm_image_patches=val_chm,
                                    mask_image_patches=val_mask,
                                    rgb_image_transform=input_transform,
                                    chm_image_transform=target_transform,
                                    mask_image_transform=mask_transform)
    
        train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        
        test_loader = None
        if args.get('side_test_rgb_images_patches') is not None and args.get('side_test_chm_image_patches') is not None and args.get('side_test_binary_mask_patches') is not None:
            test_dataset = CustomDataset(rgb_image_patches=args.get('side_test_rgb_images_patches'),
                                         chm_image_patches=args.get('side_test_chm_image_patches'),
                                         mask_image_patches=args.get('side_test_binary_mask_patches'),
                                         rgb_image_transform=input_transform,
                                         chm_image_transform=target_transform,
                                         mask_image_transform=mask_transform)
            
            test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers)
            


        return train_loader, val_loader, test_loader
        
# if __name__ == "__main__":
#     source_tiff_path = "/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/rgb_LIDAR/RGBNIR.tif"
#     reference_tiff_path = "/home/loick/DL-FOLDER/Depth-any-canopy/rgb_LIDAR/dataset/rgb_LIDAR/CHM.tif"
#     split_ratio = 0.2
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     ndvi_threshold = 0.1
#     batch_size = 32
#     num_workers = 4
#     patch_size = 518
#     overlap_ratio = 0.15
#     split_type = "horizontal"
#     top_side_as_train = True
#     visualize_patches = True
#     output_path = '/home/loick/DL-FOLDER/testing'

#     prepare_dataset = ProcessData(image_path=source_tiff_path,
#                                   chm_path=reference_tiff_path,
#                                   split_ratio=split_ratio,
#                                   mean=mean,
#                                   std=std,
#                                   batch_size=batch_size,
#                                   num_workers=num_workers,
#                                   patch_size=patch_size,
#                                   overlap_ratio=overlap_ratio,
#                                   ndvi_thr=0.1,
#                                   output_path=output_path,
#                                   visualize_patches=True,
#                                   split_type=split_type,
#                                   top_side_as_train=top_side_as_train)
    




                                                                 
                                                                 
            


# all_indices = list(range(len(self.rgb_image_bottom_patches)))
#                 train_indices, val_indices = Utils.split_dataset_indices(all_indices=all_indices)
#                 print(f"🕹️ Training contains {len(train_indices)} patches and validation contains {len(val_indices)} patches.")

            
#                 self.train_image_rgb = self.rgb_image_top_patches[train_indices]
#                 self.val_image_rgb = self.rgb_image_top_patches[val_indices]

#                 self.train_chm = self.chm_image_top_patches[train_indices]
#                 self.val_chm = self.chm_image_top_patches[val_indices]

#                 self.train_mask = self.binary_mask_top_patches[train_indices]
#                 self.val_mask = self.binary_mask_top_patches[val_indices]


            
            
#                 train_dataset = CustomDataset(rgb_image_patches=self.train_image_rgb,
#                                             chm_image_patches=self.train_chm,
#                                             mask_image_patches=self.train_mask,
#                                             rgb_image_transform=self.input_transform,
#                                             chm_image_transform=self.target_transform,
#                                             mask_image_transform=self.mask_transform)
                
#                 val_dataset = CustomDataset(rgb_image_patches=self.val_image_rgb,
#                                             chm_image_patches=self.val_chm,
#                                             rgb_image_transform=self.input_transform,
#                                             chm_image_transform=self.target_transform,
#                                             mask_image_patches=self.val_mask,
#                                             mask_image_transform=self.mask_transform)
            
#                 train_loader = DataLoader(train_dataset,
#                                         batch_size=self.batch_size,
#                                         shuffle=True,
#                                         num_workers=self.num_workers)
                
#                 val_loader = DataLoader(val_dataset,
#                                         batch_size=self.batch_size,
#                                         shuffle=False,
#                                         num_workers=self.num_workers)
            