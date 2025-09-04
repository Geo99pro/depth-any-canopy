import os
import sys
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './././'))
sys.path.append(project_root)

from glob import glob
from utils.utilities import Utils
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset

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
        self.compute_global_chm_bounds()
        self.data = self._gather_all_patches()
        self.train_loader, self.val_loader = self._make_loaders(self.data)
        if self.args.get('visualize_patches', True):
            self._visualize_patches(self.train_loader, self.val_loader, None, np.random.randint(0, 10))

    def _get_data(self):
        """
        Load RGB and depth images from the specified folders.
        """
        rgb_images, depth_images = [], []
        for rgb, depth, _ in Utils.load_dataset_from_tif(self.images_folder_path, self.chm_folder_path):
            rgb_images.append(rgb)
            depth_images.append(depth)
        return rgb_images, depth_images

    def _prepare_transforms(self):
        input_transform = Utils.apply_transformation(mean=self.mean, std=self.std, is_input=True) #images
        target_transform = Utils.apply_transformation(is_input=False, is_mask=False) #chm
        mask_transform = Utils.apply_transformation(is_input=False, is_mask=True) #mask
        return  input_transform, target_transform, mask_transform   

    def compute_global_chm_bounds(self):
        """
        Compute the global min and max CHM values across the entire dataset.
        Store them as class attributes.
        """
        print("📉 Computing global CHM min and max ...")
        all_mins = []
        all_maxs = []

        for i, (img, chm) in enumerate(zip(self.rgb_images, self.chm_images)):
            if img.shape[-1] == 3:
                chm_min = np.min(chm)
                chm_max = np.max(chm)

                all_mins.append(chm_min)
                all_maxs.append(chm_max)

                print(f"🌳 Image {i+1}: min={chm_min:.3f}, max={chm_max:.3f}")

            elif img.shape[-1] == 4:
                metadata = Utils.split_images(
                rgb_image=img[:, :, :3],
                chm_image=chm,
                binary_mask=None,
                split_type=self.args.get('split_type', 'horizontal'),
                path_to_save=self.args.get('output_path', '')
                )
                top_chm = metadata["top"]["chm_image"]

                chm_min = np.min(top_chm)
                chm_max = np.max(top_chm)

                all_mins.append(chm_min)
                all_maxs.append(chm_max)

                print(f"🌳🪚 Image {i+1} (top split): min={chm_min:.3f}, max={chm_max:.3f}")

        self.global_chm_min = min(all_mins)
        self.global_chm_max = max(all_maxs)

        print(f"\n🌍 Global CHM Min: {self.global_chm_min:.3f}")
        print(f"🌍 Global CHM Max: {self.global_chm_max:.3f}")

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
                norma_chm_image = (chm - self.global_chm_min) / (self.global_chm_max - self.global_chm_min + 1e-8)
                binary_mask = None

                Utils.visualize_histogram(chm, bins=100, title="Real CHM Distribution", xlabel="Height", ylabel="Freq", 
                                path_to_save=self.args.get('output_path'), name=f"real_chm_histogram_{i}")
                Utils.visualize_histogram(norma_chm_image, bins=100, title="Normalized CHM Distribution", xlabel="Height", ylabel="Freq", 
                                path_to_save=self.args.get('output_path'), name=f"norma_chm_histogram_{i}")
                real_top_chm_image = chm

            elif img.shape[-1] == 4:
                rgb, nir = img[:, :, :3], img[:, :, 3]
                mask_file = self.args.get('sam_mask') if self.args.get('sam_mask').endswith('.tif') else glob(os.path.join(self.args.get('sam_mask'), '*.tif'))
                if mask_file is not None:
                    import rasterio
                    binary_mask = rasterio.open(*mask_file).read(1)
                    if rgb.shape[:2] != binary_mask.shape:
                        raise ValueError(f"Shape mismatch: Mask {binary_mask.shape} does not match RGB {rgb.shape}")
                else:
                    binary_mask = None

                metadata = Utils.split_images(
                    rgb_image=rgb,
                    chm_image=chm,
                    binary_mask=binary_mask,
                    split_type=self.args.get('split_type', 'horizontal'),
                    path_to_save= self.args.get('output_path', '')
                )

                Utils.visualize_histogram(metadata["top"]["chm_image"],
                                        bins=100, title="Top side CHM Distribution", xlabel="Height", ylabel="Freq",
                                        path_to_save=self.args.get('output_path'), name=f"top_chm_histogram_{i}")

                Utils.visualize_histogram(metadata["bottom"]["chm_image"],
                                        bins=100, title="Normalized Bottom side CHM Distribution", xlabel="Height", ylabel="Freq",
                                        path_to_save=self.args.get('output_path'), name=f"norma_bottom_chm_histogram_{i}")

                rgb = metadata["top"]["rgb_image"]
                real_top_chm_image = metadata["top"]["chm_image"]
                norma_chm_image = (real_top_chm_image - self.global_chm_min) / (self.global_chm_max - self.global_chm_min + 1e-8)
                binary_mask = None #metadata["top"]["binary_mask"]

                bottom_rgb = metadata["bottom"]["rgb_image"]
                bottom_chm = metadata["bottom"]["chm_image"]
                bottom_mask = metadata["bottom"]["binary_mask"]

                bottom_n_chm = (bottom_chm - self.global_chm_min) / (self.global_chm_max - self.global_chm_min + 1e-8)

                np.save(os.path.join(self.args.get('output_path'), f"Bottom_rgb_array.npy"), bottom_rgb)
                np.save(os.path.join(self.args.get('output_path'), f"Bottom_chm_array.npy"), bottom_chm)
                np.save(os.path.join(self.args.get('output_path'), f"Bottom_mask_sam_array.npy"), bottom_mask)
                np.save(os.path.join(self.args.get('output_path'), f"Bottom_norma_chm.npy"), bottom_n_chm)

            else:
                raise ValueError("Unsupported image format. Expected RGB or RGB+NIR.")

            patches = self._extract_patches({
                'rgb_image': rgb.astype(np.uint8),
                'chm_image': norma_chm_image.astype(np.float16),
                'binary_mask': binary_mask})

            num_patches = patches['rgb'].shape[0]
            all_rgb.append(patches['rgb'])
            all_chm.append(patches['chm'])
            all_mask.append(patches['mask'])
            all_source_ids.append(np.full((num_patches,), i))

            self.depth_bounds.append((np.min(norma_chm_image), np.max(norma_chm_image), self.global_chm_min, self.global_chm_max))

        all_rgb = np.concatenate(all_rgb, axis=0)
        all_chm = np.concatenate(all_chm, axis=0)
        all_mask = None
        all_source_ids = np.concatenate(all_source_ids, axis=0)

        return {
            'rgb': all_rgb,
            'chm': all_chm,
            'mask': all_mask,
            'source_ids': all_source_ids}

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
            patch_size=self.patch_size,
            stride_ratio=self.overlap_ratio,
            binary_mask_image=data['binary_mask'].astype(np.uint8) if data['binary_mask'] is not None else None,
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
                'mask_image_patches': data['mask'][idx] if data['mask'] is not None else None,
                'source_ids': data['source_ids'][idx],
                'rgb_image_transform': self.input_transform,
                'chm_image_transform': self.target_transform,
                'mask_image_transform': self.mask_transform
            }

        train_dataset = CustomDataset(**subset(train_idx))
        val_dataset = CustomDataset(**subset(val_idx))

        train_loader = DataLoader(train_dataset,
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers)

        val_loader = DataLoader(val_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers=self.num_workers)

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

    @staticmethod
    def get_test_loader(images_folder_path,
                        chm_folder_path,
                        sam_binary_mask_path=None,
                        mean=None,
                        std=None,
                        batch_size=None,
                        num_workers=None,
                        patch_size=None,
                        overlap_ratio=None,
                        global_chm_min=None,
                        global_chm_max=None,
                        **args):
        
        if all(s.endswith('.npy') for s in [images_folder_path, chm_folder_path, sam_binary_mask_path]):
            test_rgb_image = np.load(images_folder_path).astype(np.uint8)
            test_chm_images = np.load(chm_folder_path).astype(np.float16)
            test_mask_images = np.load(sam_binary_mask_path, allow_pickle=True).astype(np.uint8)
        else:
            for rgb_image, chm_images, mask_images in Utils.load_dataset_from_tif(images_folder_path,
                                                                                chm_folder_path,
                                                                                sam_binary_mask_path):
                test_rgb_image = rgb_image.astype(np.uint8)
                test_chm_images = chm_images.astype(np.float16)
                test_mask_images = mask_images.astype(np.uint8) if mask_images is not None else None

        print(f"🖼️ RGB image shape: {test_rgb_image.shape}")
        print(f"🌳 CHM image shape: {test_chm_images.shape}")
        if test_mask_images is not None:
            print(f"🗺️ Mask image shape: {test_mask_images.shape}")

        Utils.visualize_histogram(test_chm_images, bins=100, title="Test CHM Distribution",
                                xlabel="Height", ylabel="Freq",
                                path_to_save=args.get('output_path'), name="test_chm_histogram")

        # ✅ Normalisation avec les bornes du train
        if global_chm_min is None or global_chm_max is None:
            raise ValueError("global_chm_min and global_chm_max must be provided for proper normalization.")

        if np.max(test_chm_images)>1:
            test_chm_images = (test_chm_images - global_chm_min) / (global_chm_max - global_chm_min + 1e-8)
        else:
            print(f"🌳 CHM image already normmalized.")

        input_transform = Utils.apply_transformation(mean=mean, std=std, is_input=True)
        target_transform = Utils.apply_transformation(is_input=False)
        mask_transform = Utils.apply_transformation(is_input=False, is_mask=True)

        rgb_image = test_rgb_image

        rgb_patches, chm_patches, mask_patches = Utils.extract_images_patches(
            rgb_image=rgb_image,
            reference_image=test_chm_images,
            binary_mask_image=test_mask_images,
            patch_size=patch_size,
            stride_ratio=overlap_ratio)

        print(f"🟢 Test patches extracted: {rgb_patches.shape[0]}")

        source_ids = np.zeros((rgb_patches.shape[0],), dtype=int)

        test_dataset = CustomDataset(rgb_image_patches=rgb_patches,
                                    chm_image_patches=chm_patches,
                                    mask_image_patches=mask_patches,
                                    source_ids=source_ids,
                                    rgb_image_transform=input_transform,
                                    chm_image_transform=target_transform,
                                    mask_image_transform=mask_transform)

        test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

        if args.get('visualize_patches'):
            Utils.visualize_patches(test_loader,
                                    how_many_patches=4,
                                    path_to_save=args.get('output_path'),
                                    name="test_patches")

        min_depth = np.min(test_chm_images)
        max_depth = np.max(test_chm_images)
        real_min = global_chm_min
        real_max = global_chm_max

        print(f"🟢 Depth bounds - Normalized: ({min_depth:.3f}, {max_depth:.3f}) | Real: ({real_min:.3f}, {real_max:.3f})")

        return test_loader, (min_depth, max_depth, real_min, real_max)


# if __name__ == "__main__":
#     # Example usage
#     process_data_instance = ProcessData(
#         images_folder_path="/nethome/projetos30/busca_semantica/buscaict/BigOil/users/loick.geoffrey/depth-any-canopy/dataset/rbg_images",
#         chm_folder_path="/nethome/projetos30/busca_semantica/buscaict/BigOil/users/loick.geoffrey/depth-any-canopy/dataset/targets",
#         split_ratio=0.2,
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#         batch_size=32,
#         num_workers=4,
#         patch_size=518,
#         overlap_ratio=0.15,
#         visualize_patches=True,
#         split_type="horizontal",
#         sam_mask="/nethome/projetos30/busca_semantica/buscaict/BigOil/users/loick.geoffrey/depth-any-canopy/dataset/rgb_LIDAR/mask/eroded_sam_image.tif",
#         output_path='/nethome/projetos30/busca_semantica/buscaict/BigOil/users/loick.geoffrey/depth-any-canopy/testing'
#     )

    # train_loader, val_loader, min_depth, max_depth = process_data_instance.get_loader_and_depth()
    # for batch in train_loader:
    #     rgb_images, chm_images, masks, source_ids = batch
    #     print(f"Batch RGB shape: {rgb_images.shape}, CHM shape: {chm_images.shape}, Masks shape: {masks.shape}, Source IDs shape: {source_ids.shape}")
    #     break
    # real_min = [process_data_instance.depth_bounds[s][2] for s in source_ids]
    # real_max = [process_data_instance.depth_bounds[s][3] for s in source_ids]

    # print(f"Min Depth: {min_depth}, Max Depth: {max_depth}")
    # print(f"Real Min Depths: {real_min}, Real Max Depths: {real_max}")