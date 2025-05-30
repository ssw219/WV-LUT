import os
import random
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "../")  # run under the project directory


class LowLightDataProvider:
    """Data provider for low-light image enhancement training.
    
    This class handles the data loading and batching for training low-light image enhancement models.
    It provides an infinite stream of paired low-light and normal-light images.
    """
    def __init__(self, batch_size=32, num_workers=8, data_path="../data/LOL_v1", patch_size=96):
        self.data = LowLightDataset(data_path, patch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        """Build the data loader."""
        self.data_iter = iter(DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False
        ))

    def next(self):
        """Get the next batch of data.
        
        Returns:
            tuple: (low_light_images, normal_light_images) as tensors
        """
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            print(f"Starting epoch {self.epoch}")
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]


class LowLightDataset(Dataset):
    """Dataset class for low-light image enhancement.
    
    This dataset loads paired low-light and normal-light images, and provides
    random patches for training with data augmentation.
    """
    def __init__(self, data_path, patch_size, use_augmentation=True):
        super(LowLightDataset, self).__init__()
        self.patch_size = patch_size
        self.use_augmentation = use_augmentation
        self.data_path = data_path
        self.file_list = os.listdir(os.path.join(self.data_path, "high"))
        print(f"Found {len(self.file_list)} image pairs")

        # Cache high-light (normal) images
        self.high_light_cache = os.path.join(data_path, "cache_high_light.npy")
        if not os.path.exists(self.high_light_cache):
            self._cache_high_light_images()
            print(f"High-light image cache saved to: {self.high_light_cache}")
        self.high_light_images = np.load(self.high_light_cache, allow_pickle=True).item()
        print(f"High-light image cache loaded from: {self.high_light_cache}")

        # Cache low-light images
        self.low_light_cache = os.path.join(data_path, "cache_low_light.npy")
        if not os.path.exists(self.low_light_cache):
            self._cache_low_light_images()
            print(f"Low-light image cache saved to: {self.low_light_cache}")
        self.low_light_images = np.load(self.low_light_cache, allow_pickle=True).item()
        print(f"Low-light image cache loaded from: {self.low_light_cache}")

    def _cache_low_light_images(self):
        """Cache low-light images to numpy file."""
        low_light_dict = {}
        low_light_path = os.path.join(self.data_path, "low")
        for filename in self.file_list:
            low_light_dict[filename] = np.array(Image.open(os.path.join(low_light_path, filename)))
        np.save(self.low_light_cache, low_light_dict, allow_pickle=True)

    def _cache_high_light_images(self):
        """Cache high-light (normal) images to numpy file."""
        high_light_dict = {}
        high_light_path = os.path.join(self.data_path, "high")
        for filename in self.file_list:
            high_light_dict[filename] = np.array(Image.open(os.path.join(high_light_path, filename)))
        np.save(self.high_light_cache, high_light_dict, allow_pickle=True)

    def __getitem__(self, _):
        """Get a random patch from a random image pair.
        
        Returns:
            tuple: (low_light_patch, high_light_patch) as tensors
        """
        filename = random.choice(self.file_list)
        high_light = self.high_light_images[filename]
        low_light = self.low_light_images[filename]

        # Random crop
        h, w = low_light.shape[:2]
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)

        high_light = high_light[i:i + self.patch_size, j:j + self.patch_size, :]
        low_light = low_light[i:i + self.patch_size, j:j + self.patch_size, :]

        # Data augmentation
        if self.use_augmentation:
            if random.random() < 0.5:
                high_light = np.fliplr(high_light)
                low_light = np.fliplr(low_light)

            if random.random() < 0.5:
                high_light = np.flipud(high_light)
                low_light = np.flipud(low_light)

            k = random.choice([0, 1, 2, 3])
            high_light = np.rot90(high_light, k)
            low_light = np.rot90(low_light, k)

        # Normalize and convert to tensor format (C, H, W)
        high_light = np.transpose(high_light.astype(np.float32) / 255.0, [2, 0, 1])
        low_light = np.transpose(low_light.astype(np.float32) / 255.0, [2, 0, 1])

        return low_light, high_light

    def __len__(self):
        return int(sys.maxsize)


class LowLightBenchmark(Dataset):
    """Dataset class for low-light image enhancement benchmarking.
    
    This dataset loads paired low-light and normal-light images for evaluation.
    """
    def __init__(self, data_path, datasets=['val']):
        super(LowLightBenchmark, self).__init__()
        self.images = {}
        self.files = {}
        
        for dataset in datasets:
            high_light_folder = os.path.join(data_path, dataset, 'high')
            files = sorted(os.listdir(high_light_folder))
            self.files[dataset] = files

            for filename in files:
                # Load high-light (normal) image
                high_light = np.array(Image.open(
                    os.path.join(data_path, dataset, 'high', filename)))
                
                # Handle grayscale images
                if len(high_light.shape) == 2:
                    high_light = np.expand_dims(high_light, axis=2)
                    high_light = np.concatenate([high_light] * 3, axis=2)

                key = f"{dataset}_{filename[:-4]}"
                self.images[key] = high_light

                # Load low-light image
                low_light = np.array(Image.open(
                    os.path.join(data_path, dataset, 'low', filename)))
                
                # Handle grayscale images
                if len(low_light.shape) == 2:
                    low_light = np.expand_dims(low_light, axis=2)
                    low_light = np.concatenate([low_light] * 3, axis=2)

                key = f"{dataset}_{filename[:-4]}"
                self.images[key + "_low"] = low_light

                # Verify image dimensions
                assert low_light.shape[0] == high_light.shape[0]
                assert low_light.shape[1] == high_light.shape[1]
                assert low_light.shape[2] == high_light.shape[2] == 3