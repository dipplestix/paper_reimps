import itertools
import random
from PIL import Image
from torch.utils.data import Dataset


class SignaturePairsDataset(Dataset):
    def __init__(self, originals_dir, forgeries_dir, x_train_range, x_val_range, transform=None):
        self.originals_dir = originals_dir
        self.forgeries_dir = forgeries_dir
        self.x_train_range = x_train_range
        self.x_val_range = x_val_range
        self.transform = transform

        # Prepare the data for training and validation sets
        self.train_data = self._prepare_data(self.x_train_range)
        self.val_data = self._prepare_data(self.x_val_range)

    def _prepare_data(self, x_range):
        data = []
        for x in x_range:
            # Get all original signatures for person x
            original_signatures = [f'original_{x}_{y}.png' for y in range(1, 25)]
            # Create all possible pairs (24 choose 2 = 276)
            original_pairs = list(itertools.combinations(original_signatures, 2))

            # Sample 276 pairs of original and forged signatures
            for i in range(276):
                original = f'original_{x}_{random.randint(1, 24)}.png'
                forgery = f'forgeries_{x}_{random.randint(1, 24)}.png'
                data.append((original, forgery))

            # Add original pairs to data
            data.extend(original_pairs)

        return data

    def __len__(self):
        return len(self.train_data) + len(self.val_data)

    def __getitem__(self, idx):
        if idx < len(self.train_data):
            img1_path, img2_path = self.train_data[idx]
        else:
            img1_path, img2_path = self.val_data[idx - len(self.train_data)]

        # Determine the directories for img1 and img2
        img1_dir = self.originals_dir if "original" in img1_path else self.forgeries_dir
        img2_dir = self.originals_dir if "original" in img2_path else self.forgeries_dir

        # Determine the label y
        y = 1 if "original" in img1_path and "original" in img2_path else 0

        img1 = Image.open(f'{img1_dir}/{img1_path}').convert('L')
        img2 = Image.open(f'{img2_dir}/{img2_path}').convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # return img1, img2, img1_path, img2_path, y
        return img1, img2, y
