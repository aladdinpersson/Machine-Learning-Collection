import config
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)

    def __getitem__(self, index):
        if self.train:
            image_file, label = self.data.iloc[index]
        else:
            # if test simply return -1 for label, I do this in order to
            # re-use same dataset class for test set submission later on
            image_file, label = self.image_files[index], -1
            image_file = image_file.replace(".jpeg", "")

        image = np.array(Image.open(os.path.join(self.images_folder, image_file+".jpeg")))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, image_file


if __name__ == "__main__":
    """
    Test if everything works ok
    """
    dataset = DRDataset(
        images_folder="../train/images_resized_650/",
        path_to_csv="../train/trainLabels.csv",
        transform=config.val_transforms,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=32, num_workers=2, shuffle=True, pin_memory=True
    )

    for x, label, file in tqdm(loader):
        print(x.shape)
        print(label.shape)
        import sys
        sys.exit()