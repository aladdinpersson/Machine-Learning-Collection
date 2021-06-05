import numpy as np
import pandas as pd
import os
from PIL import Image


def extract_images_from_csv(csv, column, save_folder, resize=(96, 96)):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for idx, image in enumerate(csv[column]):
        image = np.array(image.split()).astype(np.uint8)
        image = image.reshape(resize[0], resize[1])
        img = Image.fromarray(image, 'L')
        img.save(save_folder+f"img_{idx}.png")


csv = pd.read_csv("test.csv")
extract_images_from_csv(csv, "Image", "data/test/")
