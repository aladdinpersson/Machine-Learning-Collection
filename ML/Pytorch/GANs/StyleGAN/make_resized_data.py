import os
from PIL import Image
from tqdm import tqdm

root_dir = "FFHQ/images1024x1024"

for file in tqdm(os.listdir(root_dir)):
    img = Image.open(root_dir+ "/"+file).resize((128, 128))
    img.save("FFHQ_resized/"+file)
