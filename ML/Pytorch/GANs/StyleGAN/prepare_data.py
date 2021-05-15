from PIL import Image
from tqdm import tqdm
import os
from multiprocessing import Pool
root_dir = "FFHQ/images1024x1024/"
files = os.listdir(root_dir)

def resize(file, size, folder_to_save):
    image = Image.open(root_dir + file).resize((size, size), Image.LANCZOS)
    image.save(folder_to_save+file, quality=100)


if __name__ == "__main__":
    for img_size in [4, 8, 512, 1024]:
        folder_name = "FFHQ_"+str(img_size)+"/images/"
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        data = [(file, img_size, folder_name) for file in files]
        pool = Pool()
        pool.starmap(resize, data)









