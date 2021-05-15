import albumentations as A
import cv2
import torch
from math import log2

from albumentations.pytorch import ToTensorV2
#from utils import seed_everything

START_TRAIN_AT_IMG_SIZE = 32
DATASET = 'FFHQ_32'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False
SAVE_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 32, 32, 16, 8, 4, 2]
CHANNELS_IMG = 3
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [50] * 100
FIXED_NOISE = torch.randn((8, Z_DIM)).to(DEVICE)
NUM_WORKERS = 6