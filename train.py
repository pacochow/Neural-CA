from src.standard_ca import *
from src.train_utils import train
from src.grid import *
from helpers.helpers import * 
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Get target image
target_emoji = "🦎" #@param {type:"string"}
img_size = 40
grid_size = 40
img = load_emoji(target_emoji, img_size)
target_img = pad_image(img, grid_size)
# imshow(zoom(to_rgb(target_img), 2), fmt='png')

# Parameters
num_channels = 16
fire_rate = 0.5
n_epochs = 8000

# Initialise model and grid
torch.manual_seed(0)
np.random.seed(0)
model = Standard_CA(target_img, grid_size, num_channels, fire_rate)

grid = Grid(grid_size, num_channels)

# Train model
train(model, grid, n_epochs, batch_size = 8, pool_size = 1024, regenerate = True)

# Save model
torch.save(model, "./model_params/model_3.pt")