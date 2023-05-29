from src.neural_ca import *
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
# target_img = torch.Tensor(cv2.imread('./media/test_img.jpg'))
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
model = CAModel(target_img, grid_size, num_channels, fire_rate)
grid = Grid(grid_size, num_channels)

# Train model
model_losses = train(model, grid, n_epochs, sample_size = 8, pool_size = 1024, regenerate = False)

# Save model
torch.save(model, "./model_params/model_regenerate.pt")