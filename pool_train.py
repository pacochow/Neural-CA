from src import neural_ca
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Get target image
target = torch.Tensor(cv2.imread('./media/test_img.jpg'))

# Initialise model and optimizer
num_channels = 16
fire_rate = 0.5
model = neural_ca.CAModel(target, num_channels, fire_rate)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Parameters
grid_size = 28
pool_epochs = 2000

# Train model
model.pool_train(pool_epochs, grid_size, optimizer, sample_size = 4, pool_size = 32)
torch.save(model, "./model_params/model.pt")

print('lol')