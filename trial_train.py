from src.env_ca import *
from src.train_utils import train
from src.grid import *
from helpers.helpers import * 
import numpy as np
import torch

# Get target image

img_size = 40
# img = load_emoji(target_emoji, img_size)
grid_size = 50

img = np.load("./media/gecko.npy")
target_img = pad_image(img, grid_size)

# Parameters
model_channels = 16
env_channels = 2
hidden_units = 128
fire_rate = 0.5
n_epochs = 500
dynamic_env = False
env_output = False
modulate = False
angle_target = True

model_name = "trial"


# Initialise model and grid
torch.manual_seed(0)
np.random.seed(0)
model = Env_CA(target_img, grid_size, model_channels, env_channels, hidden_units, 
               fire_rate, env_output = env_output)

grid = Grid(grid_size, model_channels)

# Initialise names and environment
if env_channels == 0:
    env = None
else:
    env = grid.init_env(env_channels)
    # env = grid.add_env(env, "linear", channel = 0, angle = 45)
    # env = grid.add_env(env, "circle", channel = 0, center = (grid_size/2, grid_size/2))
    env = grid.add_env(env, "directional", channel = 0, angle = -45)

# Train model
model_losses = train(
    model, grid, n_epochs, model_name = model_name, batch_size = 8, pool_size = 1024, 
    regenerate = True, env = env, dynamic_env = dynamic_env, modulate = modulate, angle_target = angle_target)

