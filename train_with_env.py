from src.env_ca import *
from src.train_utils import train
from src.grid import *
from helpers.helpers import * 
import numpy as np
import torch

# Get target image
target_emoji = "🦎" #@param {type:"string"}
img_size = 40
grid_size = 40
img = load_emoji(target_emoji, img_size)
target_img = pad_image(img, grid_size)

# Parameters
model_channels = 16
env_channels = 0
fire_rate = 0.5
n_epochs = 8000


# Initialise model and grid
torch.manual_seed(0)
np.random.seed(0)
model = Env_CA(target_img, grid_size, model_channels, env_channels, fire_rate)

grid = Grid(grid_size, model_channels)

# Initialise names and environment
if env_channels == 0:
    model_name = f"standard_{model_channels}"
    env = None
else:
    model_name = f"env_{model_channels}_{env_channels}"
    env = grid.init_env(env_channels)
    # env = grid.add_env(env, "linear", channel = 0)
    env = grid.add_env(env, "circle", channel = 0)

# Train model
model_losses = train(model, grid, n_epochs, model_name = model_name, batch_size = 8, pool_size = 1024, regenerate = True, env = env)




# Parameters
model_channels = 16
env_channels = 1
fire_rate = 0.5
n_epochs = 8000


# Initialise model and grid
torch.manual_seed(0)
np.random.seed(0)
model = Env_CA(target_img, grid_size, model_channels, env_channels, fire_rate)

grid = Grid(grid_size, model_channels)

# Initialise names and environment
if env_channels == 0:
    model_name = f"standard_{model_channels}"
    env = None
else:
    model_name = f"env_{model_channels}_{env_channels}"
    env = grid.init_env(env_channels)
    # env = grid.add_env(env, "linear", channel = 0)
    env = grid.add_env(env, "circle", channel = 0)

# Train model
model_losses = train(model, grid, n_epochs, model_name = model_name, batch_size = 8, pool_size = 1024, regenerate = True, env = env)
