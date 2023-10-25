from src.env_ca import *
from src.train_utils import train
from src.grid import *
from src.params import ObjectView
from helpers.helpers import * 
import numpy as np
import torch

torch.cuda.empty_cache()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
params = {
    
#   Model params
    
'grid_size': 50,
'model_channels': 16, 
'env_channels': 2,
'n_layers': 2,
'hidden_units': [100, 200],      
'fire_rate': 0.5,
        
# Training params

'num_steps': [64, 97],                  # Range of iterations during training
'pool_size': 1024,       
'batch_size': 8,
'lr': 2e-3,
'milestones': [3000, 5000, 7000],       # Milestones for learning rate scheduler
'gamma': 0.3,                           # Gamma factor for learning rate scheduler
'decay': 3e-4,                          # Weight decay for adam
'n_epochs': 8000,
'dynamic_env': False,                   # Train with dynamic environment
'dynamic_env_type': "fade out",          # Type of dynamic environment
'modulate_env': True,                   # Use alpha channel to modulate environment
'angle_target': True,                   # Train with rotation-invariance
'knockout': False,                       # Whether hidden unit is fixed
'knockout_unit': 6,                     # Hidden unit to fix
'device': device}

params = ObjectView(params)

torch.manual_seed(0)
np.random.seed(0)

# Get target image

img = np.load("./media/snake.npy")
target_img = pad_image(img, params.grid_size)

model_name = "experimental"


old_model_name = "2_hidden"
model = torch.load(f"./models/{old_model_name}/final_weights.pt", map_location = device)
model.target = torch.tensor(target_img)
model.device = device


grid = Grid(params)

# Initialise names and environment
if params.env_channels == 0:
    env = None
else:
    env = grid.init_env(params.env_channels)
    env = grid.add_env(env, "directional proportional", channel = 0, angle = -45)

# Train model
model_losses = train(model, model_name, grid, env, params)

