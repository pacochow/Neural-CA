import torch   
from src.grid import *
from helpers.visualizer import create_animation

# Parameters
grid_size = 72
iterations = 800
angle = 0.0

# Load model
model = torch.load("./model_params/model_2.pt")

# Initialise grid
grid = Grid(grid_size, model.num_channels)

# Run model
state_history = grid.run(model, iterations, destroy_type = 0, destroy = False, angle = angle)

# Create animation
nSeconds = 10
filename = './media/run2.mp4'
create_animation(state_history, iterations, nSeconds, filename)