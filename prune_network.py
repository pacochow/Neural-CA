from helpers.helpers import *
import torch   
from src.standard_ca import *
from src.grid import *
from helpers.visualizer import create_animation


grid_size = 40
iterations = 400
angle = 0.0


# Load model
model = torch.load("./model_params/env_model.pt")
print(get_parameter_size(model))

# Prune model
pruned_model = prune_network(model, threshold = 0.05)
print(get_parameter_size(pruned_model))

# Initialise grid
grid = Grid(grid_size, pruned_model.model_channels)

# Initialise environment
env = grid.init_env(model.env_channels)
env = grid.add_env(env, "linear")

# Run model
state_history = grid.run(pruned_model, iterations, destroy_type = 0, destroy = True, angle = angle, env = env)

# Create animation
nSeconds = 10
filename = './media/pruned_run.mp4'
create_animation(state_history, iterations, nSeconds, filename)