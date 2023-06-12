import torch   
from src.standard_ca import *
from src.env_ca import *
from src.grid import Grid
from helpers.visualizer import *

# Parameters
iterations = 400
nSeconds = 10
angle = 0.0

# Load model
model_name = 'env_circle_16_1'
model = torch.load(f"./model_params/{model_name}/final_weights.pt")


# Initialise grid
grid_size = model.grid_size
grid = Grid(grid_size, model.model_channels)

# Initialise environment
env = grid.init_env(model.env_channels)
# env = grid.add_env(env, "linear", 0)
env = grid.add_env(env, "circle", 0)

# Run model
model.env = True
state_history = grid.run(model, iterations, destroy_type = 0, destroy = True, angle = angle, env = env)

# Create animation
filename = f'./model_params/{model_name}/run.mp4'
create_animation(state_history, iterations, nSeconds, filename)

# Create progress animation
states = load_progress_states(model_name, grid, iterations, grid_size, angle, env)
filename = f'./model_params/{model_name}/progress.mp4'
create_progress_animation(states, iterations, nSeconds, filename)