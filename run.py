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
model = torch.load(f"./models/{model_name}/final_weights.pt")

# Initialise grid
grid_size = 40
grid = Grid(grid_size, model.model_channels)

# Initialise environment
env = None
env = grid.init_env(model.env_channels)
# env = grid.add_env(env, "linear", 0)
env = grid.add_env(env, "circle", 0, circle_center = (grid_size/2, grid_size/2), circle_radius = grid_size/2)

# Run model
state_history = grid.run(model, iterations, destroy_type = 0, destroy = False, angle = angle, env = env, seed = True)

# Create animation
filename = f'./models/{model_name}/diff_env_run.mp4'
create_animation(state_history, iterations, nSeconds, filename)

# Visualize seed losses at different seed positions
# filename = f"./models/{model_name}/diff_seed_losses.png"
# visualize_seed_losses(model_name, grid, iterations, filename, destroy_type = 0, destroy = True, angle = angle, env = env)

# # Create progress animation
# states = load_progress_states(model_name, grid, iterations, grid_size, angle, env)
# filename = f'./models/{model_name}/progress.mp4'
# create_progress_animation(states, iterations, nSeconds, filename)