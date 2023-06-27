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
model_name = 'angled_env_directional_2_16_2'
model = torch.load(f"./models/{model_name}/final_weights.pt")

# Initialise grid
grid_size = 50
grid = Grid(grid_size, model.model_channels)

# Initialise environment
env = None
env = grid.init_env(model.env_channels)
# env = grid.add_env(env, "linear", 0, angle = 45)
# env = grid.add_env(env, "circle", 0, center = (grid_size/2, grid_size/2), circle_radius = model.grid_size/2)
env = grid.add_env(env, "directional", 0, angle = 45, center = (grid_size/2, grid_size/2))
# env = grid.add_env(env, 'none')
dynamic_env = False
vis_env = False
modulate = False
# model.env_output = False



# Run model
state_history, env_history = grid.run(
    model, iterations, destroy = True, angle = angle, env = env, seed = None, 
    dynamic_env = dynamic_env, dynamic_env_type='free move', modulate = modulate)

# # Create animation
# filename = f'./models/{model_name}/run.mp4'
# create_animation(state_history, env_history, iterations, nSeconds, filename, vis_env = vis_env)

# Visualize all channels
filename = f'./models/{model_name}/all_channels_run.mp4'
visualize_all_channels(state_history, iterations, nSeconds, filename, model.model_channels)

# Visualize seed losses at different seed positions
# filename = f"./models/{model_name}/diff_seed_losses.png"
# visualize_seed_losses(model_name, grid, iterations, filename, destroy = True, angle = angle, env = env)

# Create progress animation
# states, envs = load_progress_states(model_name, grid, iterations, grid_size, angle, env)
# filename = f'./models/{model_name}/progress.mp4'
# create_progress_animation(states, envs, iterations, nSeconds, filename, vis_env = True)

# Visualize parameter sizes
# filename = f'./models/{model_name}/parameter_sizes.png'
# plot_parameter_sizes(model_name, filename)


