import torch   
from src.standard_ca import *
from src.env_ca import *
from src.grid import Grid
from helpers.visualizer import *
from src.params import ObjectView

params = {
       
# Run params
'model_channels': 16, 
'env_channels': 2,
'grid_size': 50,
'iterations': 400,                  # Number of iterations in animation
'angle': 0.0,                       # Perceiving angle
'dynamic_env': False,               # Run with moving environment
'dynamic_env_type': 'free move',    # Type of moving environment    
'modulate': False,                  # Environment modulation
'destroy': True,                    # Whether pattern is disrupted mid animation
'destroy_type': 0,                  # Type of pattern disruption
'seed': None,                       # Coordinates of seed
'vis_env': False,                   # Visualize environment in animation
'nSeconds': 10}                      # Length of animation}

params = ObjectView(params)

# Load model
model_name = 'angled_env_directional_16_2'
model = torch.load(f"./models/{model_name}/final_weights.pt", map_location = torch.device('cpu'))

model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialise grid
grid = Grid(params)

# Initialise environment
env = None
env = grid.init_env(model.env_channels)
# env = grid.add_env(env, "linear", 0, angle = 45)
# env = grid.add_env(env, "circle", 0, center = (grid_size/2, grid_size/2), circle_radius = model.grid_size/2)
# env = grid.add_env(env, "directional", 0, angle = 45, center = (grid_size/2, grid_size/2))
env = grid.add_env(env, "directional", 0, angle = 45, center = (params.grid_size/2, params.grid_size/2))
# env = grid.add_env(env, 'none')



# Run model
state_history, env_history = grid.run(model, env, params)

# # Create animation
filename = f'./models/{model_name}/run.mp4'
create_animation(state_history, env_history, filename, params)

# Visualize all channels
# filename = f'./models/{model_name}/all_channels_run.mp4'
# visualize_all_channels(state_history, filename, model.model_channels, params)

# Visualize seed losses at different seed positions
# filename = f"./models/{model_name}/diff_seed_losses.png"
# visualize_seed_losses(model_name, grid, filename, params = params, env = env)

# Create progress animation
# states, envs = load_progress_states(model_name, grid, iterations, grid_size, angle, env)
# filename = f'./models/{model_name}/progress.mp4'
# create_progress_animation(states, envs, iterations, nSeconds, filename, vis_env = True)

# Visualize parameter sizes
# filename = f'./models/{model_name}/parameter_sizes.png'
# plot_parameter_sizes(model_name, filename)


