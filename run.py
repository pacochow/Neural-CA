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
'iterations': 100,                  # Number of iterations in animation
'angle': 0.0,                       # Perceiving angle
'env_angle': 45,                    # Environment angle
'dynamic_env': False,               # Run with moving environment
'dynamic_env_type': 'free move',    # Type of moving environment    
'modulate': False,                  # Environment modulation
'destroy': False,                   # Whether pattern is disrupted mid animation
'destroy_type': 0,                  # Type of pattern disruption
'seed': None,                       # Coordinates of seed
'vis_env': False,                   # Visualize environment in animation
'vis_hidden': True,                 # Visualize hidden unit activity throughout run
'hidden_loc': [(25, 25), (30, 20)], # Location of where to visualize hidden unit activity
'knockout': True,                   # Whether hidden unit is fixed
'knockout_unit': 6,                # Hidden unit to fix
'nSeconds': 10}                     # Length of animation}

params = ObjectView(params)

# Load model
model_name = "experimental"
model = torch.load(f"./models/{model_name}/final_weights.pt", map_location = torch.device('cpu'))
model.params = params
model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialise grid
grid = Grid(params)

# Initialise environment
env = None
env = grid.init_env(model.env_channels)
# env = grid.add_env(env, "linear", 0, angle = 45)
# env = grid.add_env(env, "circle", 0, center = (grid_size/2, grid_size/2), circle_radius = model.grid_size/2)
# env = grid.add_env(env, "directional", 0, angle = 45, center = (grid_size/2, grid_size/2))
env = grid.add_env(env, "directional proportional", 0, angle = params.env_angle, center = (params.grid_size/2, params.grid_size/2))
# env = grid.add_env(env, 'none')



# Run model
state_history, env_history, hidden_history = grid.run(model, env, params)
# for i in range(529):
#     plt.plot(hidden_history[0, :, i])
hidden_history = hidden_history.reshape(len(params.hidden_loc), params.iterations, 23, 23)


# # Create animation
# filename = f'./models/{model_name}/run.mp4'
# create_animation(state_history, env_history, filename, params)

# Visualize hidden units
filename = f'./models/{model_name}/knockout_hidden_units.mp4'
visualize_hidden_units(state_history, hidden_history, filename, params)

target = rotate_image(model.target, params.env_angle+45)
loss = ((state_history[-1, :, :, :4]-target[0].numpy())**2).mean()

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


