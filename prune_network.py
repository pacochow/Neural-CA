from helpers.helpers import *
import torch   
from src.standard_ca import *
from src.grid import *
from src.pruning import *
from src.params import ObjectView
from helpers.visualizer import *

# Load model
model_name = "modulated_angled_env_directional_16_2_400"
model = torch.load(f"./models/{model_name}/final_weights.pt", map_location = torch.device('cpu'))

params = {
       
# Run params
'model_channels': model.model_channels, 
'env_channels': model.env_channels,
'grid_size': 50,
'iterations': 400,                  # Number of iterations in animation
'angle': 0.0,                       # Perceiving angle
'env_angle': 45,                    # Environment angle
'dynamic_env': False,               # Run with moving environment
'dynamic_env_type': 'free move',    # Type of moving environment    
'destroy': False,                    # Whether pattern is disrupted mid animation
'destroy_type': 0,                  # Type of pattern disruption
'seed': None,                       # Coordinates of seed
'vis_env': False,                   # Visualize environment in animation
'vis_hidden': True,                 # Visualize hidden unit activity throughout run
'hidden_loc': [(25, 25), (30, 20)], # Location of where to visualize hidden unit activity
'knockout': True,                   # Whether hidden unit is fixed
'knockout_unit': 42,                # Hidden unit to fix
'nSeconds': 10,                     # Length of animation
'enhance': False}                   # Increasing activation of a channel                      

params = ObjectView(params)




model.params = params
model.knockout = params.knockout
model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialise grid
grid_size = params.grid_size
grid = Grid(params)

# Initialise environment
env = None
env = grid.init_env(model.env_channels)
# env = grid.add_env(env, "circle", 0)
env = grid.add_env(env, "directional", 0, angle = params.env_angle)

# Prune by channel
# filename = f"./models/{model_name}/visualize_pruning_by_channel.mp4"
# visualize_pruning_by_channel(model, grid, filename, params, env = env)

# Prune by units
filename = f"./models/{model_name}/enhance_unit_effects.png"
visualize_unit_effect(model, grid, env, params, [0, 150], filename)


# # Prune by channel
# channel = 8
# pruned_model = prune_by_channel(model, channel-1)

# # Run model
# state_history, _ = grid.run(pruned_model, env, params)

# # Create animation
# filename = f"./models/{model_name}/pruned_channel_{channel}_run.mp4"
# create_animation(state_history, _, filename, params)

# Visualise progress animation
# filename = f"./models/{model_name}/pruned_visualization.mp4"
# visualize_pruning(model_name, grid, filename,  params, env)





# model_name_2 = 'env_circle_16_1'
# model2 = torch.load(f"./models/{model_name_2}/final_weights.pt")

# grid2 = Grid(grid_size, model2.model_channels)
# env2 = grid2.init_env(model2.env_channels)
# env2 = grid2.add_env(env2, "circle", 0)
# comparing_pruning_losses(model_name, grid, env, model_name_2, grid2, env2, "comparing_pruned_loss.png")





# # Prune model
# percent = 23
# model_size, pruned_size, pruned_model = prune_by_percent(model, percent=percent)


