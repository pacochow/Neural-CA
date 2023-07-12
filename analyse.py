import pickle 
import numpy as np
import matplotlib.pyplot as plt
from helpers.visualizer import visualize_single_hidden_unit
import torch
from src.analysis import *
from src.params import ObjectView
from src.grid import Grid
from helpers.helpers import rotate_image

np.random.seed(0)
torch.manual_seed(0)

# Load model
model_name = "modulated_angled_env_directional_16_2_400"
model = torch.load(f"./models/{model_name}/final_weights.pt", map_location = torch.device('cpu'))

# Load hidden unit histories
filename = f'./models/{model_name}/hidden_unit_history.pkl'
with open(filename, 'rb') as fp:
    hidden_unit_history = pickle.load(fp)
living_cells = np.load(f"./models/{model_name}/living_cells.npy")


params = {
       
# Run params
'model_channels': model.model_channels, 
'env_channels': model.env_channels,
'grid_size': 50,
'iterations': 200,                  # Number of iterations in animation
'angle': 0.0,                       # Perceiving angle
'env_angle': 45,                    # Environment angle
'dynamic_env': False,               # Run with moving environment
'dynamic_env_type': 'free move',    # Type of moving environment    
'destroy': False,                   # Whether pattern is disrupted mid animation
'destroy_type': 0,                  # Type of pattern disruption
'seed': None,                       # Coordinates of seed
'vis_env': False,                   # Visualize environment in animation
'vis_hidden': True,                 # Visualize hidden unit activity throughout run
'hidden_loc': [(25, 25), (30, 20)], # Location of where to visualize hidden unit activity
'knockout': True,                   # Whether hidden unit is fixed
'knockout_unit': [42],                # Hidden unit to fix
'nSeconds': 10}                     # Length of animation}

params = ObjectView(params)


model.params = params
model.knockout = params.knockout
model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Initialise grid
grid = Grid(params)

# Initialise environment
env = None
env = grid.init_env(model.env_channels)
env = grid.add_env(env, "directional", 0, angle = params.env_angle, center = (params.grid_size/2, params.grid_size/2))


normalized_profiles, early_sorted = find_hox_units(hidden_unit_history, living_cells[:60], early = True)

filename = f'./models/{model_name}/early_hox.png'
plot_expression_profiles(normalized_profiles, early_sorted, filename)



early_loss = progressive_knockout_loss(model, early_sorted, grid, env, params)
late_loss = progressive_knockout_loss(model, early_sorted[::-1], grid, env, params)

plt.plot(np.log10(early_loss))
plt.plot(np.log10(late_loss))
plt.xlabel("Number of units knocked out")
plt.ylabel("Log loss")

plt.tight_layout()
plt.legend(["Early units", "Late units"])
filename = f'./models/{model_name}/early_vs_late.png'
plt.savefig(filename)


# save = f'./models/{model_name}/hox.mp4'
# visualize_single_hidden_unit(hidden_unit_history, units = early_sorted[:20], filename = save)

# Perform PCA on hidden units and cluster
# clusters, x_pca = cluster_hidden_units(model)
# clusters = dict(zip(np.arange(400), clusters))
