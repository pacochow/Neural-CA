import pickle 
import numpy as np
import matplotlib.pyplot as plt
from helpers.visualizer import visualize_single_hidden_unit
import torch
from src.analysis_utils import *
from src.params import ObjectView
from src.grid import Grid
from helpers.helpers import rotate_image

np.random.seed(0)
torch.manual_seed(0)

# Load model
model_name = "retrained"
model = torch.load(f"./models/{model_name}/final_weights.pt", map_location = torch.device('cpu'))
retrained_losses = torch.load(f"./models/{model_name}/losses.pt", map_location = torch.device('cpu'))

# Load hidden unit histories
filename = f'./models/{model_name}/hidden_unit_history.pkl'
with open(filename, 'rb') as fp:
    hidden_unit_history = pickle.load(fp)
living_cells = np.load(f"./models/{model_name}/living_cells.npy")

# Load model
og_model_name = "modulated_angled_env_directional_16_2_400"
og_model = torch.load(f"./models/{og_model_name}/final_weights.pt", map_location = torch.device('cpu'))
losses = torch.load(f"./models/{og_model_name}/losses.pt", map_location = torch.device('cpu'))

# Load hidden unit histories
og_filename = f'./models/{og_model_name}/hidden_unit_history.pkl'
with open(og_filename, 'rb') as fp:
    og_hidden_unit_history = pickle.load(fp)
og_living_cells = np.load(f"./models/{og_model_name}/living_cells.npy")



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
env = grid.init_env(model.env_channels)
env = grid.add_env(env, "directional", 0, angle = params.env_angle, center = (params.grid_size/2, params.grid_size/2))


# hidden_unit_history_array = np.zeros((params.grid_size**2, hidden_unit_history[(0, 0)].shape[0], model.hidden_units))
# og_hidden_unit_history_array = np.zeros((params.grid_size**2, hidden_unit_history[(0, 0)].shape[0], model.hidden_units))

# for i, j in enumerate(hidden_unit_history.keys()):
#     hidden_unit_history_array[i] = hidden_unit_history[j]
#     og_hidden_unit_history_array[i] = og_hidden_unit_history[j]
    
# hidden_unit_history_array = hidden_unit_history_array[:, :60]
# development_profiles = np.abs(hidden_unit_history_array-np.expand_dims(hidden_unit_history_array[:, 0], axis = 1))
# normalized_profiles = development_profiles/living_cells[:60].reshape(1, 60, 1)

# og_hidden_unit_history_array = og_hidden_unit_history_array[:, :60]
# og_development_profiles = np.abs(og_hidden_unit_history_array-np.expand_dims(og_hidden_unit_history_array[:, 0], axis = 1))
# og_normalized_profiles = og_development_profiles/og_living_cells[:60].reshape(1, 60, 1)

# diff = (normalized_profiles-og_normalized_profiles)**2
# plt.plot(diff.sum(axis = (0, 2)))



# quantify_retrain_improvement(losses, retrained_losses, difference = True)

normalized_profiles, early_sorted = find_hox_units(hidden_unit_history, living_cells[:60], early = True)
og_normalized_profiles, og_early_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], early = True)

diff = (normalized_profiles-og_normalized_profiles)**2
plt.plot(diff.sum(axis = 1))

# filename = f'./models/{model_name}/early_hox.png'
# plot_expression_profiles(normalized_profiles, early_sorted, filename)

# conserved =[]
# for i in range(len(early_sorted)):
#     conserved.append(len(set(early_sorted[:i]) & set(og_early_sorted[:i])))
# plt.plot(conserved)
# plt.xlabel("Top n hox genes")
# plt.ylabel("Number of conserved hox genes after retraining")
# filename = f"./models/{model_name}/conserved_hox.png"
# plt.savefig(filename)
# early_loss = progressive_knockout_loss(model, early_sorted, grid, env, params)
# late_loss = progressive_knockout_loss(model, early_sorted[::-1], grid, env, params)

# plt.plot(np.log10(early_loss))
# plt.plot(np.log10(late_loss))
# plt.xlabel("Number of units knocked out")
# plt.ylabel("Log loss")

# plt.tight_layout()
# plt.legend(["Early units", "Late units"])
# filename = f'./models/{model_name}/early_vs_late.png'
# plt.savefig(filename)


# save = f'./models/{model_name}/hox.mp4'
# visualize_single_hidden_unit(hidden_unit_history, units = early_sorted[:20], filename = save)

# Perform PCA on hidden units and cluster
# clusters, x_pca = cluster_hidden_units(model)
# clusters = dict(zip(np.arange(400), clusters))
