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


# Load model
model_name2 = "naive_ladybug"
model2 = torch.load(f"./models/{model_name2}/final_weights.pt", map_location = torch.device('cpu'))
filename2 = f'./models/{model_name2}/hidden_unit_history.pkl'
with open(filename2, 'rb') as fp:
    hidden_unit_history2 = pickle.load(fp)
living_cells2 = np.load(f"./models/{model_name2}/living_cells.npy")

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
'vis_hidden': False,                 # Visualize hidden unit activity throughout run
'hidden_loc': [(25, 25), (30, 20)], # Location of where to visualize hidden unit activity
'knockout': False,                   # Whether hidden unit is fixed
'knockout_unit': [42],                # Hidden unit to fix
'nSeconds': 10}                     # Length of animation}

params = ObjectView(params)


model.params = params
model.knockout = params.knockout
model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model2.params = params
model2.knockout = params.knockout
model2.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

og_model.params = params
og_model.knockout = params.knockout
og_model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialise grid
grid = Grid(params)

# Initialise environment
env = grid.init_env(model.env_channels)
env = grid.add_env(env, "directional", 0, angle = params.env_angle, center = (params.grid_size/2, params.grid_size/2))


# Run model
# state_history, _, _ = grid.run(model, env, params)
og_state_history, _, _ = grid.run(og_model, env, params)
# state_history2, _, _ = grid.run(model2, env, params)

# state_history = state_history[:60]
og_state_history = og_state_history[:60]
# state_history2 = state_history2[:60]



# Comparing channel differences  
# # Create a grid of subplots
# fig, axs = plt.subplots(3, 4, figsize=(25, 15))

# for j in range(12):
#     i = j+4
#     # Plot the data on subplot i, j
#     diff = (state_history[..., i:i+1] - og_state_history[..., i:i+1])**2
#     diff2 = (state_history2[..., i:i+1] - og_state_history[..., i:i+1])**2
#     axs[j//4, j%4].plot(diff.sum(axis = (1, 2, 3)))
#     axs[j//4, j%4].plot(diff2.sum(axis = (1, 2, 3)))
#     axs[j//4, j%4].set_title(f'Channel {i+1}', fontsize = 17)
#     axs[j//4, j%4].legend(["Retrained", "Naive"], fontsize = 15)
#     axs[j//4, j%4].set_xlabel("Iterations", fontsize  = 15)
#     axs[j//4, j%4].set_ylabel("Squared difference with gecko", fontsize = 15)

# plt.legend(["1-4"]+[i+1 for i in range(4, 16)])
# plt.tight_layout()
# plt.show()



_, early_sorted = find_hox_units(hidden_unit_history, living_cells[:60], phase = (0, 20))
_, mid_sorted = find_hox_units(hidden_unit_history, living_cells[:60], phase = (21, 40))
_, late_sorted = find_hox_units(hidden_unit_history, living_cells[:60], phase = (41, 60))

_, early_sorted2 = find_hox_units(hidden_unit_history2, living_cells2[:60], phase = (0, 20))
_, mid_sorted2 = find_hox_units(hidden_unit_history2, living_cells2[:60], phase = (21, 40))
_, late_sorted2 = find_hox_units(hidden_unit_history2, living_cells2[:60], phase = (41, 60))

_, og_early_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (0, 20))
_, og_mid_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (21, 34))
_, og_late_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (35, 60))

# Compare developmental stages
# compare_developmental_stages(og_early_sorted, og_mid_sorted, og_late_sorted)

# Quantify how much retraining improved training compared to naive training 
# quantify_retrain_improvement(losses, retrained_losses, difference = True)

# Plot expression profiles
# filename = f'./models/{model_name}/early_hox.png'
# plot_expression_profiles(normalized_profiles, early_sorted, filename)

# Plot conserved genes
conserved_early = []
conserved_mid = []
conserved_late = []

conserved_early2 = []
conserved_mid2 = []
conserved_late2 = []
for i in range(len(early_sorted)):
    conserved_early.append(len(set(early_sorted[:i]) & set(og_early_sorted[:i])))
    conserved_mid.append(len(set(mid_sorted[:i]) & set(og_mid_sorted[:i])))
    conserved_late.append(len(set(late_sorted[:i]) & set(og_late_sorted[:i])))
    
    conserved_early2.append(len(set(early_sorted2[:i]) & set(og_early_sorted[:i])))
    conserved_mid2.append(len(set(mid_sorted2[:i]) & set(og_mid_sorted[:i])))
    conserved_late2.append(len(set(late_sorted2[:i]) & set(og_late_sorted[:i])))
plt.plot(conserved_early, color = 'tab:blue')
plt.plot(conserved_mid, color = 'tab:orange')
plt.plot(conserved_late, color = 'tab:green')
plt.plot(conserved_early2, color = 'tab:blue', linestyle = 'dashed')
plt.plot(conserved_mid2, color = 'tab:orange', linestyle = 'dashed')
plt.plot(conserved_late2, color = 'tab:green', linestyle = 'dashed')
plt.xlabel("Top n hox genes")
plt.ylabel("Number of conserved hox genes after retraining")
plt.legend(["Retrained – Early", "Retrained – Mid", "Retrained – Late", 
            "Naive – Early", "Naive – Mid", "Naive – Late"])
plt.show()
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
