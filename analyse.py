import pickle 
import numpy as np
import matplotlib.pyplot as plt
from helpers.visualizer import visualize_single_hidden_unit
import torch
from src.analysis_utils import *
from src.params import ObjectView
from src.grid import Grid
from helpers.helpers import rotate_image
from tqdm import tqdm


np.random.seed(0)
torch.manual_seed(0)

# Load model
og_model_name = "modulated_angled_env_directional_16_2_400"
og_model = torch.load(f"./models/{og_model_name}/final_weights.pt", map_location = torch.device('cpu'))
losses = torch.load(f"./models/{og_model_name}/losses.pt", map_location = torch.device('cpu'))
og_filename = f'./models/{og_model_name}/hidden_unit_history.pkl'
with open(og_filename, 'rb') as fp:
    og_hidden_unit_history = pickle.load(fp)
og_living_cells = np.load(f"./models/{og_model_name}/living_cells.npy")

# Load model
model_name_lb = "retrained_ladybug"
model_lb = torch.load(f"./models/{model_name_lb}/final_weights.pt", map_location = torch.device('cpu'))
filename_lb = f'./models/{model_name_lb}/hidden_unit_history.pkl'
with open(filename_lb, 'rb') as fp:
    hidden_unit_history_lb = pickle.load(fp)
living_cells_lb = np.load(f"./models/{model_name_lb}/living_cells.npy")

# Load model
model_name_sl = "small_legs"
model_sl = torch.load(f"./models/{model_name_sl}/final_weights.pt", map_location = torch.device('cpu'))
retrained_losses = torch.load(f"./models/{model_name_sl}/losses.pt", map_location = torch.device('cpu'))
filename_sl = f'./models/{model_name_sl}/hidden_unit_history.pkl'
with open(filename_sl, 'rb') as fp:
    hidden_unit_history_sl = pickle.load(fp)
living_cells_sl = np.load(f"./models/{model_name_sl}/living_cells.npy")

# Load model
model_name_sn = "snake"
model_sn = torch.load(f"./models/{model_name_sn}/final_weights.pt", map_location = torch.device('cpu'))
filename_sn = f'./models/{model_name_sn}/hidden_unit_history.pkl'
with open(filename_sn, 'rb') as fp:
    hidden_unit_history_sn = pickle.load(fp)
living_cells_sn = np.load(f"./models/{model_name_sn}/living_cells.npy")

# Load model
model_name_nl = "naive_ladybug"
model_nl = torch.load(f"./models/{model_name_nl}/final_weights.pt", map_location = torch.device('cpu'))
filename_nl = f'./models/{model_name_nl}/hidden_unit_history.pkl'
with open(filename_nl, 'rb') as fp:
    hidden_unit_history_nl = pickle.load(fp)
living_cells_nl = np.load(f"./models/{model_name_nl}/living_cells.npy")

# Load model
model_name_sy = "snake_yellow"
model_sy = torch.load(f"./models/{model_name_sy}/final_weights.pt", map_location = torch.device('cpu'))
filename_sy = f'./models/{model_name_sy}/hidden_unit_history.pkl'
with open(filename_sy, 'rb') as fp:
    hidden_unit_history_sy = pickle.load(fp)
living_cells_sy = np.load(f"./models/{model_name_sy}/living_cells.npy")

# Load model
model_name_ns = "naive_snake"
model_ns = torch.load(f"./models/{model_name_ns}/final_weights.pt", map_location = torch.device('cpu'))
# filename_ns = f'./models/{model_name_ns}/hidden_unit_history.pkl'
# with open(filename_ns, 'rb') as fp:
#     hidden_unit_history_ns = pickle.load(fp)
# living_cells_ns = np.load(f"./models/{model_name_ns}/living_cells.npy")





params = {
       
# Run params
'model_channels': og_model.model_channels, 
'env_channels': og_model.env_channels,
'grid_size': 50,
'iterations': 100,                  # Number of iterations in animation
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

og_model.params = params
og_model.knockout = params.knockout
og_model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_lb.params = params
model_lb.knockout = params.knockout
model_lb.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_sl.params = params
model_sl.knockout = params.knockout
model_sl.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_sn.params = params
model_sn.knockout = params.knockout
model_sn.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_sy.params = params
model_sy.knockout = params.knockout
model_sy.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_nl.params = params
model_nl.knockout = params.knockout
model_nl.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_ns.params = params
model_ns.knockout = params.knockout
model_ns.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Initialise grid
grid = Grid(params)

# Initialise environment
env = grid.init_env(og_model.env_channels)
env = grid.add_env(env, "directional", 0, angle = params.env_angle, center = (params.grid_size/2, params.grid_size/2))

# og_state_histories = np.zeros((10, 100, 50, 50, 16))
# state_histories_lb = np.zeros((10, 100, 50, 50, 16))
# state_histories_nl = np.zeros((10, 100, 50, 50, 16))
# # Run model many times
# for seed in tqdm(range(10)):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     og_state_history, _, _ = grid.run(og_model, env, params)

#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     state_history_lb, _, _ = grid.run(model_lb, env, params)

#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     state_history_nl, _, _ = grid.run(model_nl, env, params)

#     og_state_histories[seed] = og_state_history[:100].clip(0, 1)
#     state_histories_lb[seed] = state_history_lb[:100].clip(0, 1)
#     state_histories_nl[seed] = state_history_nl[:100].clip(0, 1)

# og_state_history = og_state_histories.mean(axis = 0)
# state_history_lb = state_histories_lb.mean(axis = 0)
# state_history_nl = state_histories_nl.mean(axis = 0)

# Compare visible channel differences
# plt.figure(figsize = (8, 8))
# diff = (state_history_lb[..., :4] - og_state_history[..., :4])**2
# diff2 = (state_history_nl[..., :4] - og_state_history[..., :4])**2
# plt.plot(diff.sum(axis = (1, 2, 3)))
# plt.plot(diff2.sum(axis = (1, 2, 3)))
# plt.legend(["Retrained", "Naive"])
# plt.xlabel("Iterations")
# plt.ylabel("Visible channels squared difference with gecko")
# plt.show()

# Comparing other channel differences  
# Create a grid of subplots
# fig, axs = plt.subplots(3, 4, figsize=(25, 15))

# for j in range(12):
#     i = j+4
#     # Plot the data on subplot i, j
#     diff = (state_history_lb[..., i:i+1] - og_state_history[..., i:i+1])**2
#     diff2 = (state_history_nl[..., i:i+1] - og_state_history[..., i:i+1])**2
#     axs[j//4, j%4].plot(diff.sum(axis = (1, 2, 3)))
#     axs[j//4, j%4].plot(diff2.sum(axis = (1, 2, 3)))
#     axs[j//4, j%4].set_title(f'Channel {i+1}', fontsize = 17)
#     axs[j//4, j%4].legend(["Retrained", "Naive"], fontsize = 15)
#     axs[j//4, j%4].set_xlabel("Iterations", fontsize  = 15)
#     axs[j//4, j%4].set_ylabel("Squared difference with gecko", fontsize = 15)

# plt.tight_layout()
# plt.show()

# _, og_early_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (0, 20))
# _, og_mid_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (21, 34))
# _, og_late_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (35, 60))

# _, early_sorted_lb = find_hox_units(hidden_unit_history_lb, living_cells_lb[:60], phase = (0, 20))
# _, mid_sorted_lb = find_hox_units(hidden_unit_history_lb, living_cells_lb[:60], phase = (21, 40))
# _, late_sorted_lb = find_hox_units(hidden_unit_history_lb, living_cells_lb[:60], phase = (41, 60))

# _, early_sorted_sl = find_hox_units(hidden_unit_history_sl, living_cells_sl[:60], phase = (0, 20))
# # _, mid_sorted_sl = find_hox_units(hidden_unit_history_sl, living_cells_sl[:60], phase = (21, 40))
# _, late_sorted_sl = find_hox_units(hidden_unit_history_sl, living_cells_sl[:60], phase = (41, 60))

# _, early_sorted_sn = find_hox_units(hidden_unit_history_sn, living_cells_sn[:60], phase = (0, 20))
# # _, mid_sorted_sn = find_hox_units(hidden_unit_history_sn, living_cells_sn[:60], phase = (21, 40))
# _, late_sorted_sn = find_hox_units(hidden_unit_history_sn, living_cells_sn[:60], phase = (41, 60))

# _, early_sorted_sy = find_hox_units(hidden_unit_history_sy, living_cells_sy[:60], phase = (0, 20))
# # _, mid_sorted_sy= find_hox_units(hidden_unit_history_sy, living_cells_sy[:60], phase = (21, 40))
# _, late_sorted_sy = find_hox_units(hidden_unit_history_sy, living_cells_sy[:60], phase = (41, 60))

# _, early_sorted_nl = find_hox_units(hidden_unit_history_nl, living_cells_nl[:60], phase = (0, 20))
# _, mid_sorted_nl = find_hox_units(hidden_unit_history_nl, living_cells_nl[:60], phase = (21, 40))
# _, late_sorted_nl = find_hox_units(hidden_unit_history_nl, living_cells_nl[:60], phase = (41, 60))



# Compare developmental stages
# compare_developmental_stages(og_early_sorted, og_mid_sorted, og_late_sorted)

# Quantify how much retraining improved training compared to naive training 
# quantify_retrain_improvement(losses, retrained_losses, difference = True)

# Plot expression profiles
# filename = f'./models/{model_name}/early_hox.png'
# plot_expression_profiles(normalized_profiles, early_sorted, filename)






# Plot conserved genes
conserved_early_lb = []
conserved_mid_lb = []
conserved_late_lb = []

conserved_early_nl = []
conserved_mid_nl = []
conserved_late_nl = []

conserved_early_sl = []
conserved_late_sl = []

conserved_early_sn = []
conserved_late_sn = []

conserved_early_sy = []
conserved_late_sy = []

# Plot conserved genes at every iteration
for i in tqdm(range(60)):
    _, og_early_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (i, i+1))
    _, early_sorted_sl = find_hox_units(hidden_unit_history_sl, living_cells_sl[:60], phase = (i, i+1))
    _, early_sorted_sn = find_hox_units(hidden_unit_history_sn, living_cells_sl[:60], phase = (i, i+1))
    _, early_sorted_sy = find_hox_units(hidden_unit_history_sy, living_cells_sl[:60], phase = (i, i+1))
    _, early_sorted_lb = find_hox_units(hidden_unit_history_lb, living_cells_lb[:60], phase = (i, i+1))
    _, early_sorted_nl = find_hox_units(hidden_unit_history_nl, living_cells_nl[:60], phase = (i, i+1))
    
    conserved_early_sl.append(len(set(early_sorted_sl[:20]) & set(og_early_sorted[:20])))
    conserved_early_sn.append(len(set(early_sorted_sn[:20]) & set(og_early_sorted[:20])))
    conserved_early_sy.append(len(set(early_sorted_sy[:20]) & set(og_early_sorted[:20])))
    conserved_early_lb.append(len(set(early_sorted_lb[:20]) & set(og_early_sorted[:20])))
    conserved_early_nl.append(len(set(early_sorted_nl[:20]) & set(og_early_sorted[:20])))

plt.figure(figsize = (8,8))
plt.plot((20,)*60, color = 'black')
plt.plot(conserved_early_sl)
plt.plot(conserved_early_sn)
plt.plot(conserved_early_sy)
plt.plot(conserved_early_lb)
plt.plot(conserved_early_nl)

plt.xlabel("Iterations")
plt.ylabel("Number of conserved units")
plt.legend(["Full conservation", "Gecko – small legs", "Snake", "Snake – yellow", "Ladybug", "Naive ladybug control"])


# for i in range(len(og_early_sorted)):
#     conserved_early_sl.append(len(set(early_sorted_sl[:i]) & set(og_early_sorted[:i])))
#     conserved_late_sl.append(len(set(late_sorted_sl[:i]) & set(og_late_sorted[:i])))
    
#     conserved_early_sn.append(len(set(early_sorted_sn[:i]) & set(og_early_sorted[:i])))
#     conserved_late_sn.append(len(set(late_sorted_sn[:i]) & set(og_early_sorted[:i])))
    
#     conserved_early_sy.append(len(set(early_sorted_sy[:i]) & set(og_early_sorted[:i])))
#     conserved_late_sy.append(len(set(late_sorted_sy[:i]) & set(og_early_sorted[:i])))
    
#     conserved_early_lb.append(len(set(early_sorted_lb[:i]) & set(og_early_sorted[:i])))
#     conserved_late_lb.append(len(set(late_sorted_lb[:i]) & set(og_early_sorted[:i])))
    # conserved_mid_lb.append(len(set(mid_sorted_lb[:i]) & set(og_mid_sorted[:i])))
    # conserved_late_lb.append(len(set(late_sorted_lb[:i]) & set(og_late_sorted[:i])))
    
    # conserved_early_nl.append(len(set(early_sorted_nl[:i]) & set(og_early_sorted[:i])))
    # conserved_mid_nl.append(len(set(mid_sorted_nl[:i]) & set(og_mid_sorted[:i])))
    # conserved_late_nl.append(len(set(late_sorted_nl[:i]) & set(og_late_sorted[:i])))
    

    
    
# n_genes = 50
# plt.plot(np.arange(n_genes), color = 'black')
# plt.plot(conserved_early_sl[:n_genes])
# plt.plot(conserved_early_sn[:n_genes])
# plt.plot(conserved_early_sy[:n_genes])
# plt.plot(conserved_early_lb[:n_genes])


# plt.plot(conserved_late_sl[:n_genes])
# plt.plot(conserved_late_sn[:n_genes])
# plt.plot(conserved_late_sy[:n_genes])
# plt.plot(conserved_late_lb[:n_genes])

# plt.plot(conserved_early_lb[:n_genes], color = 'tab:blue')
# plt.plot(conserved_mid_lb[:n_genes], color = 'tab:orange')
# plt.plot(conserved_late_lb[:n_genes], color = 'tab:green')


# plt.plot(conserved_early_nl[:n_genes], color = 'tab:blue', linestyle = 'dashed')
# plt.plot(conserved_mid_nl[:n_genes], color = 'tab:orange', linestyle = 'dashed')
# plt.plot(conserved_late_nl[:n_genes], color = 'tab:green', linestyle = 'dashed')

# plt.xlabel("Top n hox genes")
# plt.ylabel("Number of conserved hox genes after retraining")

# plt.legend(["Full conservation", "Gecko – small legs", "Snake", "Snake – Yellow", "Ladybug"])
# plt.legend(["Full conservation", "Retrained – Early", "Retrained – Mid", "Retrained – Late", 
#             "Naive – Early", "Naive – Mid", "Naive – Late"])
# plt.savefig(f"./models/{model_name_nl}/retrained_vs_naive_hox_allstages.png")
# plt.show()
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

