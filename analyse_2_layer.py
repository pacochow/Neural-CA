import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from helpers.figures import *
from helpers.helpers import rotate_image
from helpers.visualizer import visualize_single_hidden_unit
from src.analysis_utils import *
from src.grid import Grid
from src.params import ObjectView

np.random.seed(0)
torch.manual_seed(0)

# Load naively trained and retrained models
og_model_name = "2_hidden"
og_model = torch.load(f"./models/{og_model_name}/final_weights.pt", map_location = torch.device('cpu'))
losses = torch.load(f"./models/{og_model_name}/losses.pt", map_location = torch.device('cpu'))
og_filename = f'./models/{og_model_name}/hidden_unit_history.pkl'
with open(og_filename, 'rb') as fp:
    og_hidden_unit_history = pickle.load(fp)
og_filename = f'./models/{og_model_name}/hidden_unit_2_history.pkl'
with open(og_filename, 'rb') as fp:
    og_hidden_unit_history_2 = pickle.load(fp)
og_living_cells = np.load(f"./models/{og_model_name}/living_cells.npy")

model_name_lb = "2_hidden_ladybug"
model_lb = torch.load(f"./models/{model_name_lb}/final_weights.pt", map_location = torch.device('cpu'))
retrained_losses = torch.load(f"./models/{model_name_lb}/losses.pt", map_location = torch.device('cpu'))
filename_lb = f'./models/{model_name_lb}/hidden_unit_history.pkl'
with open(filename_lb, 'rb') as fp:
    hidden_unit_history_lb = pickle.load(fp)
filename_lb = f'./models/{model_name_lb}/hidden_unit_2_history.pkl'
with open(filename_lb, 'rb') as fp:
    hidden_unit_history_lb_2 = pickle.load(fp)
living_cells_lb = np.load(f"./models/{model_name_lb}/living_cells.npy")

model_name_sl = "2_hidden_small_legs"
model_sl = torch.load(f"./models/{model_name_sl}/final_weights.pt", map_location = torch.device('cpu'))
filename_sl = f'./models/{model_name_sl}/hidden_unit_history.pkl'
with open(filename_sl, 'rb') as fp:
    hidden_unit_history_sl = pickle.load(fp)
filename_sl = f'./models/{model_name_sl}/hidden_unit_2_history.pkl'
with open(filename_sl, 'rb') as fp:
    hidden_unit_history_sl_2 = pickle.load(fp)
living_cells_sl = np.load(f"./models/{model_name_sl}/living_cells.npy")

model_name_sn = "2_hidden_snake"
model_sn = torch.load(f"./models/{model_name_sn}/final_weights.pt", map_location = torch.device('cpu'))
filename_sn = f'./models/{model_name_sn}/hidden_unit_history.pkl'
with open(filename_sn, 'rb') as fp:
    hidden_unit_history_sn = pickle.load(fp)
filename_sn = f'./models/{model_name_sn}/hidden_unit_2_history.pkl'
with open(filename_sn, 'rb') as fp:
    hidden_unit_history_sn_2 = pickle.load(fp)
living_cells_sn = np.load(f"./models/{model_name_sn}/living_cells.npy")

model_name_nl = "2_hidden_naive_ladybug"
model_nl = torch.load(f"./models/{model_name_nl}/final_weights.pt", map_location = torch.device('cpu'))
filename_nl = f'./models/{model_name_nl}/hidden_unit_history.pkl'
with open(filename_nl, 'rb') as fp:
    hidden_unit_history_nl = pickle.load(fp)
nl_losses = torch.load(f"./models/{model_name_nl}/losses.pt", map_location = torch.device('cpu'))
filename_nl = f'./models/{model_name_nl}/hidden_unit_2_history.pkl'
with open(filename_nl, 'rb') as fp:
    hidden_unit_history_nl_2 = pickle.load(fp)
living_cells_nl = np.load(f"./models/{model_name_nl}/living_cells.npy")

model_name_sy = "2_hidden_snake_yellow"
model_sy = torch.load(f"./models/{model_name_sy}/final_weights.pt", map_location = torch.device('cpu'))
filename_sy = f'./models/{model_name_sy}/hidden_unit_history.pkl'
with open(filename_sy, 'rb') as fp:
    hidden_unit_history_sy = pickle.load(fp)
filename_sy = f'./models/{model_name_sy}/hidden_unit_2_history.pkl'
with open(filename_sy, 'rb') as fp:
    hidden_unit_history_sy_2 = pickle.load(fp)
living_cells_sy = np.load(f"./models/{model_name_sy}/living_cells.npy")

model_name_ns = "2_hidden_naive_snake"
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
'modulate_env': True,               # Use alpha channel to modulate environment
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

# Plot living cells
# fig=plt.figure(figsize = (4, 7))
# plt.plot(og_living_cells[:80], linewidth = 3)
# plt.xticks([0, 20, 40, 60, 80], fontsize =13)
# plt.yticks([0, 200, 400, 600, 800], fontsize = 13)
# plt.xlabel("Developmental time (iterations)")
# plt.ylabel("Number of living cells")
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(2)
# filename = f"./models/{og_model_name}/living_cells.png"
# plt.savefig(filename, bbox_inches = 'tight')





# FIG 2A

# og_early_profiles, og_early_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:80], phase = (0, 10))
# og_late_profiles, og_late_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:80], phase = (31, 60))
# early_mean = og_early_profiles[:, og_early_sorted[:20]].mean(axis=1)
# early_std = og_early_profiles[:, og_early_sorted[:20]].std(axis=1)
# late_mean = og_late_profiles[:, og_late_sorted[:20]].mean(axis=1)
# late_std=og_late_profiles[:, og_late_sorted[:20]].std(axis=1)

# fig=plt.figure(figsize = (10, 8))

# for a, i in enumerate(og_early_sorted[:20]):
    
#     plt.plot(og_early_profiles[:,i], linewidth = 3.5, alpha = 0.1, color = 'tab:red');

# for a, i in enumerate(og_late_sorted[:20]):
#     plt.plot(og_late_profiles[:,i], linewidth = 3.5, c='tab:blue', alpha = 0.1);
# plt.plot(early_mean, color = 'tab:red', linewidth = 4)
# plt.plot(late_mean, color = 'tab:blue', linewidth = 4)
# plt.fill_between(np.arange(80), early_mean+early_std, early_mean-early_std, color = 'tab:red', alpha = 0.4)
# plt.fill_between(np.arange(80), late_mean+late_std, late_mean-late_std, color = 'tab:blue', alpha =0.4)
# # plt.legend(sorted_list[:20], fontsize = 16, loc = 'upper right')
# # plt.xlabel("Developmental time (iterations)", fontsize = 22)
# # plt.ylabel("Activity of hidden unit", fontsize = 22)
# plt.yticks([0, 20, 40, 60, 80, 100, 120], fontsize = 15)
# plt.xticks(fontsize = 15)
# # plt.axvline(20, color = 'black', linestyle = 'dashed')
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(3)
# plt.tight_layout()
# # plt.savefig(filename, bbox_inches = 'tight')
# plt.show()


# units = og_early_sorted[:5]
# iterations = 60
# intervals = 10
# unit_activity = np.zeros((len(units), iterations, 50, 50))
# for unit in range(len(units)):
#     for i in range(50):
#         for j in range(50):
#             unit_activity[unit, :, i, j] = og_hidden_unit_history[(i, j)][:iterations, units[unit]]

# unit_activity = np.abs(unit_activity - unit_activity[:, 0:1])

# times = np.arange(0, iterations, intervals)
# times[0] = 1

# fig, axes = plt.subplots(len(units), iterations//intervals, figsize=(15, 12))

# # Adjust subplot layout to make space for colorbar
# plt.subplots_adjust(right=0.85)

# for unit in range(len(units)):
#     for j in range(len(times)):
#         im = axes[unit, j].imshow(unit_activity[unit, times[j]], aspect='auto', vmin = 0, vmax = 0.5)
#         # if unit == 0:
#         #     axes[unit, j].set_title(f"t = {times[j]}", fontsize=20)
#         axes[unit, j].set_xticks([])
#         axes[unit, j].set_yticks([])
#         for axis in ['top','bottom','left','right']:
#             axes[unit, j].spines[axis].set_linewidth(5)
#         # if j == 0:
#             # axes[unit, j].set_ylabel(f"{units[unit]}", rotation=0, fontsize=24, va="center", labelpad=50)

# # Create a new axes for the colorbar on the right
# cbar_ax = fig.add_axes([0.88, 0.11, 0.03, 0.13])  # Adjust these values as needed
# cbar = fig.colorbar(im, cax=cbar_ax)
# ticks = cbar.get_ticks()
# if len(ticks) > 2:
#     cbar.set_ticks([ticks[0], ticks[-1]])
# cbar.ax.tick_params(labelsize=15)  # Adjust the size as needed

# # plt.savefig(filename, bbox_inches='tight')
# plt.show()


# # FIG 2B
# og_early_profiles, og_early_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:80], phase = (0, 10))
# og_late_profiles, og_late_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:80], phase = (31, 60))
# early_mean = og_early_profiles[:, og_early_sorted[:20]].mean(axis=1)
# early_std = og_early_profiles[:, og_early_sorted[:20]].std(axis=1)
# late_mean = og_late_profiles[:, og_late_sorted[:20]].mean(axis=1)
# late_std=og_late_profiles[:, og_late_sorted[:20]].std(axis=1)


# fig=plt.figure(figsize = (10, 8))

# for a, i in enumerate(og_early_sorted[:20]):
    
#     plt.plot(og_early_profiles[:,i], linewidth = 3.5, alpha = 0.1, color = 'tab:red');

# for a, i in enumerate(og_late_sorted[:20]):
#     plt.plot(og_late_profiles[:,i], linewidth = 3.5, c='tab:blue', alpha = 0.1);
# plt.plot(early_mean, color = 'tab:red', linewidth = 4)
# plt.plot(late_mean, color = 'tab:blue', linewidth = 4)
# plt.fill_between(np.arange(80), early_mean+early_std, early_mean-early_std, color = 'tab:red', alpha = 0.4)
# plt.fill_between(np.arange(80), late_mean+late_std, late_mean-late_std, color = 'tab:blue', alpha =0.4)
# # plt.legend(sorted_list[:20], fontsize = 16, loc = 'upper right')
# # plt.xlabel("Developmental time (iterations)", fontsize = 22)
# # plt.ylabel("Activity of hidden unit", fontsize = 22)
# plt.yticks([0, 20, 40, 60, 80, 100, 120], fontsize = 15)
# plt.xticks(fontsize = 15)
# # plt.axvline(20, color = 'black', linestyle = 'dashed')
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(3)
# plt.tight_layout()
# # plt.savefig(filename, bbox_inches = 'tight')
# plt.show()


# units = og_early_sorted[:5]
# iterations = 60
# intervals = 10
# unit_activity = np.zeros((len(units), iterations, 50, 50))
# for unit in range(len(units)):
#     for i in range(50):
#         for j in range(50):
#             unit_activity[unit, :, i, j] = og_hidden_unit_history_2[(i, j)][:iterations, units[unit]]

# unit_activity = np.abs(unit_activity - unit_activity[:, 0:1])

# times = np.arange(0, iterations, intervals)
# times[0] = 1

# fig, axes = plt.subplots(len(units), iterations//intervals, figsize=(15, 12))

# # Adjust subplot layout to make space for colorbar
# plt.subplots_adjust(right=0.85)

# for unit in range(len(units)):
#     for j in range(len(times)):
#         im = axes[unit, j].imshow(unit_activity[unit, times[j]], aspect='auto', vmin = 0, vmax = 0.5)
#         # if unit == 0:
#         #     axes[unit, j].set_title(f"t = {times[j]}", fontsize=20)
#         axes[unit, j].set_xticks([])
#         axes[unit, j].set_yticks([])
#         for axis in ['top','bottom','left','right']:
#             axes[unit, j].spines[axis].set_linewidth(5)
#         # if j == 0:
#             # axes[unit, j].set_ylabel(f"{units[unit]}", rotation=0, fontsize=24, va="center", labelpad=50)

# # Create a new axes for the colorbar on the right
# cbar_ax = fig.add_axes([0.88, 0.91, 0.05, 0.01])  # Adjust these values as needed
# cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
# ticks = cbar.get_ticks()
# if len(ticks) > 2:
#     cbar.set_ticks([ticks[0], ticks[-1]])
# cbar.ax.tick_params(labelsize=30)  # Adjust the size as needed
# cbar.ax.xaxis.set_ticks_position('top')
# # plt.savefig(filename, bbox_inches='tight')
# plt.show()





# save = f'./models/{og_model_name}/layer_2_expression_spatial.png'
# visualize_hidden_unit_stills(og_hidden_unit_history_2, units = og_early_sorted[:5], filename = save, dims = (15, 12))



# FIG 3A
# _, og_early_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (0, 10))
# _, og_late_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (31, 60))
# og_late_sorted_exclusive = [i for i in og_late_sorted if i not in og_early_sorted[:20]][:20]
# early_phen, early_loss = progressive_knockout_loss(og_model, og_early_sorted, grid, env, params)
# late_phen, late_loss = progressive_knockout_loss(og_model, og_late_sorted_exclusive, grid, env, params)
# inactive_phen, inactive_loss = progressive_knockout_loss(og_model, og_early_sorted[::-1], grid, env, params)

# # The x locations for the groups
# ind = np.arange(len(early_loss))

# # The width of the bars
# width = 0.3

# fig=plt.figure(figsize=(9, 6))

# # Plotting
# plt.bar(ind, np.log10(early_loss)-np.log10(early_loss[0]), width, label='Early active')
# plt.bar(ind + width, np.log10(late_loss)-np.log10(late_loss[0]), width, label='Late active')
# plt.bar(ind + 2*width, np.log10(inactive_loss)-np.log10(inactive_loss[0]), width, label='Inactive')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# # plt.xlabel('Number of units knocked out', fontsize=15)
# # plt.ylabel('Increase in log loss', fontsize=15)
# plt.xticks(ind + width, labels=[str(i) for i in range(len(early_loss))], fontsize=15)  # Adjust the labels as needed
# plt.yticks(fontsize=15)
# plt.ylim([-0.2, 6.5])
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(3)

# plt.tight_layout()
# # Save and show the plot
# filename = f'./models/{og_model_name}/early_vs_late.png'  # Replace og_model_name with the actual model name
# # plt.savefig(filename, bbox_inches='tight')
# plt.show()


# FIG 3B
# _, og_early_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:60], phase = (0, 10))
# _, og_late_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:60], phase = (31, 60))
# og_late_sorted_exclusive = [i for i in og_late_sorted if i not in og_early_sorted[:20]][:20]
# early_phen, early_loss = progressive_knockout_loss(og_model, og_early_sorted, grid, env, params)
# late_phen, late_loss = progressive_knockout_loss(og_model, og_late_sorted_exclusive, grid, env, params)
# inactive_phen, inactive_loss = progressive_knockout_loss(og_model, og_early_sorted[::-1], grid, env, params)

# # The x locations for the groups
# ind = np.arange(len(early_loss))

# # The width of the bars
# width = 0.3

# fig=plt.figure(figsize=(9, 6))

# plt.bar(ind, np.log10(early_loss)-np.log10(early_loss[0]), width, label='Early active')
# plt.bar(ind + width, np.log10(late_loss)-np.log10(late_loss[0]), width, label='Late active')
# plt.bar(ind + 2*width, np.log10(inactive_loss)-np.log10(inactive_loss[0]), width, label='Inactive')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# # plt.xlabel('Number of units knocked out', fontsize=15)
# # plt.ylabel('Increase in log loss', fontsize=15)
# plt.xticks(ind + width, labels=[str(i) for i in range(len(early_loss))], fontsize=15)  # Adjust the labels as needed
# plt.yticks([0, 1, 2, 3, 4, 5, 6], fontsize=15)
# plt.ylim([-0.2, 6.5])
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(3)

# plt.tight_layout()
# # Save and show the plot
# # filename = f'./models/{og_model_name}/early_vs_late.png'  # Replace og_model_name with the actual model name
# # plt.savefig(filename, bbox_inches='tight')
# plt.show()

# plt.imshow(early_phen)
# plt.axis('off')
# plt.tight_layout()
# plt.savefig(f'./models/{og_model_name}/layer_sequential_knockout.png', bbox_inches = 'tight')



# Quantify how much retraining improved training compared to naive training 
# fig=plt.figure(figsize = (5,6))
# retrained_color = next(plt.gca()._get_lines.prop_cycler)['color']
# naive_color = next(plt.gca()._get_lines.prop_cycler)['color']



# scatter_retrained = plt.scatter(np.arange(8001), np.log10(retrained_losses), alpha=0.1, marker='.', label="Retrained", color=retrained_color)
# scatter_naive = plt.scatter(np.arange(8001), np.log10(nl_losses), alpha=0.1, marker='.', label="Naive", color=naive_color)

# # Manually set the legend with desired properties
# legend_elements = [plt.Line2D([0], [0], linestyle='none', marker='.', alpha=1, markersize=10, color=retrained_color, label='Retrained'),
#                    plt.Line2D([0], [0], linestyle='none', marker='.', alpha=1, markersize=10, color=naive_color, label='Naive')]


# # plt.legend(handles=legend_elements, loc='upper right', fontsize = 15)
# # plt.xlabel("Training epochs", fontsize = 15)
# # plt.ylabel("Log loss", fontsize = 15)
# plt.ylim([-4.3, 0])
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(2)
# plt.xticks([0, 2000, 4000, 6000, 8000], fontsize = 14)
# plt.yticks([-4, -3, -2, -1, 0], fontsize = 14)
# filename = f"./models/{model_name_nl}/retrained_vs_naive_loss.png"
# # plt.savefig(filename, bbox_inches = 'tight')   
# plt.show()
# quantify_retrain_improvement(losses, retrained_losses, difference = True)

# Compare visible channel differences
# og_state_histories = np.zeros((50, 100, 50, 50, 16))
# state_histories_sn = np.zeros((50, 100, 50, 50, 16))
# state_histories_ns = np.zeros((50, 100, 50, 50, 16))
# # Run model many times
# for seed in tqdm(range(og_state_histories.shape[0])):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     og_state_history, _, _ = grid.run(og_model, env, params)
    
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     state_history_sn, _, _ = grid.run(model_sn, env, params)

#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     state_history_ns, _, _ = grid.run(model_ns, env, params)

#     og_state_histories[seed] = og_state_history[:100].clip(0, 1)
#     state_histories_sn[seed] = state_history_sn[:100].clip(0, 1)
#     state_histories_ns[seed] = state_history_ns[:100].clip(0, 1)

# og_state_history = og_state_histories.mean(axis = 0)
# state_history_lb = state_histories_sn.mean(axis = 0)
# state_history_nl = state_histories_ns.mean(axis = 0)

# # FIG 4E
# fig=plt.figure(figsize = (7, 10))
# diff = (state_history_lb[..., :4] - og_state_history[-1, :, :, :4])**2
# diff2 = (state_history_nl[..., :4] - og_state_history[-1, :, :, :4])**2
# plt.plot(diff.sum(axis = (1, 2, 3)), label = "Retrained", linewidth = 4)
# plt.plot(diff2.sum(axis = (1, 2, 3)), label = "Naive", linewidth = 4)
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(3)
# # plt.legend(fontsize = 20)
# # plt.xlabel("Developmental time (iterations)", fontsize = 20)
# # plt.ylabel("Quantified difference to developed gecko", fontsize = 20)
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# filename = f"./models/{model_name_ns}/retrained_vs_naive_visible.png"
# # plt.savefig(filename, bbox_inches = 'tight')
# plt.show()



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
for i in tqdm(range(1, 60)):
    _, og_early_sorted = find_hox_units(og_hidden_unit_history, og_living_cells[:60], phase = (i, i+1))
    _, early_sorted_sl = find_hox_units(hidden_unit_history_sl, living_cells_sl[:60], phase = (i, i+1))
    _, early_sorted_sn = find_hox_units(hidden_unit_history_sn, living_cells_sl[:60], phase = (i, i+1))
    _, early_sorted_sy = find_hox_units(hidden_unit_history_sy, living_cells_sl[:60], phase = (i, i+1))
    _, early_sorted_lb = find_hox_units(hidden_unit_history_lb, living_cells_lb[:60], phase = (i, i+1))
    _, early_sorted_nl = find_hox_units(hidden_unit_history_nl, living_cells_nl[:60], phase = (i, i+1))
    
    # _, og_early_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:60], phase = (i, i+1))
    # _, early_sorted_sl = find_hox_units(hidden_unit_history_sl_2, living_cells_sl[:60], phase = (i, i+1))
    # _, early_sorted_sn = find_hox_units(hidden_unit_history_sn_2, living_cells_sl[:60], phase = (i, i+1))
    # _, early_sorted_sy = find_hox_units(hidden_unit_history_sy_2, living_cells_sl[:60], phase = (i, i+1))
    # _, early_sorted_lb = find_hox_units(hidden_unit_history_lb_2, living_cells_lb[:60], phase = (i, i+1))
    # _, early_sorted_nl = find_hox_units(hidden_unit_history_nl_2, living_cells_nl[:60], phase = (i, i+1))
    
    conserved_early_sl.append(len(set(early_sorted_sl[:20]) & set(og_early_sorted[:20])))
    conserved_early_sn.append(len(set(early_sorted_sn[:20]) & set(og_early_sorted[:20])))
    conserved_early_sy.append(len(set(early_sorted_sy[:20]) & set(og_early_sorted[:20])))
    conserved_early_lb.append(len(set(early_sorted_lb[:20]) & set(og_early_sorted[:20])))
    conserved_early_nl.append(len(set(early_sorted_nl[:20]) & set(og_early_sorted[:20])))

fig = plt.figure(figsize = (15,5))
plt.plot((20,)*60, color = 'black', label = "Full conservation", linewidth = 3)
plt.plot(conserved_early_sl, label = "Gecko – small legs", linewidth = 3)
plt.plot(conserved_early_sn, label = "Snake", linewidth = 3)
plt.plot(conserved_early_sy, label = "Snake – yellow", linewidth = 3)
plt.plot(conserved_early_lb, label = "Ladybug", linewidth = 3)
plt.plot(conserved_early_nl, label = "Naive ladybug control", linewidth = 3)

# plt.xlabel("Developmental time (iterations)", fontsize = 18)
# plt.ylabel("Number of conserved units", fontsize = 18)
plt.xticks(fontsize = 15)
plt.yticks([0, 5, 10, 15, 20], fontsize = 15)
# plt.legend(fontsize = 15)
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
filename = f"conserved_hidden_unit_2_by_iteration.png"
# plt.savefig(filename, bbox_inches = 'tight')
plt.show()


# Compare developmental stages with venn diagram
_, og_early_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:60], phase = (0, 10))
_, og_mid_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:60], phase = (11, 30))
_, og_late_sorted = find_hox_units(og_hidden_unit_history_2, og_living_cells[:60], phase = (31, 60))
filename = f"./models/{og_model_name}/compare_unit_stages_2.png"
compare_developmental_stages(og_early_sorted, og_mid_sorted, og_late_sorted, filename)

# _, og_early_sorted = find_hox_units(hidden_unit_history_lb_2, living_cells_lb[:60], phase = (0, 10))
# _, og_mid_sorted = find_hox_units(hidden_unit_history_lb_2, living_cells_lb[:60], phase = (11, 30))
# _, og_late_sorted = find_hox_units(hidden_unit_history_lb_2, living_cells_lb[:60], phase = (31, 60))
# filename = f"./models/{model_name_lb}/compare_unit_stages_2.png"
# compare_developmental_stages(og_early_sorted, og_mid_sorted, og_late_sorted, filename)


# for i in range(len(og_early_sorted)):
#     conserved_early_sl.append(len(set(early_sorted_sl[:i]) & set(og_early_sorted[:i])))
#     conserved_late_sl.append(len(set(late_sorted_sl[:i]) & set(og_late_sorted[:i])))
    
#     conserved_early_sn.append(len(set(early_sorted_sn[:i]) & set(og_early_sorted[:i])))
#     conserved_late_sn.append(len(set(late_sorted_sn[:i]) & set(og_early_sorted[:i])))
    
#     conserved_early_sy.append(len(set(early_sorted_sy[:i]) & set(og_early_sorted[:i])))
#     conserved_late_sy.append(len(set(late_sorted_sy[:i]) & set(og_early_sorted[:i])))
    
#     conserved_early_lb.append(len(set(early_sorted_lb[:i]) & set(og_early_sorted[:i])))
#     conserved_late_lb.append(len(set(late_sorted_lb[:i]) & set(og_early_sorted[:i])))
#     conserved_mid_lb.append(len(set(mid_sorted_lb[:i]) & set(og_mid_sorted[:i])))
#     conserved_late_lb.append(len(set(late_sorted_lb[:i]) & set(og_late_sorted[:i])))
    
#     conserved_early_nl.append(len(set(early_sorted_nl[:i]) & set(og_early_sorted[:i])))
#     conserved_mid_nl.append(len(set(mid_sorted_nl[:i]) & set(og_mid_sorted[:i])))
#     conserved_late_nl.append(len(set(late_sorted_nl[:i]) & set(og_late_sorted[:i])))
    

    


# save = f'./models/{og_model_name}/hox.mp4'
# visualize_single_hidden_unit(og_hidden_unit_history, units = og_early_sorted[:20], filename = save)



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
#     axs[j//4, j%4].set_xlabel("Time", fontsize  = 15)
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


