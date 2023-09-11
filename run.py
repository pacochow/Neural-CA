import torch   
from src.standard_ca import *
from src.env_ca import *
from src.grid import Grid
from helpers.visualizer import *
from src.params import ObjectView
import pickle
from src.analysis_utils import *
from helpers.figures import * 

# Load model
model_name = "env_circle_16_1"
model = torch.load(f"./models/{model_name}/final_weights.pt", map_location = torch.device('cpu'))

grid_coordinates = [(x, y) for x in range(50) for y in range(50)]

# # Load hidden unit histories
# filename = f'./models/{model_name}/hidden_unit_history.pkl'
# with open(filename, 'rb') as fp:
#     hidden_unit_history = pickle.load(fp)
# living_cells = np.load(f"./models/{model_name}/living_cells.npy")

# _, sorted = find_hox_units(hidden_unit_history, living_cells[:60], phase = (0, 20))

params = {
       
# Run params
'model_channels': model.model_channels, 
'env_channels': model.env_channels,
'grid_size': 50,
'iterations': 100,                  # Number of iterations in animation
'angle': 0.0,                       # Perceiving angle
'env_angle': 0,                    # Environment angle
'dynamic_env': False,               # Run with moving environment
'dynamic_env_type': 'fade out',    # Type of moving environment    
'destroy': False,                   # Whether pattern is disrupted mid animation
'destroy_type': 0,                  # Type of pattern disruption
'seed': None,                       # Coordinates of seed
'vis_env': False,                   # Visualize environment in animation
'vis_hidden': False,                 # Visualize hidden unit activity throughout run
'modulate_env': False,               # Use alpha channel to modulate environment
'hidden_loc': grid_coordinates, # Location of where to visualize hidden unit activity
'knockout': False,                   # Whether hidden unit is fixed
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
# env = grid.add_env(env, "linear", 0, angle = params.env_angle)
env = grid.add_env(env, "circle", 0, center = (params.grid_size/2, params.grid_size/2))
# env = grid.add_env(env, "directional", 0, angle = params.env_angle, center = (params.grid_size/2, params.grid_size/2))
# env = grid.add_env(env, 'none')


# Run model
state_history, env_history, hidden_history = grid.run(model, env, params, manual = False)

# Save last image
# filename = f'./models/{model_name}/grown.png'
# plt.imshow(state_history[-1, :, :, :4])
# plt.axis('off')
# plt.savefig(filename, bbox_inches = 'tight')
# plt.show()

# Create animation
# filename = f'./models/{model_name}/run.mp4'
# create_animation(state_history, env_history, filename, params)

# Create animation stills
filename = f'./models/{model_name}/run.png'
create_stills(state_history, env_history, filename, params, intervals = 3, format = (3, 8), dims = (16, 6))

# Visualise all channels stills
# filename = f'./models/{model_name}/all_channels_run.png'
# visualise_hidden_channels_stills(state_history, filename, params, intervals = 5, format = (13,13), dims = (26, 26))

# Visualize hidden units
# filename = f'./models/{model_name}/hidden_units.mp4'
# visualize_hidden_units(state_history, hidden_history, filename, params)


# target = rotate_image(model.target, params.env_angle+45)
# loss = ((state_history[-1, :, :, :4]-target[0].numpy())**2).mean()

# Visualise all channels
# filename = f'./models/{model_name}/all_channels_run.mp4'
# visualize_all_channels(state_history, filename, model.model_channels, params)

# Visualize seed losses at different seed positions
# filename = f"./models/{model_name}/diff_seed_losses.png"
# visualize_seed_losses(model, grid, filename, params = params, env = env)

# Create progress animation
# states = load_progress_states(model_name, grid, params, env)
# filename = f'./models/{model_name}/progress.mp4'
# create_progress_animation(states, filename, params)

# filename = f'./models/{model_name}/progress.png'
# create_progress_stills(states, filename)


# Save rotations
# filename = f'./models/{model_name}/rotation.png'
# fig, axes = plt.subplots(1, 5, figsize=(10, 7)) 
# angles = np.arange(-45, 170, 45)
# for i, ax in enumerate(axes):
#     params.env_angle = angles[i]
#     env = grid.add_env(env, "directional", 0, angle = params.env_angle, center = (params.grid_size/2, params.grid_size/2))
#     state_history, env_history, hidden_history = grid.run(model, env, params, manual = False)
#     if params.vis_env == True:
#         ax.imshow(env_history[-1, 0]+env_history[-1, 1], cmap = create_colormap(), vmin = 0, vmax = 1)
#     ax.imshow(state_history[-1, :, :, :4])
#     ax.axis('off')  
# plt.savefig(filename, bbox_inches = 'tight')
# plt.show()


# Save all hidden units 
# grid_dict = {coordinate: None for coordinate in grid_coordinates}
# for i in range(len(grid_coordinates)):
#     grid_dict[grid_coordinates[i]] = hidden_history[i]

# filename = f'./models/{model_name}/hidden_unit_history.pkl'
# with open(filename, 'wb') as fp:
#     pickle.dump(grid_dict, fp)

# # Compute number of living cells at each iteration
# living_cells = np.zeros((state_history.shape[0], 1))
# for i in range(state_history.shape[0]):
#     living_cells[i, 0] = (state_history[i, :, :, 3]>0.1).sum()
    
# np.save(f'./models/{model_name}/living_cells.npy', living_cells)

# save = f'./models/{model_name}/hox.mp4'
# visualize_single_hidden_unit(grid_dict, units = np.arange(20), filename = save)


# iterations = 100
# unit_activity = np.zeros((1, iterations, 50, 50))
# for unit in range(1):
#     for i in range(50):
#         for j in range(50):
#             unit_activity[0, :, i, j] = grid_dict[(i, j)][:iterations, 7]

