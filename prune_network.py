from helpers.helpers import *
import torch   
from src.standard_ca import *
from src.grid import *
from src.pruning import *
from helpers.visualizer import create_animation
from helpers.visualizer import visualize_pruning

iterations = 400
nSeconds = 10
angle = 0.0


# Load model
model_name = 'standard_16'
model = torch.load(f"./models/{model_name}/final_weights.pt")


# Initialise grid
grid_size = model.grid_size
grid = Grid(grid_size, model.model_channels)

# Initialise environment
env = None
# env = grid.init_env(model.env_channels)
# env = grid.add_env(env, "circle", 0)

# Visualise progress animation
filename = f"./models/{model_name}/pruned_visualization.mp4"
visualize_pruning(model_name, grid, iterations, nSeconds, filename = filename, angle = angle, env = env)



# # Prune model
# percent = 23
# model_size, pruned_size, pruned_model = prune_by_percent(model, percent=percent)

# # Run model
# state_history = grid.run(pruned_model, iterations, destroy_type = 0, destroy = True, angle = angle, env = env)

# # Create animation
# filename = f"./models/{model_name}/pruned_run.mp4"
# create_animation(state_history, iterations, nSeconds, filename)
