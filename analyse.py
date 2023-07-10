import pickle 
import numpy as np
import matplotlib.pyplot as plt
from helpers.visualizer import visualize_single_hidden_unit, cluster_hidden_units
import torch
from src.analysis import *

# Load model
model_name = "modulated_angled_env_directional_16_2_400"
model = torch.load(f"./models/{model_name}/final_weights.pt", map_location = torch.device('cpu'))

# Load hidden unit histories
filename = f'./models/{model_name}/hidden_unit_history.pkl'
with open(filename, 'rb') as fp:
    hidden_unit_history = pickle.load(fp)
    

development_profiles, early_sorted = find_hox_units(hidden_unit_history, early = True)

filename = f'./models/{model_name}/early_hox.png'
plot_expression_profiles(development_profiles, early_sorted, filename)



# save = f'./models/{model_name}/hox.mp4'
# visualize_single_hidden_unit(hidden_unit_history, units = early_sorted[:20], filename = save)

# Perform PCA on hidden units and cluster
# clusters, x_pca = cluster_hidden_units(model)
# clusters = dict(zip(np.arange(400), clusters))
