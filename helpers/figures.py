import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.pruning import *
from helpers.helpers import create_colormap

def create_stills(states: np.ndarray, envs: np.ndarray, filename: str, params, intervals: int, format: tuple, dims: tuple = (15, 7)):
    
    """
    Create developmental period visualisation plots
    """
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)[...,:4]

    times = np.arange(0, len(states), intervals)
    
    if format[0] == 1:
        # Plotting
        fig, axes = plt.subplots(1, format[1], figsize=(15, 5)) 

        for i, ax in enumerate(axes):
            if params.vis_env == True:
                ax.imshow(envs[times[i], 0], cmap = create_colormap(), vmin = 0, vmax = 1)
            ax.imshow(states[times[i]])
            # ax.set_title(f"t = {times[i]}")
            ax.axis('off')  # To turn off axis numbers
    else:
        fig, axes = plt.subplots(format[0], format[1], figsize = dims)
        
        for i in range(format[0]*format[1]):
            if params.vis_env == True:
                axes[i//format[1], i%format[1]].imshow(envs[times[i], 0], cmap = create_colormap(), vmin = 0, vmax = 1)
            axes[i//format[1], i%format[1]].imshow(states[times[i]])
            # axes[i//format[1], i%format[1]].set_title(f"t = {times[i]}", fontsize = 22)
            # ax.axis('off')  # To turn off axis numbers
            axes[i//format[1], i%format[1]].set_xticks([])
            axes[i//format[1], i%format[1]].set_yticks([])
            for axis in ['top','bottom','left','right']:
                axes[i//format[1], i%format[1]].spines[axis].set_linewidth(3)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()
    
    
def visualize_pruning_stills(model: nn.Module, grid, filename: str, params, env = None):
    
    """
    Visualise pruning outcomes
    """
    
    
    model.params = params
    model.knockout = params.knockout
    model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    grid_size = params.grid_size
    
    states = np.zeros((6, 1, grid_size, grid_size, 4))

    # Run model without pruning
    full_states, _, _ = grid.run(model, env, params)
    states[0] = full_states[-1, :, :, :4]
    
    # Run model after pruning each percent
    percents = [5, 10, 15, 20, 25]
    pruned_percents = [0]
    for i in range(len(percents)):
        
        # Prune model
        model_size, pruned_size, pruned_model = prune_by_percent(model, percent=percents[i])

        pruned_model.params = params
        pruned_model.knockout = params.knockout
        pruned_model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Compute pruned percentages
        pruned_percentage = (model_size - pruned_size)*100/model_size
        pruned_percents.append(pruned_percentage)
        
        # Run model
        full_states, _, _ = grid.run(pruned_model, env, params)
        states[i+1] = full_states[-1, :, :, :4]
        

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(48,8))  # 6 subplots for 6 animations
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)
    
    # Titles
    titles = [f"{pruned_percents[i]:.2f}%" for i in range(len(pruned_percents))]
    
    for j in range(6):  # loop over your new dimension
        a = states[j, -1]  # the initial state for each animation
        im = axs[j].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
        axs[j].axis('off')
        axs[j].set_title(titles[j], fontsize = 50)
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



def visualize_hidden_unit_stills(hidden_unit_history: dict, units: str, filename: str, dims: tuple):
    """
    Visualise hidden unit activity profiles
    """
    
    iterations = 60
    intervals = 10
    unit_activity = np.zeros((len(units), iterations, 50, 50))
    for unit in range(len(units)):
        for i in range(50):
            for j in range(50):
                unit_activity[unit, :, i, j] = hidden_unit_history[(i, j)][:iterations, units[unit]]

    unit_activity = np.abs(unit_activity - unit_activity[:, 0:1])

    times = np.arange(0, iterations, intervals)
    times[0] = 1
    
    fig, axes = plt.subplots(len(units), iterations//intervals, figsize=dims)
    
    # Adjust subplot layout to make space for colorbar
    plt.subplots_adjust(right=0.85)

    for unit in range(len(units)):
        for j in range(len(times)):
            im = axes[unit, j].imshow(unit_activity[unit, times[j]], aspect='auto', vmin = 0, vmax = 0.5)
            if unit == 0:
                axes[unit, j].set_title(f"t = {times[j]}", fontsize=20)
            axes[unit, j].set_xticks([])
            axes[unit, j].set_yticks([])

            if j == 0:
                axes[unit, j].set_ylabel(f"{units[unit]}", rotation=0, fontsize=24, va="center", labelpad=50)

    # Create a new axes for the colorbar on the right
    cbar_ax = fig.add_axes([0.88, 0.11, 0.03, 0.13])  # Adjust these values as needed
    cbar = fig.colorbar(im, cax=cbar_ax)
    ticks = cbar.get_ticks()
    if len(ticks) > 2:
        cbar.set_ticks([ticks[0], ticks[-1]])
    cbar.ax.tick_params(labelsize=15)  # Adjust the size as needed

    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def visualize_pruning_by_channel_stills(model: nn.Module, grid, filename: str, params, env: torch.Tensor = None):
    
    """
    Visualise outcome of development after pruning
    """
    
    fps = params.iterations/params.nSeconds
    n_channels = model.model_channels
    ncols = 5
    n_plots = n_channels-4+1
    nrows = n_plots//ncols+1
    
    full_states, _, _ = grid.run(model, env, params)
    states = np.zeros((n_plots, params.iterations, grid.grid_size, grid.grid_size, 4))
    states[0] = full_states[..., :4]
    
    
    for i in range(4, n_channels):
        pruned_model = prune_by_channel(model, i, enhance = params.enhance)
        full_pruned_states, _, _ = grid.run(pruned_model, env, params)
        states[i-3] = full_pruned_states[...,:4]



    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,8*nrows))  # 4 subplots for 4 animations

    # Clip values between 0 and 1
    states = states.clip(0, 1)
    
    titles = list(np.arange(5, n_channels+1, 1))
    titles.insert(0, "Without pruning")

    for j in range(nrows*ncols):  # loop over your new dimension
        if j < n_plots:
        
            im = axs[j//ncols, j%ncols].imshow(states[j, -1], interpolation='none', aspect='auto', vmin=0, vmax=1)
            
            axs[j//ncols, j%ncols].set_title(titles[j], fontsize = 45)
        axs[j//ncols, j%ncols].axis('off')      
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()