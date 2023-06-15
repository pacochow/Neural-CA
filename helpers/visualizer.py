import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from IPython.display import clear_output
from helpers.helpers import *
from src.pruning import *
from tqdm import tqdm


def create_animation(states: np.ndarray, envs: np.ndarray, iterations: int, nSeconds: int, filename: str, vis_env = False):

    fps = iterations/nSeconds

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)
    
    a = states[0]
    b = envs[0]
    cm = create_colormap()
    if vis_env == True:
        im2 = plt.imshow(b, cmap = cm, interpolation = 'gaussian', aspect = 'auto', vmin = 0, vmax=1)
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
    plt.axis('off')

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        if vis_env == True:
            im2.set_array(envs[i])
        im.set_array(states[i])

        return [im]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = iterations,
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')
    

def load_progress_states(model_name: str, grid, iterations: int, grid_size: int, angle: float = 0.0, env = None):
    """
    Create array of models run at each saved epoch
    """
    
    # Initialise states
    states = np.zeros((4, iterations, grid_size, grid_size, 4))
    
    # Loop over all saved epochs and run model
    saved_epochs = [100, 500, 1000, 4000]
    for i in range(len(saved_epochs)):
        model = torch.load(f"./models/{model_name}/{saved_epochs[i]}.pt")
        states[i] = grid.run(model, iterations, destroy_type = 0, destroy = True, angle = angle, env = env)
    
    return states

def create_progress_animation(states: np.ndarray, iterations: int, nSeconds: int, filename: str):
    fps = iterations/nSeconds

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(32,8))  # 4 subplots for 4 animations
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)
    
    # Titles
    titles = [100, 500, 1000, 4000]
    
    # Create an array to hold your image objects
    ims = []
    for j in range(4):  # loop over your new dimension
        a = states[j, 0]  # the initial state for each animation
        im = axs[j].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
        axs[j].axis('off')
        axs[j].set_title(titles[j], fontsize = 40)
        ims.append(im)
    
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        for j in range(4):  # loop over your new dimension
            ims[j].set_array(states[j, i])  # update each animation

        return ims

    anim = animation.FuncAnimation(
        fig, 
        animate_func, 
        frames = iterations,
        interval = 1000 / fps, # in ms
        )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Progress animation done!')

def plot_log_loss(ax, epoch, loss):
    ax.set_title("Loss history", fontsize = 40)
    ax.set_xlabel("Iterations", fontsize =30)
    ax.set_ylabel("Log loss", fontsize = 30)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.scatter(list(range(epoch+1)), np.log10(loss), marker = '.', alpha = 0.3)

def visualize_batch(axs, x0, x):
    x0 = state_to_image(x0).detach().numpy().clip(0, 1)
    x = state_to_image(x).detach().numpy().clip(0, 1)

    # Remove axes for all subplots
    for ax in axs.ravel():
        ax.axis('off')

    for i in range(8):
        axs[0, i].imshow(x0[i])  
        axs[1, i].imshow(x[i])  
        
    # Add labels
    axs[0, 0].set_title('Before', loc='left', fontsize = 30)
    axs[1, 0].set_title('After', loc='left', fontsize = 30)

def visualize_training(epoch, loss, x0, x):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1]) 

    ax0 = plt.subplot(gs[0])
    plot_log_loss(ax0, epoch, loss)  # Log loss plot in the first subplot

    # Create subplots for images
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=gs[1:])
    axs = np.empty((2,8), dtype=object)
    for i in range(2):
        for j in range(8):
            axs[i, j] = fig.add_subplot(gs1[i, j])
    
    visualize_batch(axs, x0, x)  # Images plot in the second subplot

    clear_output(wait=True)  # Clear the previous plots
    plt.tight_layout()
    plt.show()
    

def save_loss_plot(n_epochs: int, model_losses: list, filename: str):
    
    plt.scatter(list(range(n_epochs)), np.log10(model_losses), marker = '.', alpha = 0.3)
    plt.xlabel("Iterations", fontsize =12)
    plt.ylabel("Log loss", fontsize = 12)
    plt.title("Loss history", fontsize = 18)
    plt.tight_layout()
    plt.savefig(filename)
    
def visualize_seed_losses(model_name: str, grid, iterations, filename, destroy: bool = True, angle: float = 0.0, env = None):
    
    model = torch.load(f"./models/{model_name}/final_weights.pt")
    losses = np.zeros((model.grid_size, model.grid_size))
    
    for i in tqdm(range(model.grid_size)):
        for j in range(model.grid_size):
            
            states = grid.run(model, iterations, destroy = destroy, angle = angle, env = env, seed = (i, j))
            
             # Compute loss
            losses[i, j] = ((states[-1]-model.target.numpy())**2).mean()
    
    plt.imshow(np.log10(losses), vmax = 0)
    plt.title(f"Log loss at different seed positions\n{model_name}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def comparing_pruning_losses(model1: str, grid1, env1, model2: str, grid2, env2, filename: str, iterations: int, angle: float = 0.0):
    
    percents, loss1 = compute_pruning_losses(model1, grid1, iterations, angle, env1)
    percents, loss2 = compute_pruning_losses(model2, grid2, iterations, angle, env2)
    plt.scatter(percents, np.log10(loss1))
    plt.scatter(percents, np.log10(loss2))
    plt.xlabel("Pruned percentage (%)", fontsize =12)
    plt.ylabel("Log loss", fontsize = 12)
    plt.title("Loss after pruning", fontsize = 18)
    plt.legend([model1, model2])
    plt.tight_layout()
    plt.savefig(filename)
    
    
def visualize_pruning(model_name: str, grid, iterations: int, nSeconds: int, filename: str, angle: float = 0.0, env = None):
    
    model = torch.load(f"./models/{model_name}/final_weights.pt")
    
    grid_size = model.grid_size
    
    states = np.zeros((6, iterations, grid_size, grid_size, 4))

    # Run model without pruning
    states[0] = grid.run(model, iterations, destroy = True, angle = angle, env = env)
    
    # Run model after pruning each percent
    percents = [5, 10, 15, 20, 25]
    pruned_percents = [0]
    for i in range(len(percents)):
        
        # Prune model
        model_size, pruned_size, pruned_model = prune_by_percent(model, percent=percents[i])
        # Compute pruned percentages
        pruned_percentage = (model_size - pruned_size)*100/model_size
        pruned_percents.append(pruned_percentage)
        
        # Run model
        states[i+1] = grid.run(pruned_model, iterations, destroy = True, angle = angle, env = env)
        
    fps = iterations/nSeconds

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(48,8))  # 6 subplots for 6 animations
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)
    
    # Titles
    titles = [f"{pruned_percents[i]:.2f}%" for i in range(len(pruned_percents))]
    
    # Create an array to hold your image objects
    ims = []
    for j in range(6):  # loop over your new dimension
        a = states[j, 0]  # the initial state for each animation
        im = axs[j].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
        axs[j].axis('off')
        axs[j].set_title(titles[j], fontsize = 40)
        ims.append(im)
        
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        for j in range(6):  # loop over your new dimension
            ims[j].set_array(states[j, i])  # update each animation

        return ims

    anim = animation.FuncAnimation(
        fig, 
        animate_func, 
        frames = iterations,
        interval = 1000 / fps, # in ms
        )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Pruning animation done!')

    