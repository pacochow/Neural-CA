import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from IPython.display import clear_output
from helpers.helpers import *
from src.pruning import *
from tqdm import tqdm



def create_animation(states: np.ndarray, envs: np.ndarray, filename: str, params):

    fps = params.iterations/params.nSeconds

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)[...,:4]
    a = states[0]
    b = envs[0]

    cm = create_colormap()
    if params.vis_env == True:
        im2 = plt.imshow(b[0], cmap = cm, interpolation = 'gaussian', aspect = 'auto', vmin = 0, vmax=1)
        if b.shape[0] > 1:
            im3 = plt.imshow(b.sum(axis = 0)/(b.sum(axis = 0).max()), cmap = cm, interpolation = 'gaussian', aspect = 'auto', vmin = 0, vmax=1)
        
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
    plt.axis('off')

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.95, '', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=20)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        if params.vis_env == True:
            im2.set_array(envs[i, 0])
            if b.shape[0] > 1:
                im3.set_array(envs[i].sum(axis = 0)/(b.sum(axis = 0).max()))
        
        im.set_array(states[i])

        # Update iteration number text
        iteration_text.set_text(f't = {i}')

        return [im, iteration_text]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = params.iterations,
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')


    
def visualize_hidden_units(states: np.ndarray, hidden_states: np.ndarray, filename: str, params):

    hidden_states = hidden_states.reshape(len(params.hidden_loc), params.iterations, 20, 20)
    
    fps = params.iterations/params.nSeconds

    nrows = 1
    ncols = 1+hidden_states.shape[0]

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,8*nrows))  
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)[...,:4]
    hidden_states = hidden_states.clip(0, 1)
    
    # Create an array to hold your image objects
    ims = []
    
    titles = [params.hidden_loc[i] for i in range(hidden_states.shape[0])]
    
    a = states[0]  # the initial state for each animation
    im = axs[0].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
    axs[0].tick_params(labelsize = 20)
    ims.append(im)
    
    
    for j in range(hidden_states.shape[0]):
        b = hidden_states[j, 0]
        im2 = axs[j+1].imshow(b, interpolation = 'none', aspect = 'auto', vmin = 0, vmax = 0.5)
        axs[j+1].set_title(titles[j], fontsize = 40)
        axs[j+1].axis('off')
        ims.append(im2)
        
    # Create a text field to display the iteration number
    iter_text = axs[0].text(0.5, 1, '', fontsize=40, ha='center', va='bottom', transform=axs[0].transAxes)  # Adjust the position as needed
    
    
    
    
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        ims[0].set_array(states[i])  # update each animation
        
        for j in range(hidden_states.shape[0]):
            ims[j+1].set_array(hidden_states[j, i])
            
        # Update the iteration number text field
        iter_text.set_text(f't = {i}')
        
            

        return ims
    
    anim = animation.FuncAnimation(
        fig, 
        animate_func, 
        frames = params.iterations,
        interval = 1000 / fps, # in ms
        )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')
    
def visualize_single_hidden_unit(hidden_unit_history: dict, units: list, filename: str):

    iterations = 100
    unit_activity = np.zeros((len(units), iterations, 50, 50))
    for unit in range(len(units)):
        for i in range(50):
            for j in range(50):
                unit_activity[unit, :, i, j] = hidden_unit_history[(i, j)][:iterations, units[unit]]

    
    fps = iterations/10
    max = 15
    ncols = len(units) if len(units)<max else max
    nrows = len(units)//ncols+1 if len(units)%ncols!=0 else len(units)//ncols
    

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,8*nrows))  # 4 subplots for 4 animations


    # Create an array to hold your image objects
    ims = []

    for j in range(nrows*ncols):  # loop over your new dimension
        if j < len(units):
            a = unit_activity[j, 0]
            im = axs[j//ncols, j%ncols].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax = 0.1)
            
            axs[j//ncols, j%ncols].set_title(units[j], fontsize = 40)
            ims.append(im)
        axs[j//ncols, j%ncols].axis('off')
    
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        for j in range(len(units)):  # loop over your new dimension
            ims[j].set_array(unit_activity[j, i])
            

        return ims

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = iterations,
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')
    

def load_progress_states(model_name: str, grid, params, env = None):
    """
    Create array of models run at each saved epoch
    """
    
    # Initialise states
    states = np.zeros((4, params.iterations, params.grid_size, params.grid_size, 4))

    
    # Loop over all saved epochs and run model
    saved_epochs = [100, 500, 1000, 4000]
    for i in range(len(saved_epochs)):
        model = torch.load(f"./models/{model_name}/{saved_epochs[i]}.pt", map_location = torch.device('cpu'))
        model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        full_states, _, _ = grid.run(model, env, params)
        states[i] = full_states[...,:4]
        
    return states

def create_progress_animation(states: np.ndarray, filename: str, params):
    fps = params.iterations/params.nSeconds

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(32,8))  # 4 subplots for 4 animations
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)[...,:4]
    
    # Titles
    titles = [100, 500, 1000, 4000]
    
    # Create an array to hold your image objects
    ims = []
    
    cm = create_colormap()
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
        frames = params.iterations,
        interval = 1000 / fps, # in ms
        )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Progress animation done!')


def visualize_all_channels(states: np.ndarray, filename: str, n_channels: int, params):

    fps = params.iterations/params.nSeconds

    ncols = 5
    n_plots = n_channels-4+1
    nrows = n_plots//5+1
    

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,8*nrows))  # 4 subplots for 4 animations

    # Clip values between 0 and 1
    states = states.clip(0, 1)

    # Create an array to hold your image objects
    ims = []
    
    titles = list(np.arange(5, n_channels+1, 1))
    titles.insert(0, "1-4")

    for j in range(nrows*ncols):  # loop over your new dimension

        if j < n_plots:
            if j == 0:            
                a = states[0][..., :4]  # the initial state for each animation
            else:
                a = states[0][..., j+3]
            im = axs[j//ncols, j%ncols].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
            
            axs[j//ncols, j%ncols].set_title(titles[j], fontsize = 40)
            ims.append(im)
        axs[j//ncols, j%ncols].axis('off')

    
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        for j in range(n_plots):  # loop over your new dimension
            if j == 0:
                ims[j].set_array(states[i][..., :4])  # update each animation
            else:
                ims[j].set_array(states[i][..., j+3])
            

        return ims

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = params.iterations,
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')

def plot_log_loss(ax, epoch, loss):
    ax.set_title("Loss history", fontsize = 40)
    ax.set_xlabel("Time", fontsize =30)
    ax.set_ylabel("Log loss", fontsize = 30)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.scatter(list(range(epoch+1)), np.log10(loss), marker = '.', alpha = 0.3)

def visualize_batch(axs, x0, x):
    x0 = state_to_image(x0)[..., :4].detach().cpu().numpy().clip(0, 1)
    x = state_to_image(x)[..., :4].detach().cpu().numpy().clip(0, 1)

    # Remove axes for all subplots
    for ax in axs.ravel():
        ax.axis('off')

    for i in range(x0.shape[0]):
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
    gs1 = gridspec.GridSpecFromSubplotSpec(2, x0.shape[0], subplot_spec=gs[1:])
    axs = np.empty((2,x0.shape[0]), dtype=object)
    for i in range(2):
        for j in range(x0.shape[0]):
            axs[i, j] = fig.add_subplot(gs1[i, j])
    
    visualize_batch(axs, x0, x)  # Images plot in the second subplot

    clear_output(wait=True)  # Clear the previous plots
    plt.tight_layout()
    plt.show()
    

def save_loss_plot(n_epochs: int, model_losses: list, filename: str):
    
    plt.scatter(list(range(n_epochs)), np.log10(model_losses), marker = '.', alpha = 0.3)
    plt.xlabel("Time", fontsize =12)
    plt.ylabel("Log loss", fontsize = 12)
    plt.title("Loss history", fontsize = 18)
    plt.tight_layout()
    plt.savefig(filename)
    
def visualize_seed_losses(model_name: str, grid, filename, params, env = None):
    
    model = torch.load(f"./models/{model_name}/final_weights.pt")
    losses = np.zeros((model.grid_size, model.grid_size))
    
    for i in tqdm(range(model.grid_size)):
        for j in range(model.grid_size):
            
            states, _, _ = grid.run(model, env, params)
            states = states[...,:4]
             # Compute loss
            losses[i, j] = ((states[-1]-model.target.numpy())**2).mean()
    
    plt.imshow(np.log10(losses), vmax = 0)
    plt.title(f"Log loss at different seed positions\n{model_name}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



    
    
def visualize_pruning(model_name: str, grid, filename: str, params, env = None):
    
    model = torch.load(f"./models/{model_name}/final_weights.pt")
    
    grid_size = model.grid_size
    
    states = np.zeros((6, params.iterations, grid_size, grid_size, 4))

    # Run model without pruning
    full_states, _, _ = grid.run(model, env, params)
    states[0] = full_states[...,:4]
    
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
        full_states, _, _ = grid.run(pruned_model, env, params)
        states[i+1] = full_states[...,:4]
        
    fps = params.iterations/params.nSeconds

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
        frames = params.iterations,
        interval = 1000 / fps, # in ms
        )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Pruning animation done!')


def visualize_pruning_by_channel(model: nn.Module, grid, filename: str, params, env: torch.Tensor = None):
    
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

    # Create an array to hold your image objects
    ims = []
    
    titles = list(np.arange(5, n_channels+1, 1))
    titles.insert(0, "Without pruning")

    for j in range(nrows*ncols):  # loop over your new dimension
        if j < n_plots:
            a = states[0, 0]  # the initial state for each animation
        
            im = axs[j//ncols, j%ncols].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
            
            axs[j//ncols, j%ncols].set_title(titles[j], fontsize = 40)
            ims.append(im)
        axs[j//ncols, j%ncols].axis('off')      
    
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        for j in range(n_plots):  # loop over your new dimension
            ims[j].set_array(states[j, i])  # update each animation
            

        return ims

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = params.iterations,
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Pruning animation done!')
    

def visualize_unit_effect_loss(model: nn.Module, grid, env, params, filename: str):
    _, loss = prune_by_unit(model, grid, env, params)
    units = np.arange(model.hidden_units)
    plt.scatter(units, np.log10(loss))
    plt.xlabel("Unit")
    plt.ylabel("Log loss after ablation")
    plt.savefig(filename)
    plt.show()
    
    return loss

def visualize_unit_effect(model: nn.Module, grid, env, params, prune_units, filename: str):
    phenotypes, _ = prune_by_unit(model, grid, env, params, prune_units = prune_units)
    phenotypes = phenotypes.clip(0, 1)
    
    ncols = 10
    nrows = 10
    
    # Create a grid of subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 8*nrows))

    for i, ax in enumerate(axs.flatten()):

        ax.imshow(phenotypes[i])
        ax.axis('off')  # Hide axis

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    


