import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from IPython.display import clear_output
from helpers.helpers import *
from src.pruning import *
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        if params.vis_env == True:
            im2.set_array(envs[i, 0])
            if b.shape[0] > 1:
                im3.set_array(envs[i].sum(axis = 0)/(b.sum(axis = 0).max()))
        im.set_array(states[i])

        return [im]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = params.iterations,
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')
    
def visualize_hidden_units(states: np.ndarray, hidden_states: np.ndarray, filename: str, params):

    fps = params.iterations/params.nSeconds

    nrows = 1
    ncols = 1+hidden_states.shape[0]

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,8*nrows))  # 2 subplots for 2 animations
    
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
    
    # Create colorbar for right plot
    # cbar = fig.colorbar(im2, ax=axs[1])
    # cbar.ax.tick_params(labelsize=14)  # Change label size here
    
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        ims[0].set_array(states[i])  # update each animation
        
        for j in range(hidden_states.shape[0]):
            ims[j+1].set_array(hidden_states[j, i])
            

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

    iterations = 150
    unit_activity = np.zeros((len(units), iterations, 50, 50))
    for unit in range(len(units)):
        for i in range(50):
            for j in range(50):
                unit_activity[unit, :, i, j] = hidden_unit_history[(i, j)][:iterations, units[unit]]


    fps = iterations/10

    ncols = len(units) if len(units)<15 else 15
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
    envs = np.zeros((4, params.iterations, params.grid_size, params.grid_size))
    
    # Loop over all saved epochs and run model
    saved_epochs = [100, 500, 1000, 4000]
    for i in range(len(saved_epochs)):
        model = torch.load(f"./models/{model_name}/{saved_epochs[i]}.pt")
        full_states, envs[i], _ = grid.run(model, env, params)
        states[i] = full_states[...,:4]
        
    return states, envs

def create_progress_animation(states: np.ndarray, envs: np.ndarray, iterations: int, nSeconds: int, filename: str, vis_env: bool = False):
    fps = iterations/nSeconds

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(32,8))  # 4 subplots for 4 animations
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)[...,:4]
    
    # Titles
    titles = [100, 500, 1000, 4000]
    
    # Create an array to hold your image objects
    ims = []
    ims2 = []
    
    cm = create_colormap()
    for j in range(4):  # loop over your new dimension
        a = states[j, 0]  # the initial state for each animation
        b = envs[j, 0]
        if vis_env == True:
            im2 = axs[j].imshow(b, cmap = cm, interpolation = 'gaussian', aspect = 'auto', vmin = 0, vmax = 1)
        im = axs[j].imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
        axs[j].axis('off')
        axs[j].set_title(titles[j], fontsize = 40)
        ims.append(im)
        ims2.append(im2)
    
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        for j in range(4):  # loop over your new dimension
            ims2[j].set_array(envs[j, i])
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
    ax.set_xlabel("Iterations", fontsize =30)
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
    plt.xlabel("Iterations", fontsize =12)
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


def comparing_pruning_losses(model1: str, grid1, env1, model2: str, grid2, env2, filename: str, params):
    
    percents, loss1 = compute_pruning_losses(model1, grid1, params.iterations, params.angle, env1)
    percents, loss2 = compute_pruning_losses(model2, grid2, params.iterations, params.angle, env2)
    plt.scatter(percents, np.log10(loss1))
    plt.scatter(percents, np.log10(loss2))
    plt.xlabel("Pruned percentage (%)", fontsize =12)
    plt.ylabel("Log loss", fontsize = 12)
    plt.title("Loss after pruning", fontsize = 18)
    plt.legend([model1, model2])
    plt.tight_layout()
    plt.savefig(filename)
    
    
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
    
def plot_parameter_sizes(model_name: str, filename: str):

    model = torch.load(f"./models/{model_name}/final_weights.pt")
    
    # Get parameters
    params = [x.data for x in model.parameters()]
    
    # Average over all weights for each channel to give array of length num_channels
    weights = np.abs(params[2][:, :, 0, 0].mean(dim = 1).numpy())

    # Categories and numbers
    categories = list(range(1, model.model_channels+1))

    # Bar chart
    x = np.arange(len(categories))  # Label locations

    fig, ax = plt.subplots()
    rects = ax.bar(x[:model.model_channels], weights[:model.model_channels], label='Model channels')

    # Special label for the last category
    if model.env_channels > 0 and model.env_output == True:
        rects_last = ax.bar(x[-1], weights[-1], label='Environment channel')

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Parameter size', fontsize = 13)
    ax.set_xlabel('Channels', fontsize = 13)
    ax.set_title('Mean size of parameters for each channel', fontsize = 16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation = 0)
    ax.legend()

    plt.savefig(filename)
    plt.show()

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
    
    # Create a 5x5 grid of subplots
    fig, axs = plt.subplots(10, 10, figsize=(40, 40))

    for i, ax in enumerate(axs.flatten()):

        ax.imshow(phenotypes[i])
        ax.axis('off')  # Hide axis

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    
def cluster_hidden_units(model: nn.Module, filename: str = None):
    
    """
    Perform PCA on weights of each hidden unit and cluster 
    """

    params = [i for i in model.parameters()]
    X = params[0].squeeze(-2, -1).detach().numpy()


    # assume X is your data

    # Apply PCA for dimensionality reduction (optional)
    pca = PCA(n_components=30)  # or another number less than 54
    X_pca = pca.fit_transform(X)

    # Create a kmeans model
    kmeans = KMeans(n_clusters=2); # you can change the number of clusters
    kmeans.fit(X_pca)

    # Get the cluster assignments for each data point
    clusters = kmeans.predict(X_pca)


    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Visualization of clustered data', fontweight='bold')
    plt.colorbar()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

