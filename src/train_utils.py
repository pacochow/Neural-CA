import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from helpers.helpers import *
from helpers.visualizer import *
import copy

    
    
def train(model: nn.Module, grid, n_epochs: int, model_name: str, batch_size: int = 8, pool_size: int = 1024, 
          regenerate: bool = True, env: torch.Tensor = None, dynamic_env: bool = False, modulate: bool = False):
    """ 
    Train with pool. 
    Set regenerate = True to train regeneration capabilities. 
    Set env = None to train without environment. 
    Set dynamic_env = True to train with dynamic environment. 
    """
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr = 2e-3, eps = 1e-7)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2000], gamma = 0.1)
    
    grid_size = grid.grid_size
    
    # Initialise pool
    seed = grid.init_seed(grid_size).numpy()
    pool = np.repeat(seed, pool_size, axis = 0)
    
    model_losses = []
    
    pbar = tqdm(total = n_epochs+1)
    
    # Make env have dimensions (n, env_channels, grid_size, grid_size)
    if env is not None:
        env = env.repeat(batch_size, 1, 1, 1)
        
    for epoch in range(n_epochs+1):
        
        optimizer.zero_grad()
        
        # Sample from pool
        
        # Select indices from pool
        indices = np.random.randint(pool_size, size = batch_size)
        x0 = pool[indices]
        
        # Calculate loss of samples
        sample_images = state_to_image(torch.tensor(x0))[..., :4]
        losses = ((sample_images - model.target)**2).mean(dim = [1, 2, 3])
        
        # Sort indices of losses with highest loss first
        loss_rank = losses.numpy().argsort()[::-1]
        x0 = x0[loss_rank]
        
        # Reseed highest loss sample
        x0[0] = seed
        
        if regenerate == True:
            
            # Disrupt pattern for samples with lowest 3 loss
            for i in range(1,4):
                x0[-i] = create_circular_mask(x0[-i], grid_size)
            
        # Train with sample   
        iterations = np.random.randint(64, 97)
        # Run model
        x = torch.Tensor(x0)
        
        modulate_vals = torch.zeros(batch_size, 1, grid_size, grid_size)
        
        if env is not None:
            
            # Make a copy of the original environment
            new_env = copy.deepcopy(env)
            for t in range(iterations):
                
                if modulate == True:
                    new_env = modulate_vals*env
                
                # Get new environment
                if dynamic_env == True:
                    new_env = grid.get_env(t, env, type = 'phase')
                x, new_env = model.update(x, new_env)
                modulate_vals = state_to_image(x)[..., 4].unsqueeze(1)
                
        else:
            for _ in range(iterations):      
                x, _ = model.update(x)

        # Pixel-wise L2 loss
        transformed_img = state_to_image(x)[..., :4]
        
        loss = ((transformed_img - model.target)**2).mean()
        
            
        model_losses.append(loss.item())
        loss.backward()

        # Normalize gradients
        for param in model.parameters():
            param.grad.data = param.grad.data / (torch.norm(param.grad.data) + 1e-8)
   
        
        optimizer.step()
        scheduler.step()
        
        # Replace pool with output
        pool[indices] = x.detach()
        
        # Visualise progress
        pbar.set_description("Loss: %.4f" % np.log10(loss.item()))
        pbar.update()
        if epoch%100 == 0:
            visualize_training(epoch, model_losses, torch.tensor(x0), x)
           
        # Save progress 
        if epoch in [100, 500, 1000, 4000]:
            torch.save(model, f'models/{model_name}/{epoch}.pt')
        
    pbar.close()
    
    # Save model 
    torch.save(model, f"./models/{model_name}/final_weights.pt")
    torch.save(model_losses, f"./models/{model_name}/losses.pt")

    # Save loss plot
    save_loss_plot(n_epochs+1, model_losses, f"./models/{model_name}/loss.png")
    
        
# Pattern disruption
def create_circular_mask(grid, grid_size: int, center_radius: float = 8.0):
    """
    Returns masked out grid

    :param grid: n, 16, 28, 28
    :type grid: Numpy array
    :type grid_size: int
    :param center_radius: Radius of where center of mask is located, defaults to 8
    :type center_radius: float, optional
    :return: Masked out grid
    :rtype: Numpy array
    """
    # Create mask
    center = np.random.randint(grid_size//2 - center_radius, grid_size//2 + center_radius, size = 2)
    mask_radius = np.random.randint(3, 8)
    
    Y, X = np.ogrid[:grid_size, :grid_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= mask_radius
    
    # Mask out grid
    grid = grid*(1-mask)
    return grid

def create_block_mask(grid, grid_size: int, type: int = 0, mask_size: float = 4.0):
    """
    Returns masked out grid
    """
    
    
    if type == 0:
        grid[:, :, grid_size//2:] = 0
    
    if type == 1:
        grid[:, :, :grid_size//2] = 0
    
    if type == 2:
        grid[:, :, :, grid_size//2:] = 0
        
    if type == 3:
        grid[:, :, :, :grid_size//2] = 0
        
    if type == 4:
        grid[:, :, grid_size//2-mask_size:grid_size//2+mask_size, grid_size//2-mask_size:grid_size//2+mask_size] = 0

    return grid
