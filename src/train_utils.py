import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from helpers.helpers import *
from helpers.visualizer import *
import copy

    
    
def train(model: nn.Module, grid, n_epochs: int, model_name: str, batch_size: int = 8, pool_size: int = 1024, 
          regenerate: bool = True, env: torch.Tensor = None, dynamic_env: bool = False, modulate: bool = False,
          angle_target: bool = False):
    """ 
    Train with pool. 
    Set regenerate = True to train regeneration capabilities. 
    Set env = None to train without environment. 
    Set dynamic_env = True to train with dynamic environment. 
    Set modulate = True to train with environment modulated by a channel.
    Set angle_target = True to train with targets rotated. 
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    model = model.to(device)
    env = env.to(device)
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr = 2e-3, eps = 1e-7)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2000], gamma = 0.1)
    
    grid_size = grid.grid_size
    
    # Initialise pool
    seed = grid.init_seed(grid_size).to(device)
    pool = seed.repeat(pool_size, 1, 1, 1)
    
    model_losses = []
    
    # Initialise progress bar
    pbar = tqdm(total = n_epochs+1)
    
    if env is not None:
        repeated_env = env.repeat(batch_size, 1, 1, 1)
    
    # Initialize history of pool losses to 0
    pool_losses = torch.zeros(pool_size)
    
    for epoch in range(n_epochs+1):
        
        optimizer.zero_grad()
        
        # Sample from pool
        
        # Select indices from pool
        indices = np.random.randint(pool_size, size = batch_size)
        x0 = pool[indices]
        
        # Sort indices of losses with highest loss first
        loss_rank = pool_losses[indices].argsort(descending = True)
        x0 = x0[loss_rank]
        
        # Reseed highest loss sample
        x0[0] = seed
        
        if regenerate == True:
            
            # Disrupt pattern for samples with lowest 3 loss
            for i in range(1,4):
                mask = create_circular_mask(grid_size).to(device)
                x0[-i]*=mask
        
        # Train with sample   
        iterations = np.random.randint(64, 97)
        # Run model
        x = x0
        
        modulate_vals = torch.zeros(batch_size, 1, grid_size, grid_size)
        
        if env is not None:
            
            
            if angle_target == True:
                
                # Randomly initialize angles
                angles = list(np.random.uniform(0, 360, batch_size))
                
                # Rotate images
                target_imgs = rotate_image(model.target, angles).to(device)
                
                angled_env = torch.zeros(batch_size, model.env_channels, grid_size, grid_size).to(device)
                
                # Angle each environment in the batch based on initialised angles
                for i in range(batch_size):
                    angled_env[i] = grid.add_env(env, type = 'directional', channel = 0, angle = angles[i]-45, 
                                                    center = (grid_size/2, grid_size/2))
                    # angled_env[i] = grid.add_env(env, type = 'linear', channel = 0, angle = angles[i]+45)
                
                new_env = copy.deepcopy(angled_env)
            
            else:
                target_imgs = model.target.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
                new_env = copy.deepcopy(repeated_env)
            

            
            for t in range(iterations):
                
                # If environment is to be modulated by a channel, modulate the given repeated environment
                if modulate == True:
                    new_env = modulate_vals*repeated_env
                
                # Get new environment
                if dynamic_env == True:
                    
                    # If we have angled targets, get new environment from angled target environments
                    if angle_target == True:
                        new_env = grid.get_env(t, angled_env, type = 'phase')
                    
                    # If we have normal targets, get new environment from given environment
                    else:
                        new_env = grid.get_env(t, repeated_env, type = 'phase')

                x, new_env = model.update(x, new_env)
                modulate_vals = state_to_image(x)[..., 4].unsqueeze(1)
                
        else:
            for _ in range(iterations):      
                x, _ = model.update(x)

        # Pixel-wise L2 loss
        transformed_img = state_to_image(x)[..., :4]
        
        pool_loss = ((transformed_img - target_imgs)**2).mean(dim = [1, 2, 3])
        loss = pool_loss.mean()

        # Update pool losses
        pool_losses[indices] = pool_loss.detach()
            
        # Update model loss
        model_losses.append(loss.item())
        
        # Compute gradients
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
        if epoch%5 == 0:
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
def create_circular_mask(grid_size: int, center_radius: float = 8.0) -> torch.Tensor:
    """
    Returns masked out grid

    :param grid: n, 16, 28, 28
    :type grid: Torch tensor
    :type grid_size: int
    :param center_radius: Radius of where center of mask is located, defaults to 8
    :type center_radius: float, optional
    :return: Mask
    :rtype: Torch tensor
    """
    # Create mask
    center = np.random.randint(grid_size//2 - center_radius, grid_size//2 + center_radius, size = 2)
    mask_radius = np.random.randint(3, 8)
    
    Y, X = np.ogrid[:grid_size, :grid_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= mask_radius
    
    return torch.tensor(1-mask)

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
