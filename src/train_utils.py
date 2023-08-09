import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from helpers.helpers import *
from helpers.visualizer import *
import copy

    
    
def train(model: nn.Module, model_name: str, grid, env: torch.Tensor, params):
    """ 
    Train with pool. 
    Set regenerate = True to train regeneration capabilities. 
    Set env = None to train without environment. 
    Set dynamic_env = True to train with dynamic environment. 
    Set modulate = True to train with environment modulated by a channel.
    Set angle_target = True to train with targets rotated. 
    """
    device = params.device
        
    model = model.to(device)
    env = env.to(device)
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr = params.lr, eps = 1e-7, weight_decay = params.decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = params.milestones, gamma = params.gamma)
    
    grid_size = params.grid_size
    
    # Initialise pool
    seed = grid.init_seed(grid_size).to(device)
    pool = seed.repeat(params.pool_size, 1, 1, 1)
    
    model_losses = []
    
    # Initialise progress bar
    pbar = tqdm(total = params.n_epochs+1)
        
    
    # Initialize history of pool losses to 0
    pool_losses = torch.zeros(params.pool_size).to(device)
    
    for epoch in range(params.n_epochs+1):
        
        optimizer.zero_grad()

        
        # Sample from pool
        
        # Select indices from pool
        indices = np.random.randint(params.pool_size, size = params.batch_size)
        x0 = pool[indices]
        
        # Sort indices of losses with highest loss first
        loss_rank = pool_losses[indices].argsort(descending = True)
        x0 = x0[loss_rank]
        
        # Reseed highest loss sample
        x0[0] = seed
    
            
        # Disrupt pattern for samples with lowest 3 loss
        for i in range(1,4):
            mask = create_circular_mask(grid_size).to(device)
            x0[-i]*=mask
        
        # Train with sample   
        iterations = np.random.randint(params.num_steps[0], params.num_steps[1])
        # Run model
        x = x0
        
        modulate_vals = torch.zeros(params.batch_size, 1, grid_size, grid_size, device = device)
        
        if env is not None:
            
            
            if params.angle_target == True:
                
                # Randomly initialize angles
                angles = list(np.random.uniform(0, 360, params.batch_size))
                
                # Rotate images
                target_imgs = rotate_image(model.target.cpu(), angles).to(device)
                
                repeated_env = torch.zeros(params.batch_size, model.env_channels, grid_size, grid_size).to(device)
                
                # Angle each environment in the batch based on initialised angles
                for i in range(params.batch_size):
                    repeated_env[i] = grid.add_env(env, type = 'directional', channel = 0, angle = angles[i]-45, 
                                                    center = (grid_size/2, grid_size/2))
                    # repeated_env[i] = grid.add_env(env, type = 'linear', channel = 0, angle = angles[i]+45)
                
                
            
            else:
                target_imgs = model.target.unsqueeze(0).repeat(params.batch_size, 1, 1, 1).to(device)
                repeated_env = env.repeat(params.batch_size, 1, 1, 1)

            
            new_env = copy.deepcopy(repeated_env)
            
            for t in range(iterations):
            
                
                # Get new environment
                if params.dynamic_env == True:
                    
                    new_env = grid.get_env(t, repeated_env, type = params.dynamic_env_type)
                    
                # Modulate the environment so that environment is only visible where there are cells
                if params.modulate_env == True:
                    new_env = modulate_vals*new_env
                    
                x, new_env = model.update(x, new_env)
                
                # Modulate with transparency channel
                modulate_vals = state_to_image(x)[..., 3].unsqueeze(1)
                
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
            if param.requires_grad:
                param.grad.data = param.grad.data / (torch.norm(param.grad.data) + 1e-8)
   
        
        optimizer.step()
        scheduler.step()
        
        # Replace pool with output
        pool[indices] = x.detach()
        
        # Visualise progress
        pbar.set_description("Loss: %.4f" % np.log10(loss.item()))
        pbar.update()
        # if epoch%1 == 0:
        #     visualize_training(epoch, model_losses, torch.tensor(x0), x)
           
        # Save progress 
        if epoch in [100, 500, 1000, 4000]:
            torch.save(model, f'models/{model_name}/{epoch}.pt')
        
    pbar.close()
    
    # Save model 
    torch.save(model, f"./models/{model_name}/final_weights.pt")
    torch.save(model_losses, f"./models/{model_name}/losses.pt")

    # Save loss plot
    save_loss_plot(params.n_epochs+1, model_losses, f"./models/{model_name}/loss.png")
    
        
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
