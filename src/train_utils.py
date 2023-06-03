import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from helpers.helpers import *
from helpers.visualizer import visualize_training
from src import grid

    
    
def train(model, grid, n_epochs, sample_size = 8, pool_size = 1024, regenerate = True):
    """ 
    Train with pool
    """
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr = 2e-3, eps = 1e-7)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2000], gamma = 0.1)
    
    grid_size = grid.grid_size
    
    # Initialise pool
    seed = grid.init_seed(grid_size)
    pool = grid.init_pool(grid_size, pool_size)
    
    model_losses = []
    
    for epoch in tqdm(range(n_epochs+1)):
        
        optimizer.zero_grad()
        
        # Sample from pool
        
        # Select indices from pool
        indices = np.random.randint(pool_size, size = sample_size)
        x0 = pool[indices]
        
        # Calculate loss of samples
        sample_images = state_to_image(x0)
        losses = ((sample_images - model.target)**2).mean(dim = [1, 2, 3])
        
        # Sort indices of losses with lowest loss first
        loss_indices = list(losses.argsort(descending=False).numpy())
        
        # Reseed highest loss sample
        x0[loss_indices[-1]] = seed
        
        if regenerate == True and epoch >= 2000:
            
            # Disrupt pattern for samples with lowest 3 loss
            for i in loss_indices[:3]:
                x0[i] = create_circular_mask(x0[i], grid_size)
            
            # if epoch%200 == 0:
            #     plt.imshow(sample_images[[low_indices[0]]][:,:,0])
            #     plt.show()
            
        # Train with sample   
        iterations = np.random.randint(64, 97)
        
        # Run model
        x = x0
        for _ in range(iterations):            
            x = model.update(x)
            
        # Pixel-wise L2 loss
        transformed_img = state_to_image(x)
        
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
        if epoch%100 == 0:
            visualize_training(epoch, model_losses, x0, x)
        #     plt.imshow(transformed_img[0].detach().numpy())
        #     plt.show()
        #     print(loss)
        
        
# Pattern disruption
def create_circular_mask(grid, grid_size, center_radius = 8):
    """
    Returns masked out grid

    :param grid: n, 16, 28, 28
    :type grid: Torch tensor
    :type grid_size: int
    :param center_radius: Radius of where center of mask is located, defaults to 8
    :type center_radius: int, optional
    :return: Masked out grid
    :rtype: Torch tensor
    """
    # Create mask
    center = np.random.randint(grid_size//2 - center_radius, grid_size//2 + center_radius, size = 2)
    mask_radius = np.random.randint(3, 8)
    
    Y, X = np.ogrid[:grid_size, :grid_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= mask_radius
    
    # Mask out grid
    grid = grid*(1-mask)
    return grid.float()

def create_block_mask(grid, grid_size, type, mask_size = 4):
    """Returns masked out grid
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
