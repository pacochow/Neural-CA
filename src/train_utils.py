import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from helpers.helpers import *
from src import grid

    
    
def train(model, grid, n_epochs, sample_size = 8, pool_size = 64, regenerate = True):
    """ Train with pool
    """
    
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2000], gamma = 0.1)
    grid_size = grid.grid_size
    
    # Initialise pool
    seed = grid.init_seed(grid_size)
    pool = grid.init_pool(grid_size, pool_size)
    
    model_losses = []
    
    for epoch in tqdm(range(n_epochs)):
        
        optimizer.zero_grad()
        
        # Sample from pool
        
        # Select indices from pool
        indices = np.random.randint(pool_size, size = sample_size)
        sample = pool[indices]
        
        # Calculate loss of samples
        sample_images = state_to_image(sample)
        losses = ((sample_images - model.target)**2).mean(dim = [1, 2, 3])
        
        # Find index with highest loss
        index = int(losses.argmax())
        
        # Reseed highest loss sample
        sample[index] = seed
        
        if regenerate == True and epoch >= n_epochs//2:
            
            # Find indices with lowest loss
            low_indices = list(losses.topk(3, largest = False)[1])
            
            # Disrupt pattern
            for i in low_indices:
                sample[i] = create_circular_mask(sample[i], grid_size)
            
            # if epoch%200 == 0:
            #     plt.imshow(sample_images[[low_indices[0]]][:,:,0])
            #     plt.show()
            
        # Train with sample   
        iterations = np.random.randint(64, 97)
        
        # Run model
        for t in range(iterations):            
            sample = model.update(sample)
            
            
        # Pixel-wise L2 loss
        transformed_img = state_to_image(sample)
        
        loss = ((transformed_img - model.target)**2).mean()
        # Visualise progress
        if epoch%100 == 0:
            plt.imshow(transformed_img.detach().numpy()[0])
            plt.show()
            print(loss)
            
        model_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Replace pool with output
        pool[indices] = sample.detach()
        
    return model_losses
        
        
# Pattern disruption
def create_circular_mask(grid, grid_size, center_radius = 8):
    """Returns masked out grid

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
