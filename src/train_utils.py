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

    
    
def train(model, grid, n_epochs, batch_size = 8, pool_size = 1024, regenerate = True, env = None):
    """ 
    Train with pool
    """
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr = 2e-3, eps = 1e-7)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2000], gamma = 0.1)
    
    grid_size = grid.grid_size
    
    # Initialise pool
    seed = grid.init_seed(grid_size).numpy()
    pool = np.repeat(seed, pool_size, axis = 0)
    # pool = SamplePool(x=np.repeat(seed, pool_size, 0))
    
    model_losses = []
    
    pbar = tqdm(total = n_epochs+1)
    for epoch in range(n_epochs+1):
        
        optimizer.zero_grad()
        
        # Sample from pool
        
        # Select indices from pool
        # batch = pool.sample(batch_size)
        # x0 = batch.x
        indices = np.random.randint(pool_size, size = batch_size)
        x0 = pool[indices]
        
        # Calculate loss of samples
        sample_images = state_to_image(torch.tensor(x0))
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
        if env is not None:
            for _ in range(iterations):
                x = model.update(x, env)
        else:
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
        # batch.x[:] = x.detach().numpy()
        # batch.commit()
        pool[indices] = x.detach()
        
        # Visualise progress
        pbar.set_description("Loss: %.4f" % np.log10(loss.item()))
        pbar.update()
        if epoch%100 == 0:
            visualize_training(epoch, model_losses, torch.tensor(x0), x)
        
    pbar.close()
    
    return model_losses
        
class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

        
# Pattern disruption
def create_circular_mask(grid, grid_size, center_radius = 8):
    """
    Returns masked out grid

    :param grid: n, 16, 28, 28
    :type grid: Numpy array
    :type grid_size: int
    :param center_radius: Radius of where center of mask is located, defaults to 8
    :type center_radius: int, optional
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
