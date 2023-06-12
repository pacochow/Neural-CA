import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.helpers import *

class Env_CA(nn.Module):
    """
    Input: n, 48, grid_size, grid_size
    Output: n, 16, grid_size, grid_size
    """
    def __init__(self, target: np.ndarray, grid_size: int, model_channels = 16, env_channels = 1, fire_rate = 0.5):
        super(Env_CA, self).__init__()
        
        self.target = torch.tensor(target)
        self.grid_size = grid_size
        self.model_channels = model_channels
        self.env_channels = env_channels
        
        if env_channels > 0:
            self.env = True
        else:
            self.env = False
        self.fire_rate = fire_rate
    
        self.num_channels = model_channels + env_channels
        self.input_dim = self.num_channels*3
        
        # Update network
        self.conv1 = nn.Conv2d(self.input_dim, 128, 1)
        self.conv2 = nn.Conv2d(128, self.model_channels, 1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.relu = nn.ReLU()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        return out

    def perceive(self, state_grid: torch.Tensor, angle = 0.0):
        """ 
        Compute perception vectors

        :param state_grid: n, num_channels, grid_size, grid_size
        :type state_grid: torch tensor
        :return: n, input_dim, grid_size, grid_size
        :rtype: torch tensor
        """    
        
        # Identity filter
        identify = np.float32([0, 1, 0])
        identify = torch.tensor(np.outer(identify, identify))
        
        # Sobel filters
        dx = torch.tensor(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0)  # Sobel filter
        dy = dx.T
        
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        
        # Stack filters together
        kernel_stack = torch.stack([identify, c*dx-s*dy, s*dx+c*dy], 0)
        kernel = kernel_stack.unsqueeze(1)
        
        # Repeat kernels to form num_channels x 1 x 3 x 3 filter
        kernel = kernel.repeat(self.num_channels, 1, 1, 1).float()

        state_repeated = state_grid.repeat_interleave(kernel_stack.shape[0],dim = 1)
        
        # Perform convolution
        perception_grid = F.conv2d(state_repeated, kernel, padding=1, groups=kernel.size(0))

        return perception_grid

    def stochastic_update(self, state_grid: torch.Tensor, ds_grid: torch.Tensor):
        """ 
        Apply stochastic mask so that all cells do not update together.

        :param state_grid: n,16,grid_size,grid_size
        :type state_grid: torch tensor
        :param ds_grid: n,16,grid_size,grid_size
        :type ds_grid: torch tensor
        :return: n,16,grid_size,grid_size
        :rtype: torch tensor
        """
        
        
        size = ds_grid.shape[-1]
        
        # Random mask 
        rand_mask = (torch.rand(ds_grid.shape[0], 1, size,size)<=self.fire_rate)
        
        # Apply same random mask to every channel of same position
        rand_mask = rand_mask.repeat(1, self.model_channels, 1, 1)
        
        # Zero updates for cells that are masked out
        ds_grid = ds_grid*rand_mask
        return state_grid+ds_grid

    def alive_masking(self, state_grid: torch.Tensor):
        """ Returns mask for dead cells
        
        :param state_grid: n,17,grid_size,grid_size
        :type state_grid: torch tensor
        :return: n,1,grid_size,grid_size
        :rtype: torch tensor
        """
        
        # Max pool to find cells with alive neighbours
        
        alpha = state_grid[:,3,:,:]
        with torch.no_grad():
            alive = F.max_pool2d(alpha, kernel_size = 3, stride = 1, padding = 1) > 0.1

        return alive.unsqueeze(1)
    
    def update(self, state_grid: torch.Tensor, env = None, angle = 0.0):
        
        # Pre update life mask
        pre_mask = self.alive_masking(state_grid)
        
        # Perceive
        if self.env == True:
            env = env.repeat(state_grid.shape[0], 1, 1, 1)      
            full_grid = torch.cat([state_grid,env], dim = 1)
            perception_grid = self.perceive(full_grid, angle)
        else: 
            perception_grid = self.perceive(state_grid, angle)
        
        # Apply update rule to all cells
        ds_grid = self.forward(perception_grid)

        # Stochastic update mask
        state_grid = self.stochastic_update(state_grid, ds_grid)
        
        # Post update life mask
        post_mask = self.alive_masking(state_grid)
        
        life_mask = pre_mask & post_mask
        
        # Zero out dead cells
        state_grid = life_mask*state_grid
        
        return state_grid
    