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

class Standard_CA(nn.Module):
    """

    Input: n,48,grid_size,grid_size
    Output: n,16,grid_size,grid_size
    """
    def __init__(self, target, grid_size, num_channels = 16, fire_rate = 0.5):
        super(Standard_CA, self).__init__()
        
        self.target = torch.tensor(target)
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.fire_rate = fire_rate
    
        self.input_dim = self.num_channels*3
        
        # Update network
        self.conv1 = nn.Conv2d(self.input_dim, 128, 1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.conv2 = nn.Conv2d(128, self.num_channels, 1)
        self.relu = nn.ReLU()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        return out

    def perceive(self, state_grid, angle = 0.0):
        """ 
        Compute perception vectors

        :param state_grid: n, 16, grid_size, grid_size
        :type state_grid: torch tensor
        :return: n, 48, grid_size, grid_size
        :rtype: torch tensor
        """    
        
        identify = np.float32([0, 1, 0])
        identify = torch.tensor(np.outer(identify, identify))
        dx = torch.tensor(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0)  # Sobel filter
        dy = dx.T
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        kernel_stack = torch.stack([identify, c*dx-s*dy, s*dx+c*dy], 0)

        kernel = kernel_stack.unsqueeze(1)
        kernel = kernel.repeat(self.num_channels, 1, 1, 1).float()

        state_repeated = state_grid.repeat_interleave(kernel_stack.shape[0],dim = 1)
        perception_grid = F.conv2d(state_repeated, kernel, padding=1, groups=kernel.size(0))

        return perception_grid

    def stochastic_update(self, state_grid, ds_grid):
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
        rand_mask = rand_mask.repeat(1, self.num_channels, 1, 1)
        
        # Zero updates for cells that are masked out
        ds_grid = ds_grid*rand_mask.float()
        return state_grid+ds_grid

    def alive_masking(self, state_grid):
        """ 
        Returns mask for dead cells
        
        :param state_grid: n,16,grid_size,grid_size
        :type state_grid: torch tensor
        :return: n,1,grid_size,grid_size
        :rtype: torch tensor
        """
        
        # Max pool to find cells with alive neighbours
        
        alpha = state_grid[:,3,:,:]
        with torch.no_grad():
            alive = F.max_pool2d(alpha, kernel_size = 3, stride = 1, padding = 1) > 0.1
        return alive.unsqueeze(1)
    
    def update(self, state_grid, angle = 0.0, step_size = 1.0):
        
        # Pre update life mask
        pre_mask = self.alive_masking(state_grid)
        
        # Perceive
        perception_grid = self.perceive(state_grid, angle)

        # Apply update rule to all cells
        ds_grid = self.forward(perception_grid)*step_size

        # Stochastic update mask
        state_grid = self.stochastic_update(state_grid, ds_grid)
        
        # Post update life mask
        post_mask = self.alive_masking(state_grid)
        
        life_mask = pre_mask & post_mask
        
        # Zero out dead cells
        state_grid = life_mask.float()*state_grid
        
        return state_grid
    