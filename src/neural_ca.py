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

class CAModel(nn.Module):
    """

    Input: n,48,grid_size,grid_size
    Output: n,16,grid_size,grid_size
    """
    def __init__(self, target, grid_size, num_channels = 16, fire_rate = 0.5):
        super(CAModel, self).__init__()
        
        self.target = torch.tensor(target)
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.fire_rate = fire_rate
    
        self.input_dim = self.num_channels*3
        
        # Update network
        self.conv1 = nn.Conv2d(self.input_dim, 128, 1)
        self.mid = nn.Conv2d(128, 256, 1) 
        self.conv2 = nn.Conv2d(256, self.num_channels, 1)
        self.relu = nn.ReLU()
        self.conv2.weight.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.mid(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        return out

    def perceive(self, state_grid, angle = 0.0):
        """ Compute perception vectors

        :param state_grid: n, 16, grid_size, grid_size
        :type state_grid: torch tensor
        :return: 1, 48, grid_size, grid_size
        :rtype: torch tensor
        """    
        
        # Sobel filters
        sobel_x = torch.tensor(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0).float()
        sobel_y = sobel_x.T
        
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        dx = c*sobel_x-s*sobel_y
        dy = s*sobel_x+c*sobel_y
        
        # Stack sobel filters 16 times
        dx = dx.view(1, 1, 3, 3).repeat(self.num_channels, 1, 1, 1)
        dy = dy.view(1, 1, 3, 3).repeat(self.num_channels, 1, 1, 1)
        
        # Convolve sobel filters
        with torch.no_grad():
            grad_x = F.conv2d(state_grid, dx, padding = 1, groups = self.num_channels)
            grad_y = F.conv2d(state_grid, dy, padding = 1, groups = self.num_channels)
        
        # Concatenate
        perception_grid = torch.concat((state_grid, grad_x, grad_y), dim = 1)
        
        return perception_grid

    def stochastic_update(self, state_grid, ds_grid):
        """ Apply stochastic mask so that all cells do not update together.

        :param state_grid: n,16,grid_size,grid_size
        :type state_grid: torch tensor
        :param ds_grid: n,16,grid_size,grid_size
        :type ds_grid: torch tensor
        :return: 1,16,grid_size,grid_size
        :rtype: torch tensor
        """
        
        
        size = ds_grid.shape[-1]
        
        # Random mask 
        rand_mask = (torch.rand(1, 1, size,size)<self.fire_rate)
        
        # Apply same random mask to every channel of same position
        rand_mask = rand_mask.repeat(ds_grid.shape[0], 1, 1, 1)
        
        # Zero updates for cells that are masked out
        ds_grid = ds_grid*rand_mask
        return state_grid+ds_grid

    def alive_masking(self, state_grid):
        """ Returns mask for dead cells
        
        :param state_grid: n,16,grid_size,grid_size
        :type state_grid: torch tensor
        :return: n,16,grid_size,grid_size
        :rtype: torch tensor
        """
        
        # Max pool to find cells with alive neighbours
        
        alpha = state_grid[:,3,:,:]
        with torch.no_grad():
            alive = F.max_pool2d(alpha, kernel_size = 3, stride = 1, padding = 1) > 0.1

        return alive.unsqueeze(1)
    
    def update(self, state_grid, angle = 0.0):
        
        # Pre update life mask
        pre_mask = self.alive_masking(state_grid)
        
        ds_grid = torch.zeros(self.num_channels, 1, self.grid_size, self.grid_size)
        
        # Perceive
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
    