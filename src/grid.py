import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from helpers.helpers import *
from src.train_utils import create_block_mask

class Grid:
    
    def __init__(self, grid_size, model_channels, env_channels = 0):
        self.grid_size = grid_size
        self.model_channels = model_channels
        self.env_channels = env_channels
        self.num_channels = model_channels+env_channels
    
    
    def init_seed(self, grid_size):
        """ 
        Initialise seed

        :return: n, 16 channels, grid_size, grid_size
        :rtype: torch tensor
        """    
        
        self.grid_size = grid_size
        
        # Initialise seed to zeros everywhere
        seed = torch.zeros(1, self.model_channels, grid_size, grid_size)
        
        # Set seed in the centre to be equal to 1 for all channels except RGB
        seed[:, 3:, grid_size//2, grid_size//2] = 1
        
        return seed
    
    
    def init_pool(self, grid_size, pool_size = 1024):
        """ 
        Initialise pool

        :return: Pool of seed
        :rtype: pool_size, 16, grid_size, grid_size
        """
        
        
        seed = self.init_seed(grid_size)
        pool = seed.repeat(pool_size, 1, 1, 1)
        
        return pool
    
    def run(self, model, iterations, destroy_type, destroy = True, angle = 0.0):
        """ 
        Run model and save state history
        """
    
        state_grid = self.init_seed(self.grid_size)
        state_history = np.zeros((iterations, self.grid_size, self.grid_size, 4))
        
        for t in range(iterations):
            
            with torch.no_grad():
                # Visualize state
                transformed_img = state_to_image(state_grid)[0]
                state_history[t] = transformed_img.detach().numpy().reshape(self.grid_size, self.grid_size, 4)
                
                # Update step
                state_grid = model.update(state_grid, angle = angle)
                
                # Disrupt pattern
                if destroy == True and t == iterations//2:
                    state_grid = create_block_mask(state_grid, self.grid_size, type = destroy_type)
                
        return state_history

    
    def init_env(self, env_channels, type, angle = 0, center = (0,0)):
        
        env = torch.zeros(env_channels, self.grid_size, self.grid_size)
        
        if type == "linear":
            env[0] = create_angular_gradient(self.grid_size, angle)
        elif type == "circle":
            env[0] = create_circular_gradient(self.grid_size, center)

            
        return env
    

    
    