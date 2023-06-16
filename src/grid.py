import numpy as np
import torch
from helpers.helpers import *
from src.train_utils import create_block_mask
import matplotlib.pyplot as plt

class Grid:
    
    def __init__(self, grid_size: int, model_channels: int, env_channels: int = 0):
        self.grid_size = grid_size
        self.model_channels = model_channels
        self.env_channels = env_channels
        self.num_channels = model_channels+env_channels
    
    
    def init_seed(self, grid_size: int, center: tuple = None) -> torch.Tensor:
        """ 
        Initialise seed. Set center to True to initialise seed location in the center.
        Set center to any tuple to change seed location.

        :return: 1, 16 channels, grid_size, grid_size
        :rtype: Torch tensor array 
        """    
        
        self.grid_size = grid_size
        
        # Initialise seed to zeros everywhere
        seed = torch.zeros(1, self.model_channels, grid_size, grid_size)
        
        if center == None:
            # Set seed in the center to be equal to 1 for all channels except RGB
            seed[:, 3:, grid_size//2, grid_size//2] = 1
        else:
            seed[:, 3:, center[0], center[1]] = 1
        
        return seed
    
    def run(self, model, iterations: int, destroy: bool = True, angle: float = 0.0, env: torch.Tensor = None, 
            seed = None, dynamic_env = False, dynamic_env_type: str = None) -> np.ndarray:
        """ 
        Run model and save state history
        """
        state_grid = self.init_seed(self.grid_size, seed)
        state_history = np.zeros((iterations, self.grid_size, self.grid_size, 4))
        env_history = np.zeros((iterations, self.grid_size, self.grid_size))
        for t in range(iterations):
            
            if env is not None: 
                if dynamic_env == True:
                    env = self.get_env(t, env, dynamic_env_type)
                env_history[t, :, :] = env[0].numpy()
                
                
                
            with torch.no_grad():
                # Visualize state
                transformed_img = state_to_image(state_grid)[0]
                state_history[t] = transformed_img.detach().numpy().reshape(self.grid_size, self.grid_size, 4)
                
                # Update step
                state_grid, env = model.update(state_grid, env, angle = angle)

                # Disrupt pattern
                if destroy == True and t == iterations//2:
                    state_grid = create_block_mask(state_grid, self.grid_size)
        

        return state_history, env_history

    
    def init_env(self, env_channels: int) -> torch.Tensor:
        """
        Initialise environment with zeros

        :param env_channels: Number of environment channels
        :return: 1, env_channels, grid_size, grid_size
        """
        
        env = torch.zeros(1, env_channels, self.grid_size, self.grid_size)
        return env
        
    def add_env(self, env: torch.Tensor, type = 'linear', channel: int = 0, angle: float = 45.0, circle_center: tuple = (20, 20), circle_radius: float = 20.0) -> torch.Tensor:
        """
        Add environment

        :param env: 1, env_channels, grid_size, grid_size
        :param type: Environment type
        :param channel: Channel number
        :param angle: Angle of gradient, defaults to 0.0
        :param circle_center: Center of circular gradient, defaults to (20,20)
        :param circle_radius: Radius of circle, defaults to 20
        :return: 1, env_channels, grid_size, grid_size
        """
        
        
        if type == "linear":
            # Angle of 0 gives 0 to 1 from top to bottom
            env[:, channel] = create_angular_gradient(self.grid_size, angle)
        elif type == "circle":
            env[:, channel] = create_circular_gradient(self.grid_size, circle_center=circle_center, circle_radius=circle_radius)
        elif type == "none":
            env[:, channel] = torch.zeros(self.grid_size, self.grid_size)

            
        return env
    

    def get_env(self, t: int, env: torch.Tensor, type = 'pulse') -> torch.Tensor:
        """
        Returns new environment as a function of time
        """
        
        if type == 'pulse':
            if t <= 10:
                radius = 20
            else:
                radius = 20+10*np.sin(0.05*(t-10))
            env = self.add_env(env, type = 'circle', circle_center = (self.grid_size/2, self.grid_size/2), circle_radius = radius)
            return env
        elif type == 'translation':
            if t <= 20:
                mid = 50
            elif t <= 50:
                mid = 50 - (t - 20)
            elif 150 <= t <= 210:
                mid = 20 + (t - 150)
            elif t <= 350:
                return env
            else:
                mid = 20
            env = self.add_env(env, type = 'circle', circle_center = (mid, mid))
            return env
        elif type == 'phase':
            opacity = 0.5+0.5*np.sin(0.2*(t+10*np.pi/4))
            return opacity*env
        else:
            return env

        
    