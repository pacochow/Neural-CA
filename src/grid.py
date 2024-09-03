import numpy as np
import torch
from helpers.helpers import *
from src.train_utils import create_block_mask
import copy

class Grid:
    
    def __init__(self, params):
        
        self.grid_size = params.grid_size
        self.model_channels = params.model_channels
    
    def init_seed(self, grid_size: int, center: tuple = None) -> torch.Tensor:
        """ 
        Initialise seed. Set center to True to initialise seed location in the center.
        Set center to any tuple to change seed location.

        :return: 1, model_channels, grid_size, grid_size
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
    
    def run(self, model, env, params, manual = False) -> np.ndarray:
        """ 
        Run model and save state history
        """
        state_grid = self.init_seed(self.grid_size, params.seed)
        state_history = np.zeros((params.iterations, self.grid_size, self.grid_size, model.model_channels))
        env_history = np.zeros((params.iterations, model.env_channels, self.grid_size, self.grid_size))
        
        
        modulate_vals = state_grid[:, 3]
        
        hidden_history = np.zeros((len(params.hidden_loc), params.iterations, model.hidden_units_2))
        # hidden_history = np.zeros((len(params.hidden_loc), params.iterations, 200))
        
        for t in range(params.iterations):
            
            # Uncomment to perform knockout experiment at specific time iterations
            # if t < params.iterations/2:
            #     model.knockout = True
            # else:
            #     model.knockout = False
                
            new_env = None
            if env is not None: 
                updated_env = copy.deepcopy(env)
                if params.dynamic_env == True:
                    updated_env = self.get_env(t, env, params.dynamic_env_type)
                    
                new_env = copy.deepcopy(updated_env)
                
                # Modulate environment with alpha channel
                if params.modulate_env == True:
                    new_env = updated_env*modulate_vals
                    
                env_history[t, :, :, :] = new_env[0, :].numpy()
                
                
            with torch.no_grad():
                # Visualize state
                transformed_img = state_to_image(state_grid)[0]
                state_history[t] = transformed_img.detach().numpy()
                
                # Update step
                state_grid, new_env = model.update(state_grid, new_env, angle = params.angle, manual = manual)

                # Save alpha channel for modulating environment
                modulate_vals = state_to_image(state_grid)[..., 3]
                
                # Save hidden unit activation history
                if params.vis_hidden == True:
                    for i in range(len(params.hidden_loc)):
                        hidden_history[i, t] = model.hidden_activity[0, :, params.hidden_loc[i][0], params.hidden_loc[i][1]]
                
                # Disrupt pattern
                if params.destroy == True and t == params.iterations//2:
                    state_grid = create_block_mask(state_grid, self.grid_size, type = params.destroy_type)

        return state_history, env_history, hidden_history

    
    def init_env(self, env_channels: int) -> torch.Tensor:
        """
        Initialise environment with zeros

        :param env_channels: Number of environment channels
        :return: 1, env_channels, grid_size, grid_size
        """
        
        env = torch.zeros(1, env_channels, self.grid_size, self.grid_size)
        return env
        
    def add_env(self, env: torch.Tensor, type = 'linear', channel: int = 0, angle: float = -45.0, center: tuple = (25, 25), circle_radius: float = 20.0) -> torch.Tensor:
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
            env[:, channel] = create_circular_gradient(self.grid_size, circle_center=center, circle_radius=circle_radius)
        elif type == "none":
            env[:, channel] = torch.zeros(self.grid_size, self.grid_size)
        elif type == "directional":
            dist = np.sqrt(50)
            angle = angle*np.pi/180
            env[:, channel] = create_circular_gradient(
                self.grid_size, (center[0]+dist*np.sin(angle), center[1]+dist*np.cos(np.pi+angle)), circle_radius = 15)
            env[:, channel+1] = create_circular_gradient(
                self.grid_size, (center[0]+dist*np.sin(np.pi+angle), center[1]+dist*np.cos(angle)), circle_radius = 15)
        elif type == "directional proportional":
  
            dist = np.sqrt(50)*self.grid_size/50
            angle = angle*np.pi/180
            env[:, channel] = create_circular_gradient(
                self.grid_size, (center[0]+dist*np.sin(angle), center[0]+dist*np.cos(np.pi+angle)), 0.3*self.grid_size)
            env[:, channel+1] = create_circular_gradient(
                self.grid_size, (center[1]+dist*np.sin(np.pi+angle), center[1]+dist*np.cos(angle)), 0.3*self.grid_size)
            
        return env
    

    def get_env(self, t: int, env: torch.Tensor, type = 'pulse') -> torch.Tensor:
        """
        Returns new environment as a function of time
        """
        
        if type == 'pulse':
            if t <= 10:
                radius = 20
            else:
                radius = 20+5*np.sin(0.05*(t-10))
            env = self.add_env(env, type = 'circle', center = (self.grid_size/2, self.grid_size/2), circle_radius = radius)
            return env
        elif type == 'translation':
            if t <= 20:
                mid = 50
            elif t <= 50:
                mid = 50 - (t - 20)
            # elif 150 <= t <= 210:
            #     mid = 20 + (t - 150)
            elif t <= 100:
                return env
            else:
                mid = 80
            env = self.add_env(env, type = 'circle', center = (mid, mid))
            # env = self.add_env(env, type = 'directional', angle = -45, center = (mid, mid))
            return env
        elif type == 'phase':
            opacity = 0.5+0.5*np.sin(0.2*(t+10*np.pi/4))
            return opacity*env
        elif type == 'fade out':
            # Opacity exponentially decreases to 0 after 35 iterations to fade out environment
            opacity = 1-1/(1+np.exp(-0.7*(t-39)))
            return opacity*env
        elif type == 'rotating':

            angle = t-45
            env = self.add_env(env, type = 'directional', channel = 0, angle = angle, center = (self.grid_size/2, self.grid_size/2))
            return env
        elif type == 'free move':
            
            if t <= 135:
                angle = t-45
                center = (self.grid_size/2, self.grid_size/2)
            
            elif 136<t <= 200:
                angle = 90
                center = (self.grid_size/2, self.grid_size/2)
            elif 201 < t <= 227:
                angle = 90
                mid = t-152
                center = (mid, self.grid_size/2)
            elif 228<t<=300:
                angle = 90
                center = (75, self.grid_size/2)
            
            elif 301 < t <= 352:
                angle = 90
                mid = 377-t
                center = (mid, self.grid_size/2)
            else:
                angle = 90
                center = (25, self.grid_size/2)
                
            
            env = self.add_env(env, type = 'directional', channel = 0, angle = angle, center = center)
            return env
        else:
            return env

        
    