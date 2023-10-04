import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Env_CA(nn.Module):
    """
    Input: n, n_channels*3, grid_size, grid_size
    Output: n, n_channels, grid_size, grid_size
    """
    def __init__(self, target: np.ndarray, params):
        super(Env_CA, self).__init__()
        
        self.target = torch.tensor(target)
        self.model_channels = params.model_channels
        self.env_channels = params.env_channels
        self.hidden_units = params.hidden_units[0]
        
        self.env = True if self.env_channels > 0 else False
        self.fire_rate = params.fire_rate
    
        self.num_channels = self.model_channels + self.env_channels
        self.input_dim = self.num_channels*3
        
        # Update network
        self.conv1 = nn.Conv2d(self.input_dim, self.hidden_units, 1)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(self.hidden_units, self.model_channels, 1)
        if params.n_layers > 1:
            self.hidden_units_2 = params.hidden_units[1]
            self.hidden_layer_2 = nn.Conv2d(self.hidden_units, self.hidden_units_2, 1)
            nn.init.xavier_uniform_(self.hidden_layer_2.weight)
            nn.init.zeros_(self.hidden_layer_2.bias)
            self.conv2 = nn.Conv2d(self.hidden_units_2, self.model_channels, 1)
            if params.n_layers > 2:
                self.hidden_units_3 = params.hidden_units[2]
                self.hidden_layer_3 = nn.Conv2d(self.hidden_units_2, self.hidden_units_3, 1)
                nn.init.xavier_uniform_(self.hidden_layer_3.weight)
                nn.init.zeros_(self.hidden_layer_3.bias)
                self.conv2 = nn.Conv2d(self.hidden_units_3, self.model_channels, 1)
            
        
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        
        self.device = params.device
        self.params = params
        self.knockout = self.params.knockout

    def forward(self, x: torch.Tensor, living_cells = None):
        out = self.relu(self.conv1(x))
        
        # Save activation of hidden units
        self.hidden_activity = out
        # If performing experiment with knockout, only knockout pixels with living cells
        if self.knockout == True:
            for i in self.params.knockout_unit:
                out[0, i] = 0*living_cells
        
        out = self.relu(self.hidden_layer_2(out))
       
        
        
        
        out = self.relu(self.hidden_layer_3(out))
        
        out = self.conv2(out)
        
        return out

    def perceive(self, state_grid: torch.Tensor, angle = 0.0) -> torch.Tensor:
        """ 
        Compute perception vectors

        :param state_grid: n, num_channels, grid_size, grid_size
        :return: n, input_dim, grid_size, grid_size
        """    
        
        # Identity filter
        identify = torch.tensor(np.outer([0, 1, 0], [0, 1, 0]))
        
        # laplace = torch.tensor(([1, 2, 1], [2, -12, 2], [1, 2, 1]))/8
        # kernel_stack = torch.stack([identify,  laplace], 0)
        # kernel = kernel_stack.unsqueeze(1)
        # kernel = kernel.float().repeat(self.num_channels, 1, 1, 1).to(self.device)
        # state_repeated = state_grid.repeat_interleave(kernel_stack.shape[0], dim = 1)
        # perception_grid = F.conv2d(state_repeated, kernel, padding = 1, groups = kernel.size(0))
        # return perception_grid
        
        # Sobel filters
        dx = torch.tensor(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0)  # Sobel filter
        dy = dx.T
        
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        
        # Stack filters together
        kernel_stack = torch.stack([identify, c*dx-s*dy, s*dx+c*dy], 0)
        kernel = kernel_stack.unsqueeze(1)
        
        # Repeat kernels to form num_channels x 1 x 3 x 3 filter
        kernel = kernel.float().repeat(self.num_channels, 1, 1, 1).to(self.device)

        state_repeated = state_grid.repeat_interleave(kernel_stack.shape[0],dim = 1)
        
        # Perform convolution
        perception_grid = F.conv2d(state_repeated, kernel, padding=1, groups=kernel.size(0))

        return perception_grid

    def stochastic_update(self, grid: torch.Tensor, ds_grid: torch.Tensor) -> torch.Tensor:
        """ 
        Apply stochastic mask so that all cells do not update together.

        :param grid: n, channels, grid_size, grid_size
        :param ds_grid: n, channels, grid_size, grid_size
        :return: n, channels, grid_size, grid_size
        """
        
        # Random mask 
        rand_mask = (torch.rand(ds_grid.shape[0], 1, ds_grid.shape[-1], ds_grid.shape[-1])<=self.fire_rate).to(self.device)
        return grid+ds_grid*rand_mask

    def alive_masking(self, state_grid: torch.Tensor) -> torch.Tensor:
        """ Returns mask for dead cells
        
        :param state_grid: n, model_channels, grid_size, grid_size
        :return: n, 1, grid_size, grid_size
        """

        return F.max_pool2d(state_grid[:,3:4,:,:], kernel_size = 3, stride = 1, padding = 1) > 0.1
    
    def update(self, state_grid: torch.Tensor, env = None, angle = 0.0, manual = False) -> torch.Tensor:
        
        """
        Takes in state as input and outputs updated state at next time iteration
        """
        
        
        # Pre update life mask
        pre_mask = self.alive_masking(state_grid)

        # Perceive
        if self.env == True:
            
            # Create full grid by concatenating state grid with env
            full_grid = torch.cat([state_grid,env], dim = 1)
            perception_grid = self.perceive(full_grid, angle)
        else: 
            perception_grid = self.perceive(state_grid, angle)
        
        
        if manual == False:
            # If manual is set to False, apply neural network to all pixels simultaneously
            ds_grid = self.forward(perception_grid, state_grid[0, 3])
        else:
            # If manual is set to True, loop over each pixel sequentially (for knocking out units at specific pixels)
            ds_grid = torch.zeros(perception_grid.shape[0], self.model_channels, perception_grid.shape[-1], perception_grid.shape[-1])
            # Apply update rule to all cells
            for i in range(perception_grid.shape[-1]):
                for j in range(perception_grid.shape[-1]):
                    self.knockout = True if 30<i<50 and 5<j<20 else False
                    ds_grid[:, :, i, j] = self.forward(perception_grid[:, :, i:i+1, j:j+1], state_grid[0, 3, i, j])[..., 0, 0]
            


            

        state_grid = self.stochastic_update(state_grid, ds_grid)
        
        # Post update life mask
        post_mask = self.alive_masking(state_grid)
        
        life_mask = pre_mask & post_mask
        
        # Zero out dead cells
        state_grid = life_mask*state_grid
        return state_grid, env
    