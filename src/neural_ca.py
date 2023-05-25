import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

class CAModel(nn.Module):
    """

    Input: 1x48x2828
    Output: 1x16x28x28
    """
    def __init__(self, target, num_channels = 16, fire_rate = 0.5):
        super(CAModel, self).__init__()
        
        self.target = target
        self.num_channels = num_channels
        self.fire_rate = fire_rate
    
        self.input_dim = self.num_channels*3
        
        # Update network
        self.conv1 = nn.Conv2d(self.input_dim, 128, 1)
        self.conv2 = nn.Conv2d(128, self.num_channels, 1)
        self.relu = nn.ReLU()
        self.conv2.weight.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        return out

    def init_seed(self, grid_size):
        """ Initialise seed

        :return: 1, 16 channels, 28 pixels, 28 pixels
        :rtype: torch tensor
        """    
        
        self.grid_size = grid_size
        
        # Initialise seed to zeros everywhere
        seed = torch.zeros(1, self.num_channels, grid_size, grid_size)
        
        # Set seed in the centre to be equal to 255 for RGB channels
        seed[0, 3:, grid_size//2, grid_size//2] = 255
        return seed
    
    def init_pool(self, grid_size, pool_size = 1024):
        """ Initialise pool

        :return: Pool of seed
        :rtype: pool_size, 16, 28, 28
        """
        
        
        seed = self.init_seed(grid_size)
        pool = seed.repeat(pool_size, 1, 1, 1)
        
        return pool

    def perceive(self, state_grid, angle = 0.0):
        """ Compute perception vectors

        :param state_grid: 1, 16, 28, 28
        :type state_grid: torch tensor
        :return: 1, 48, 28, 28
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

        :param state_grid: 1x16x28x28
        :type state_grid: torch tensor
        :param ds_grid: 1x16x28x28
        :type ds_grid: torch tensor
        :return: 1x16x28x28
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
        """ Mask out dead cells.
        
        :param state_grid: 1x16x28x28
        :type state_grid: torch tensor
        :return: 1x16x28x28
        :rtype: torch tensor
        """
        
        # Max pool to find cells with alive neighbours
        
        with torch.no_grad():
            alive = F.max_pool2d(state_grid, kernel_size = 3, stride = 1, padding = 1) > 0.1
        
        # Zero out cells who have dead neighbours
        state_grid = state_grid*alive
        return state_grid
    
    def update(self, state_grid, grid_size):
        
        # Life mask
        state_grid = self.alive_masking(state_grid)
        
        ds_grid = torch.zeros(self.num_channels, 1, grid_size, grid_size)
        
        # Perceive
        perception_grid = self.perceive(state_grid)
        
        # Apply update rule to all cells
        ds_grid = self.forward(perception_grid)

        # Stochastic update mask
        state_grid = self.stochastic_update(state_grid, ds_grid)
        
        return state_grid
        
    
    def train(self, n_epochs, grid_size, optimizer):
        
        """Naive training
        """
        
        self.losses = []
        
        for epoch in range(n_epochs):  
            optimizer.zero_grad()
            
            state_grid = self.init_seed(grid_size)
            
            # Sample random number of CA steps
            iterations = np.random.randint(64, 97)
            
            for t in range(iterations):
                
                state_grid = self.update(state_grid, grid_size)
                
                # Pixel-wise L2 loss
                if t==iterations-1:
                    transformed_img = state_to_image(state_grid)
                    
                    loss = ((transformed_img[:, :,:,0] - self.target[:,:,0])**2).sum()
                    
                    # Visualise progress
                    # if epoch%50==0:
                    #     plt.imshow(transformed_img[0, :,:,0].detach().numpy())
                    #     plt.show()
                
            self.losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            return state_grid
    
    
    def pool_train(self, n_epochs, grid_size, optimizer, sample_size = 32, pool_size = 128):
        """ Train with pool
        """
        
        # Initialise pool
        seed = self.init_seed(grid_size)
        self.pool = self.init_pool(grid_size)
        
        self.losses = []
        
        for epoch in tqdm(range(n_epochs)):
            
            optimizer.zero_grad()
            
            # Sample from pool
            
            # Select indices from pool
            indices = np.random.randint(pool_size, size = sample_size)
            sample = self.pool[indices]
            
            # Calculate loss of samples
            sample_images = state_to_image(sample)
            losses = ((sample_images[:, :, :, 0] - self.target[:,:,0])**2).sum([1, 2])
            
            # Find index with highest loss
            index = int(losses.argmax())
            
            # Reseed highest loss sample
            sample[index] = seed
             
            # Train with sample   
            iterations = np.random.randint(64, 97)
            
            for t in range(iterations):
                
                sample = self.update(sample, grid_size)
                
                # Pixel-wise L2 loss
                if t==iterations-1:
                    transformed_img = state_to_image(sample)
                    
                    loss = ((transformed_img[:,:,:,0] - self.target[:,:,0])**2).sum()
                    
                    # Visualise progress
                    # plt.imshow(transformed_img[0,:,:,0].detach().numpy())
                    # plt.show()
                
            self.losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            # Replace pool with output
            self.pool[indices] = sample.detach()

        
    
    def run(self, iterations, grid_size = 28):
        """ Run model and save state history
        """
    
        state_grid = self.init_seed(grid_size)
        state_history = np.zeros((iterations, 28, 28, 1))
        
        for t in range(iterations):
            
            with torch.no_grad():
                # Visualize state
                transformed_img = state_to_image(state_grid)[0]
                state_history[t] = transformed_img[:,:,0].detach().numpy().reshape(28, 28, 1)
                
                # Update step
                state_grid = self.update(state_grid, grid_size)
                
                # plt.imshow(transformed_img[:,:,0].detach().numpy())
                # plt.show()
                
        return state_history

    
    
    


# Helper functions
def state_to_image(state):
    """ Convert state to image

    :param state: nx16x28x28
    :type state: Tensor
    :return: 28x28x16
    :rtype: Array
    """
    return state.permute(0, 2, 3, 1)

    
