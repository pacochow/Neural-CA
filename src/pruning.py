import torch
import torch.nn as nn
import copy
import numpy as np
from helpers.helpers import state_to_image
from tqdm import tqdm
import matplotlib.pyplot as plt

def prune_network(model: nn.Module, threshold: float) -> nn.Module:
  """
  Prunes a given PyTorch model by setting weights and biases under a given threshold to zero. 
  Returns new model with pruned parameters.
  """

  model_copy = copy.deepcopy(model)

  # Prune weights below the threshold
  with torch.no_grad():
    for p in model_copy.parameters():
      p *= (p.abs() >= threshold).float()

  return model_copy


def get_parameter_size(model: nn.Module) -> nn.Module:
  """
  Computes number of non-zero parameters
  """
  non_zero_params = 0
  for param in model.parameters():
    non_zero_params+=(param!=0).sum()
  return non_zero_params.numpy()


def prune_by_percent(model: nn.Module, percent: float):
    
    finished = False
    model_size = get_parameter_size(model)
    tolerance = 100
    desired_size = model_size*(100-percent)/100
    
    lower_bound = 0.001
    upper_bound = 0.5
    guess = (lower_bound+upper_bound)/2
    
    while finished == False:
        
        # Prune model based on guessed threshold
        pruned_model = prune_network(model, threshold = guess)
        # Calculate size of pruned model
        pruned_size = get_parameter_size(pruned_model)
        # If pruned model size is not within tolerance, update guess
        if desired_size - tolerance <= pruned_size <= desired_size + tolerance:  
            finished = True
        else:
            # Apply binary search algorithm
            diff = np.abs(desired_size - pruned_size)
            if pruned_size < desired_size:
                upper_bound -= 0.1 * diff/desired_size
            else:
                lower_bound += 0.1 * diff/desired_size
            
            guess = (lower_bound+upper_bound)/2
    pruned_percentage = (model_size - pruned_size)*100/model_size
    # print(f'Pruning completed! {pruned_percentage:.2f}% of the model was pruned.\n')
    return model_size, pruned_size, pruned_model
  
  
def compute_pruning_losses(model_name: str, grid, iterations: int, angle: float = 0.0, env = None) -> tuple[list, list]:
  
  model = torch.load(f"./models/{model_name}/final_weights.pt")

  losses = []
  
  # Run model without pruning
  states = grid.run(model, iterations, destroy = True, angle = angle, env = env)
  
  # Compute loss
  losses.append(((states[-1]-model.target.numpy())**2).mean())
  
  # Run model after pruning each percent
  percents = np.linspace(2, 25, 30)
  
  for i in tqdm(range(len(percents))):
      
      # Prune model
      _, _, pruned_model = prune_by_percent(model, percent=percents[i])
      
      # Run model
      if model_name == 'env_circle_16_1':
        pruned_model.env = True
      states = grid.run(pruned_model, iterations, destroy = True, angle = angle, env = env)
      
      # Compute loss
      losses.append(((states[-1]-pruned_model.target.numpy())**2).mean())
  
  percents = [0]+list(percents)
  return percents, losses