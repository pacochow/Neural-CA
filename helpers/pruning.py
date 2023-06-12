import torch
import torch.nn as nn
import copy
import numpy as np

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
    
    lower_bound = 0.001
    upper_bound = 0.5
    guess = (lower_bound+upper_bound)/2
    
    while finished == False:
        
        # Prune model based on guessed threshold
        pruned_model = prune_network(model, threshold = guess)
        
        # Calculate size of pruned model
        pruned_size = get_parameter_size(pruned_model)
        
        # If pruned model size is not within tolerance, update guess
        if model_size*(100-percent)/100 - tolerance <= pruned_size <= model_size*(100-percent)/100 + tolerance:  
            finished = True
        else:
            if pruned_size < model_size*(100-percent)/100:
                lower_bound += 0.1
            else:
                upper_bound -= 0.1
            
            guess = (lower_bound+upper_bound)/2
            
    return pruned_model