import torch
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
from helpers.helpers import rotate_image
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
  
  
def compute_pruning_losses(model_name: str, grid, params, env = None) -> tuple:
  
  model = torch.load(f"./models/{model_name}/final_weights.pt")

  losses = []
  
  # Run model without pruning
  full_states, _, _ = grid.run(model, env, params)
  states = full_states[..., :4]
  
  # Compute loss
  losses.append(((states[-1]-model.target.numpy())**2).mean())
  
  # Run model after pruning each percent
  percents = np.linspace(2, 25, 30)
  
  for i in tqdm(range(len(percents))):
      
      # Prune model
      _, _, pruned_model = prune_by_percent(model, percent=percents[i])

      # Run model
      full_states, _, _ = grid.run(pruned_model, env, params)
      states = full_states[..., :4]
      # Compute loss
      losses.append(((states[-1]-pruned_model.target.numpy())**2).mean())
  
  percents = [0]+list(percents)
  return percents, losses


def prune_by_channel(model: nn.Module, channel: int, enhance: bool = False) -> nn.Module:
  
  model_copy = copy.deepcopy(model)
  
  # Get parameters
  params = [x.data for x in model_copy.parameters()]
  
  # Prune weights below the threshold
  with torch.no_grad():
    if enhance == False:
      params[-2][channel] = 0
      params[-1][channel] = 0
    else:
      params[-2][channel] = 0.01
      params[-1][channel] = 0

  return model_copy

def prune_by_unit(model: nn.Module, grid, env, params, prune_units = None):
  
  units = np.arange(model.hidden_units)
  
  
  
  if prune_units == None:
    prune_range = units
  
  else:
    prune_range = np.arange(prune_units[0], prune_units[1])
  
  losses = np.zeros(len(prune_range))
  phenotype = np.zeros((len(prune_range), grid.grid_size, grid.grid_size, 4))
    
  with torch.no_grad():
    for i in tqdm(range(len(prune_range))):
      params.knockout_unit = [units[prune_range[i]]]
      
      # Run model
      state_history, _, _ = grid.run(model, env, params)

      # Compute loss
      target = rotate_image(model.target, params.env_angle+45)
      losses[i] = ((state_history[-1, :, :, :4]-target[0].numpy())**2).mean()
      
      phenotype[i] = state_history[-1, :, :, :4]
      
  
  return phenotype, losses


def comparing_pruning_losses(model1: str, grid1, env1, model2: str, grid2, env2, filename: str, params):
    
    percents, loss1 = compute_pruning_losses(model1, grid1, params.iterations, params.angle, env1)
    percents, loss2 = compute_pruning_losses(model2, grid2, params.iterations, params.angle, env2)
    plt.scatter(percents, np.log10(loss1))
    plt.scatter(percents, np.log10(loss2))
    plt.xlabel("Pruned percentage (%)", fontsize =12)
    plt.ylabel("Log loss", fontsize = 12)
    plt.title("Loss after pruning", fontsize = 18)
    plt.legend([model1, model2])
    plt.tight_layout()
    plt.savefig(filename)