import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from helpers.helpers import rotate_image
from matplotlib_venn import venn3, venn3_circles

def find_hox_units(hidden_unit_history: dict, living_cells, phase: tuple) -> np.ndarray:
    
    """
    Takes in dictionary of hidden unit histories as input. Set early = True to find early hox genes.
    Returns array of hidden unit indices sorted from most hox-like to least.
    """

    # Get temporal profiles across all pixels
    temporal_profiles = sum(list(hidden_unit_history.values()))
    
    # Normalise temporal profiles
    temporal_profiles -= temporal_profiles[0]
    development_profiles = np.abs(temporal_profiles[:80])
        
    # Compute normalized expression levels
    # normalized_profiles = development_profiles/living_cells


        
    # Find units that have highest cumulative activity between iterations set by phase argument 
    exp = development_profiles[phase[0]:phase[1]].sum(axis=0)
    sorted = exp.argsort()[::-1]
    return development_profiles, sorted
    
    
def plot_expression_profiles(profiles: np.ndarray, sorted_list: np.ndarray, filename: str):
    
    """
    Plot activity profiles of hidden units. Hidden unit profiles are given by input parameter profiles and 
    hidden unit number are given by input parameter sorted_list. 
    """
    
    plt.figure(figsize = (14, 8))

    # Find hox genes
    for i in sorted_list:
        
        plt.plot(profiles[:,i], linewidth = 3.5);

    # plt.legend(sorted_list[:20], fontsize = 16, loc = 'upper right')
    plt.xlabel("Developmental time (iterations)", fontsize = 22)
    plt.ylabel("Activity of hidden unit", fontsize = 22)
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    # plt.axvline(20, color = 'black', linestyle = 'dashed')
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()
    
    
def progressive_knockout_loss(model: nn.Module, units: np.ndarray, grid, env, params):
    
    """
    Progressively knockout hidden units and returns final phenotype and loss
    """
    
    losses = []
    params.knockout = True
    
    for i in range(20):
        
        # Knockout units
        params.knockout_unit = units[:i]
        
        # Run model
        state_history, _, _ = grid.run(model, env, params)

        # Compute loss
        
        target = rotate_image(model.target, params.env_angle+45)
        loss = ((state_history[-1, :, :, :4]-target[0].numpy())**2).mean()
        losses.append(loss)
        
        
    final_phenotype = state_history[-1, :, :, :4]
    return final_phenotype, losses
        
        
def quantify_retrain_improvement(naive_loss, retrain_loss, difference = False):
    
    """"
    Quantify how much retraining improved training compared to naive training 
    """
    
    diff = []
    benchmarks = np.arange(-3.9, -1, 0.001)
    naive_indices = []
    retrain_indices= []
    for j in benchmarks:
        
        retrain_index = next(i for i, x in enumerate(np.log10(retrain_loss)) if x<j)
        naive_index = next(i for i, x in enumerate(np.log10(naive_loss)) if x<j)
        diff.append(naive_index-retrain_index)
        naive_indices.append(naive_index)
        retrain_indices.append(retrain_index)
        
    if diff == False:
        plt.plot(benchmarks, naive_indices)
        plt.plot(benchmarks, retrain_indices)
        
        plt.ylabel("Iterations needed to reach benchmark")
        plt.legend(["Naive", "Retrain"])
    else:
        plt.plot(benchmarks, diff)
        plt.ylabel("Difference in iterations required to reach benchmark")

    plt.xlabel("Log loss benchmark")
    plt.show()
    
    

def compare_developmental_stages(early: list, mid: list, late: list, filename: str):
    
    """
    Takes in hox units at different developmental stages and plots Venn diagram.
    """
    
    set1 = set(early[:20])
    set2 = set(mid[:20])
    set3 = set(late[:20])

    plt.figure(figsize=(7, 7))  
    v=venn3([set1, set2, set3], ["Early", "Mid", "Late"])
    # Customizing the labels
    v.get_label_by_id('100').set_text('\n'.join(map(str, set1 - set2 - set3)))
    v.get_label_by_id('010').set_text('\n'.join(map(str, set2 - set1 - set3)))
    v.get_label_by_id('001').set_text('\n'.join(map(str, set3 - set1 - set2)))
    v.get_label_by_id('110').set_text('\n'.join(map(str, (set1 & set2) - set3)))
    v.get_label_by_id('101').set_text('\n'.join(map(str, (set1 & set3) - set2)))
    v.get_label_by_id('011').set_text('\n'.join(map(str, (set2 & set3) - set1)))
    v.get_label_by_id('111').set_text('\n'.join(map(str, set1 & set2 & set3)))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()