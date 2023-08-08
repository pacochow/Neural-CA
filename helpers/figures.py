import numpy as np
import matplotlib.pyplot as plt

def create_stills(states: np.ndarray, envs: np.ndarray, filename: str, params, intervals):
    
    # Clip values between 0 and 1
    states = states.clip(0, 1)[...,:4]

    times = np.arange(0, len(states), intervals)
    
    # Plotting
    fig, axes = plt.subplots(1, len(times), figsize=(15, 5)) 

    for i, ax in enumerate(axes):
        ax.imshow(states[times[i]])
        ax.set_title(f"t = {times[i]}")
        ax.axis('off')  # To turn off axis numbers

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    
