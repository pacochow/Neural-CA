import pickle 
import numpy as np
import matplotlib.pyplot as plt

# Load model
model_name = "angled_env_directional_16_2_529"


# Load hidden unit histories
filename = f'./models/{model_name}/hidden_unit_history.pkl'
with open(filename, 'rb') as fp:
    hidden_unit_history = pickle.load(fp)
    

unit = hidden_unit_history[(20, 25)]
active_units = unit.sum(axis = 0)
sorted = active_units.argsort()[::-1][:10]

print((active_units==0).sum())


# plt.plot(unit[:, sorted[-1]])