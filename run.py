import torch   

from helpers.visualizer import create_animation

iterations = 1600

# Load model
model = torch.load("./model_params/model.pt")

# Run model
state_history = model.run(iterations)

# Create animation
nSeconds = 10
filename = './media/run2.mp4'
create_animation(state_history, iterations, nSeconds, filename)