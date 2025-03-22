import torch

# Load the checkpoint
checkpoint = torch.load("output/net_000000_00.135.pth", map_location=torch.device('cpu'))

# Print the keys in the checkpoint
print(checkpoint.keys())

# If it's a model state_dict, you can load it into a model
# Example:
# model.load_state_dict(checkpoint)
# Assuming it is a state_dict
for layer_name, param in checkpoint.items():
    print(f"Layer: {layer_name}")
    print(f"Shape: {param.shape}")
    # print(f"Values: {param}\n") 
import numpy as np

# Load the .npy file
data = np.load("erl_run_valid_position.npy")

# Print the shape and contents of the file
print("Shape of the data:", data.shape)
print("Contents:\n", data)
