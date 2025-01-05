import torch
import torch.nn as nn
from utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS, OUT_CHANNELS


# Dummy dataset generation (replace with real data)
def generate_dummy_data(num_samples):
    inputs = torch.rand(num_samples, IN_CHANNELS, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)  # Example shape
    targets = torch.rand(num_samples, IN_CHANNELS, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)  # Example shape

    return inputs, targets