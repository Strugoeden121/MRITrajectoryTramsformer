import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS, \
    OUT_CHANNELS, FEATURE_DIM


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(513*16, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 64)  # Second fully connected layer

    def forward(self, x):
        # Flatten the input
        current_frame, shots, samples, dim = x.size()
        x = x.permute(0, 3, 1, 2)
        flat_x = x.reshape(current_frame, dim, shots * samples)
        # Pass through the first fully connected layer
        x = torch.relu(self.fc1(flat_x))
        # Pass through the second fully connected layer
        x = torch.relu(self.fc2(x))
        # Reshape the output to (8, 64, 1, 1)
        x = x.permute(1,0,2)
        return x

class BasicTransformerTraj(nn.Module):
    def __init__(self):
        super(BasicTransformerTraj, self).__init__()
        self.encoder = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=FEATURE_DIM, kernel_size=3, stride=1, padding=1)
        # Adaptive pooling to reduce spatial dimensions to a fixed feature dimension
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, FEATURE_DIM, 1, 1)

        self.mlp = MLP()

        self.transformer = nn.Transformer(d_model=FEATURE_DIM, nhead=4, num_encoder_layers=6)

        self.linear = nn.Linear(FEATURE_DIM, INPUT_NUM_SHOTS * INPUT_NUM_SAMPLES)

    def forward(self, x, num_dims=2):
        # x = x.permute(3, 0, 1, 2)
        # encode_out = self.encoder(x)
        # pooled_out = self.global_pool(encode_out)  # Shape: (batch_size, FEATURE_DIM, 1, 1)
        x = self.mlp(x)

        # Step 3: Reshape to (batch_size, 1, FEATURE_DIM)
        reshaped_encode_out = x.view(num_dims, 1, FEATURE_DIM)

        predictions = torch.zeros(num_dims, INPUT_NUM_FRAMES, FEATURE_DIM)
        # Autoregressive loop over frames
        current_input = reshaped_encode_out[:, :, :]  # Start with the first input

        for t in range(INPUT_NUM_FRAMES - 1):
            output = self.transformer(current_input.permute(1, 0, 2), reshaped_encode_out.permute(1, 0, 2))
            output = output.permute(1, 0, 2)
            predictions[:, t, :] = reshaped_encode_out[:, 0, :]

            current_input = torch.cat((current_input[:, :, :], output[:, 0, :].unsqueeze(1)), dim=1)

            # Update current_input for next time step
            # current_input = output[:, 0, :, :].unsqueeze(1)

        current_input = self.linear(current_input)

        current_input = current_input.reshape(num_dims, INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        return current_input


class BasicTransformerShot(nn.Module):
    def __init__(self):
        super(BasicTransformerShot, self).__init__()

        self.encoder = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=FEATURE_DIM, kernel_size=3, stride=1, padding=1)
        self.transformer = nn.Transformer(d_model=INPUT_NUM_SAMPLES, nhead=3, num_encoder_layers=6)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, time_stamp=None):
        current_frames, shots, samples, dim = x.size()
        num_dims = 2
        x = x.permute(3, 0, 1, 2)
        output = torch.zeros(num_dims, 1, FEATURE_DIM)
        # reshaped_x = x.view(INPUT_NUM_FRAMES, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        encode_out = self.encoder(x)
        pooled_out = self.global_pool(encode_out)
        reshaped_encode_out = pooled_out.view(num_dims, 1, FEATURE_DIM)
        output = self.transformer(reshaped_encode_out, output)
        output = output.reshape(1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, num_dims)
        return output


class BasicTransformerShotAllFrames(nn.Module):
    def __init__(self):
        super(BasicTransformerShotAllFrames, self).__init__()

        self.encoder = nn.Conv2d(in_channels = 2, out_channels = FEATURE_DIM, kernel_size = 3, stride = 1, padding = 1)
        self.transformer = nn.Transformer(d_model=FEATURE_DIM, nhead=4, num_encoder_layers=6)
        self.global_pool = nn.AdaptiveAvgPool2d((2, 1))
        self.linear = nn.Linear(FEATURE_DIM, INPUT_NUM_SHOTS * INPUT_NUM_SAMPLES)
        self.mlp = MLP()


    def forward(self, x, time_stamp=None):
        current_frames, shots, samples, dim = x.size()
        num_dims = 2
        x = x.permute(0, 3, 1, 2)
        output = torch.zeros(num_dims, 1, FEATURE_DIM)
        # reshaped_x = x.view(INPUT_NUM_FRAMES, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        # encode_out = self.encoder(x)
        # pooled_out = self.global_pool(encode_out)
        # reshaped_encode_out = pooled_out.reshape(num_dims, current_frames, FEATURE_DIM)
        x = self.mlp(x)
        output = self.transformer(x.permute(1, 0, 2), output.permute(1, 0, 2))
        output = self.linear(output)
        output = output.permute(1, 0, 2).reshape(num_dims, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        return output

if __name__ == '__main__':
    model = BasicTransformerTraj()
    x = torch.randn(INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
    y = model(x)
    print(y.shape)
