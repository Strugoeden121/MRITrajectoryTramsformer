import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS, OUT_CHANNELS,FEATURE_DIM

class BasicTransformer(nn.Module):
    def __init__(self):
        super(BasicTransformer, self).__init__()
        self.encoder = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=FEATURE_DIM, kernel_size=3, stride=1, padding=1)
        # Adaptive pooling to reduce spatial dimensions to a fixed feature dimension
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, FEATURE_DIM, 1, 1)
        
        self. transformer = nn.Transformer(d_model=FEATURE_DIM, nhead=4, num_encoder_layers=6)

        self.linear = nn.Linear(FEATURE_DIM, INPUT_NUM_SHOTS * INPUT_NUM_SAMPLES)

    def forward(self, x, batch_size):
        #reshaped_x = x.view(INPUT_NUM_FRAMES, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        encode_out = self.encoder(x)
        pooled_out = self.global_pool(encode_out)  # Shape: (batch_size, FEATURE_DIM, 1, 1)

        # Step 3: Reshape to (batch_size, 1, FEATURE_DIM)
        reshaped_encode_out = pooled_out.view(batch_size, 1, FEATURE_DIM)  
       # reshaped_encode_out = encode_out.permute(2, 0, 3,
                                 #                1).contiguous()  # Shape: [INPUT_NUM_SHOTS, batch_size, INPUT_NUM_SAMPLES, OUT_CHANNELS]
        #reshaped_encode_out = reshaped_encode_out.view(batch_size,
                           #                            -1)  # Merge last two dims for d_model
        #reshaped_encode_out = reshaped_encode_out.unsqueeze(1)  # Add seq dim
        
        predictions = torch.zeros(batch_size, INPUT_NUM_FRAMES, FEATURE_DIM)
        # Autoregressive loop over frames
        current_input = reshaped_encode_out[:, :, :]  # Start with the first input

        for t in range(INPUT_NUM_FRAMES-1):
            
            output = self.transformer(current_input.permute(1, 0, 2), reshaped_encode_out.permute(1, 0, 2))
            output = output.permute(1,0,2)
            predictions[:, t, :] = reshaped_encode_out[:, 0, :]
           

            current_input = torch.cat((current_input[:, :, :], output[:, 0, :].unsqueeze(1)), dim=1)

            # Update current_input for next time step
            #current_input = output[:, 0, :, :].unsqueeze(1)
            
            
        current_input = self.linear(current_input)

        current_input = current_input.reshape(batch_size, INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        return current_input
    
    

import torch
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self, input_channels, feature_dim, seq_length, transformer_hidden_dim, num_heads, num_layers):
        super(AutoregressiveModel, self).__init__()
        self.conv = nn.Conv2d(input_channels, feature_dim, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        
        # Transformer
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, feature_dim))
        self.transformer = nn.Transformer(
            d_model=feature_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers
        )
        self.output_layer = nn.Linear(feature_dim, feature_dim)  # Adjust as per output dimensions

    def forward(self, x, target_seq=None, training=True):
        batch_size = x.size(0)
        # Conv2D Feature Extraction
        features = self.conv(x)  # Output shape: (batch_size, feature_dim, height, width)
        features = features.mean(dim=[-1, -2])  # Global Average Pooling -> (batch_size, feature_dim)

        # Autoregressive Loop
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, self.feature_dim).to(x.device)  # Start token

        for t in range(self.seq_length):
            if training and target_seq is not None and t > 0:
                decoder_input = target_seq[:, :t, :]  # Use ground truth for teacher forcing

            # Add positional encoding
            decoder_input = decoder_input + self.positional_encoding[:, :decoder_input.size(1), :]

            # Transformer decoding
            transformer_output = self.transformer(
                src=features.unsqueeze(1),  # Add seq dim to features -> (batch_size, 1, feature_dim)
                tgt=decoder_input
            )

            # Generate output for the current timestep
            current_output = self.output_layer(transformer_output[:, -1, :])  # Last output in sequence
            outputs.append(current_output.unsqueeze(1))  # Add timestep to outputs
            
            # Use the current output as the next input in inference
            decoder_input = torch.cat([decoder_input, current_output.unsqueeze(1)], dim=1)

        outputs = torch.cat(outputs, dim=1)  # Concatenate outputs along the sequence dimension
        return outputs
    
    
if __name__ == '__main__':
    model = BasicTransformer()
    x = torch.randn(INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
    y = model(x)
    print(y.shape)
    