import torch as th
import torch.nn as nn


class BasicTransformer(nn.Module):
    def __init__(self):
        super(BasicTransformer, self).__init__()
        
        
        
        # Conv1D expects input in [batch_size, feature_dim, seq_len], so we reshape first
        self.conv1d = nn.Conv1d(in_channels=513, out_channels=513, kernel_size=1)  # No reduction in feature_dim
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduce sequence length to 1
        self.flatten = nn.Flatten()  # Collapse to a latent vector

    def forward(self, x):
        # Transpose input to [batch_size, feature_dim, seq_len] for Conv1D
        x = x.permute(1, 2, 0)  # From [batch_size, seq_len, feature_dim] to [batch_size, feature_dim, seq_len]
        x = self.conv1d(x)  # Apply Conv1D
        x = self.pool(x)  # Pool over the sequence dimension to get [batch_size, feature_dim, 1]
        x = self.flatten(x)  # Flatten to [batch_size, feature_dim]
        return x


def testRun():
    print("Start testing...")
    # generate a random tensor of size 8X16X513 
    x = th.rand(8, 16, 513)

    # define a 1D convolutional layer that inputs in shape 8X16X513 and outputs 16X513
    model = Conv1DToLatent()
    

    # apply the convolutional layer to the input tensor
    y = model(x)

    # print the shape of the output tensor
    print(y.shape)
    
    #initialize the nn.Transformer model
    model = nn.Transformer(nhead=8, num_encoder_layers=6)
    # each input is vector of size 1X513 - from y above
    src = y.unsqueeze(0)
    tgt = y.unsqueeze(0)
    out = th.nn.Transformer.forward(model, src, tgt)
    print(out.shape)
    
if __name__ == '__main__':
    testRun()

