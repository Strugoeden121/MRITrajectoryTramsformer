import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS, OUT_CHANNELS

class BasicTransformer(nn.Module):
    def __init__(self):
        super(BasicTransformer, self).__init__()
        self.encoder = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self. transformer = nn.Transformer(d_model=INPUT_NUM_SAMPLES, nhead=3, num_encoder_layers=6)

        
    def forward(self, x):
        batch_size, channels, shots, samples = x.size()
        #reshaped_x = x.view(INPUT_NUM_FRAMES, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        encode_out = self.encoder(x)
        reshaped_encode_out = encode_out.permute(2, 0, 3,
                                                 1).contiguous()  # Shape: [INPUT_NUM_SHOTS, batch_size, INPUT_NUM_SAMPLES, OUT_CHANNELS]
        reshaped_encode_out = reshaped_encode_out.view(INPUT_NUM_SHOTS, batch_size,
                                                       -1)  # Merge last two dims for d_model
        output = self.transformer(reshaped_encode_out, reshaped_encode_out)
        output = output.view(INPUT_NUM_SHOTS, batch_size, OUT_CHANNELS, INPUT_NUM_SAMPLES).permute(1, 2, 0,
                                                                                                   3).contiguous()
        return output
    

class BasicTransformerTotal(nn.Module):
    def __init__(self, in_channels):
        super(BasicTransformerTotal, self).__init__()
        self.encoder = nn.Conv2d(in_channels=in_channels, out_channels=OUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.transformer = nn.Transformer(d_model=INPUT_NUM_SAMPLES, nhead=3, num_encoder_layers=6)

    def forward(self, x):
        batch_size, channels, shots, samples = x.size()
        # Encoding the input
        encode_out = self.encoder(x)
        reshaped_encode_out = encode_out.permute(2, 0, 3,
                                                 1).contiguous()  # Shape: [shots, batch_size, samples, OUT_CHANNELS]
        reshaped_encode_out = reshaped_encode_out.view(shots, batch_size,
                                                       -1)  # Merge last two dims for d_model
        # Transformer processing
        output = self.transformer(reshaped_encode_out, reshaped_encode_out)
        output = output.view(shots, batch_size, OUT_CHANNELS, INPUT_NUM_SAMPLES).permute(1, 2, 0,
                                                                                         3).contiguous()
        return output
    
    
if __name__ == '__main__':
    model = BasicTransformer()
    x = torch.randn(INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
    y = model(x)
    print(y.shape)
    