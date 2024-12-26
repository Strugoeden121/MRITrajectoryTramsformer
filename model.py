import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS,
                        OUT_CHANNELS

class BasicTransformer(nn.Module):
    def __init__(self):
        super(BasicTransformer, self).__init__()
        self.encoder = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self. transformer = nn.Transformer(nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1)
        
    def forward(self, x):
        reshaped_x = x.view(INPUT_NUM_FRAMES, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        encode_out = self.encoder(reshaped_x)
        reshaped_encode_out = encode_out.view(INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        output = self.transformer(encode_out, encode_out)
        return output

if __name__ == '__main__':
    model = BasicTransformer()
    x = torch.randn(INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
    y = model(x)
    print(y.shape)
    