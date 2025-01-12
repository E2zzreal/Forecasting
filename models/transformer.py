import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size):
        super(TransformerEncoder, self).__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_proj = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 输出投影
        output = self.output_proj(x)
        
        return output 