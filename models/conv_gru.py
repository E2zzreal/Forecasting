import torch
import torch.nn as nn
from .attention import MultiHeadAttention, TemporalAttention, LocalGlobalAttention

class MultiScaleConv1d(nn.Module):
    """多尺度卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels//3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels//3, kernel_size=7, padding=3)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # 多尺度特征提取
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        conv7_out = self.conv7(x)
        
        # 特征拼接
        out = torch.cat([conv3_out, conv5_out, conv7_out], dim=1)
        out = self.norm(out)
        out = self.activation(out)
        return out

class ConvBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3):
        super().__init__()
        
        # 多尺度卷积特征提取
        self.multi_scale_conv = MultiScaleConv1d(input_size, hidden_size)
        
        # BiGRU层
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 注意力机制
        self.attention = MultiHeadAttention(
            d_model=hidden_size * 2,
            n_heads=8
        )
        
        self.temporal_attention = TemporalAttention(
            hidden_size=hidden_size * 2
        )
        
        self.local_global_attention = LocalGlobalAttention(
            hidden_size=hidden_size * 2
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度卷积特征提取
        conv_out = self.multi_scale_conv(x)
        
        # [batch, features, seq_len] -> [batch, seq_len, features]
        conv_out = conv_out.transpose(1, 2)
        
        # BiGRU处理
        gru_out, _ = self.gru(conv_out)
        
        # 多头注意力
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # 时序注意力
        temporal_context, _ = self.temporal_attention(attn_out)
        
        # 局部-全局注意力
        output = self.local_global_attention(temporal_context)
        
        # 输出层
        output = self.fc(output)
        
        return output 