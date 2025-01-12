import torch
import torch.nn as nn

class TSDiff(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_steps):
        super(TSDiff, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps
        
        # 时间步嵌入
        self.time_embedding = nn.Embedding(num_steps, hidden_dim)
        
        # LSTM编码器
        self.encoder = nn.LSTM(
            input_size=input_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t):
        # 时间步嵌入
        t_emb = self.time_embedding(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # 连接输入和时间嵌入
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # LSTM处理
        lstm_out, _ = self.encoder(x_t)
        
        # 输出预测
        output = self.output_layer(lstm_out)
        
        return output

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """线性beta调度"""
    return torch.linspace(start, end, timesteps)

def forward_diffusion(x, t, beta_schedule):
    """前向扩散过程"""
    noise = torch.randn_like(x)
    alpha = 1 - beta_schedule(t)
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    mean = torch.sqrt(alpha_bar) * x
    var = torch.sqrt(1 - alpha_bar) * noise
    return mean + var, noise

def reverse_diffusion(model, x, t, beta_schedule):
    """反向扩散过程"""
    predicted_noise = model(x, t)
    alpha = 1 - beta_schedule(t)
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    mean = (1 / torch.sqrt(alpha)) * (x - (beta_schedule(t) / torch.sqrt(1 - alpha_bar)) * predicted_noise)
    
    if t > 0:
        z = torch.randn_like(x)
    else:
        z = 0
        
    var = z * torch.sqrt(beta_schedule(t))
    return mean + var 