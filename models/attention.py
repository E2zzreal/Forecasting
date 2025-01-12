import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, q, k, v, mask=None):
        # q, k, v shape: [batch_size, seq_len, d_k]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        residual = q
        
        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            
        # Apply attention
        output, attn = self.attention(q, k, v, mask=mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        
        # Final linear projection
        output = self.fc(output)
        
        # Add & Norm
        output = self.layer_norm(output + residual)
        
        return output, attn

class TemporalAttention(nn.Module):
    """时序注意力"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        
        # 计算注意力权重
        attn_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权求和
        context = torch.bmm(x.transpose(1, 2), attn_weights)  # [batch_size, hidden_size, 1]
        context = context.transpose(1, 2)  # [batch_size, 1, hidden_size]
        
        return context, attn_weights

class LocalGlobalAttention(nn.Module):
    """局部-全局注意力"""
    def __init__(self, hidden_size, local_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.local_size = local_size
        
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 局部注意力
        local_out = []
        for i in range(seq_len):
            start = max(0, i - self.local_size)
            end = min(seq_len, i + self.local_size + 1)
            local_x = x[:, start:end, :]
            local_attn, _ = self.local_attention(x[:, i:i+1, :], local_x, local_x)
            local_out.append(local_attn)
        local_out = torch.cat(local_out, dim=1)
        
        # 全局注意力
        global_out, _ = self.global_attention(x, x, x)
        
        # 融合局部和全局特征
        output = self.fusion(torch.cat([local_out, global_out], dim=-1))
        
        return output

class CrossAttention(nn.Module):
    """交叉注意力"""
    def __init__(self, query_dim, key_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.head_dim = query_dim // num_heads
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 分头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.query_dim)
        
        # 输出投影
        output = self.out_proj(context)
        
        return output, attn 