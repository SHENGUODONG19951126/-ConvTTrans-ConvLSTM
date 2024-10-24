import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
    
        self.d_model = d_model[0]
        self.d_h = d_model[1]
        self.d_w = d_model[2]
        self.norm = nn.BatchNorm2d(d_model[0], track_running_stats=False)
    
    def forward(self, x):

        bs = x.size(0)
        seq = x.size(1)
        norm = self.norm(x.contiguous().view(-1, self.d_model, self.d_h, self.d_w)).view(bs, seq, self.d_model, self.d_h, self.d_w)

        return norm

def ConvAttention(q, k, v, d_k, d_h, d_w, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k * d_h * d_w)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class ConvMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):  # d_model:[C, H, W]
        super().__init__()
        
        self.d_model = d_model[0]
        self.d_k = d_model[0] // heads
        self.d_h = d_model[1]
        self.d_w = d_model[2]
        self.h = heads

        self.q_conv = nn.Conv2d(d_model[0], d_model[0], 3, padding=1)
        self.v_conv = nn.Conv2d(d_model[0], d_model[0], 3, padding=1)
        self.k_conv = nn.Conv2d(d_model[0], d_model[0], 3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Conv2d(d_model[0], d_model[0], 3, padding=1)
    
    def forward(self, q, k, v, mask=None):  # q, k, v:[batch, length, channel, height, width]
                                            # orig: [batch, length, dim]
        bs = q.size(0)
        seq_q = q.size(1)
        seq_k = k.size(1)
        seq_v = v.size(1)
        
        # perform convolution operation and split into N heads
        k = self.k_conv(k.view(-1, self.d_model, self.d_h, self.d_w)).view(bs, seq_k, self.h, self.d_k, self.d_h, self.d_w)
        q = self.q_conv(q.view(-1, self.d_model, self.d_h, self.d_w)).view(bs, seq_q, self.h, self.d_k, self.d_h, self.d_w)
        v = self.v_conv(v.view(-1, self.d_model, self.d_h, self.d_w)).view(bs, seq_v, self.h, self.d_k, self.d_h, self.d_w)

        # transpose to get dimensions batch * head * length * d_k (channel) * height * width
        k = k.transpose(1,2).view(bs, self.h, seq_k, -1)
        q = q.transpose(1,2).view(bs, self.h, seq_q, -1)
        v = v.transpose(1,2).view(bs, self.h, seq_v, -1)

        # calculate attention using function we will define next    (batch, head, length, dim)
        scores = ConvAttention(q, k, v, self.d_k, self.d_h, self.d_w, mask, self.dropout)
        # concatenate heads and put through final linear layer   (batch, length, head, dim)
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, seq_q, self.d_model, self.d_h, self.d_w)
        output = self.out(concat.view(-1, self.d_model, self.d_h, self.d_w)).view(bs, seq_q, self.d_model, self.d_h, self.d_w)

        return output

class ConvFeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model[0]
        self.d_h = d_model[1]
        self.d_w = d_model[2]

        self.dropout = nn.Dropout(dropout)

        self.conv_1 = nn.Conv2d(d_model[0], d_model[0] * 4, 3, padding=1)
        self.conv_2 = nn.Conv2d(d_model[0] * 4, d_model[0], 3, padding=1)
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        bs = x.size(0)
        seq = x.size(1)

        x = self.dropout(self.leaky(self.conv_1(x.view(-1, self.d_model, self.d_h, self.d_w))))
        x = self.conv_2(x).view(bs, seq, self.d_model, self.d_h, self.d_w)

        return x
