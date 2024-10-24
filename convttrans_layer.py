import torch.nn as nn
from convttrans_sublayer import ConvFeedForward, ConvNorm, ConvMultiHeadAttention


class ConvTTransEncoder(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = ConvNorm(d_model)
        self.norm_2 = ConvNorm(d_model)

        self.attn = ConvMultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = ConvFeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class ConvTTransDecoder(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = ConvNorm(d_model)
        self.norm_2 = ConvNorm(d_model)
        self.norm_3 = ConvNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = ConvMultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = ConvMultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = ConvFeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
