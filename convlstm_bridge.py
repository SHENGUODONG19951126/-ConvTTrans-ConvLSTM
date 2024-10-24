import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.modules.utils as utils


class ConvLSTM_AlphaBeta(nn.Module):
    # ConvLSTM alpha and beta (optional) in the manuscript. ConvLSTM beta incorporates hidden states from ConvLSTM alpha.
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(ConvLSTM_AlphaBeta, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features * 2,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features//32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_list=None, seq_len=10):

        device=inputs.device
        hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                         self.shape[1],device=device)  # b,c,h,w
        cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                         self.shape[1],device=device)

        if hidden_list is None: # decide whether to incorporate the hidden states from previous ConvLSTM alpha layer
            flag=False
        else:
            flag=True

        output_inner = []
        new_hidden_list=[]

        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1],device=device)
            else:
                x = inputs[index, ...]

            if not flag:    # no extra previous hidden layer info
                hx_e = torch.zeros(hx.size(),device=device)
            else:
                hx_e,_ = hidden_list[index]

            combined = torch.cat((x,hx,hx_e), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cx = (forgetgate * cx) + (ingate * cellgate)
            hx = outgate * torch.tanh(cx)
            output_inner.append(hx)
            new_hidden_list.append((hx, cx))

        return torch.stack(output_inner),new_hidden_list
