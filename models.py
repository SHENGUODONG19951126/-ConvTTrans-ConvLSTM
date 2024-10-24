import torch
import torch.nn as nn
from convttrans_layer import ConvTTransEncoder, ConvTTransDecoder
from convlstm_bridge import ConvLSTM_AlphaBeta

class convttrans_lstm(nn.Module):
    def __init__(self, d_model, heads, dropout, lamda):
        super().__init__()

        self.lamda = lamda # merging coefficient

        # convolutional and deconvolutional modules
        self.conv_1 = nn.Conv2d(d_model[0], 16, 3, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, 3, padding=1)

        self.deconv_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.conv_4 = nn.Conv2d(16, d_model[0], 3, padding=1)

        # convolutional self-attention modules
        self.conv_tsfm_en_1 = ConvTTransEncoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_en_2 = ConvTTransEncoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_en_3 = ConvTTransEncoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_en_4 = ConvTTransEncoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_en_5 = ConvTTransEncoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)

        # decoder forward
        self.conv_tsfm_de_1_f = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_2_f = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_3_f = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_4_f = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_5_f = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        # decoder backward
        self.conv_tsfm_de_1_b = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_2_b = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_3_b = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_4_b = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)
        self.conv_tsfm_de_5_b = ConvTTransDecoder([64, int(d_model[1] / 4), int(d_model[2] / 4)], heads, dropout)


        # batch normalization
        self.bn_1 = nn.BatchNorm2d(16, track_running_stats=False)
        self.bn_2 = nn.BatchNorm2d(32, track_running_stats=False)
        self.bn_3 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn_4 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn_5 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn_6 = nn.BatchNorm2d(32, track_running_stats=False)
        self.bn_7 = nn.BatchNorm2d(16, track_running_stats=False)

        # maxpooling, activation
        self.pl = nn.MaxPool2d(2, 2)
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

        # Convlstm
        self.convlstm_1f = ConvLSTM_AlphaBeta(shape=(d_model[1], d_model[2]), input_channels=16, filter_size=5,
                                              num_features=16)
        self.convlstm_2f = ConvLSTM_AlphaBeta(shape=(d_model[1], d_model[2]), input_channels=16, filter_size=5,
                                              num_features=16)
        self.convlstm_1b = ConvLSTM_AlphaBeta(shape=(d_model[1], d_model[2]), input_channels=16, filter_size=5,
                                              num_features=16)
        self.convlstm_2b = ConvLSTM_AlphaBeta(shape=(d_model[1], d_model[2]), input_channels=16, filter_size=5,
                                              num_features=16)


    def forward(self, clip, t_src, src_mask, trg_mask):

        b_clip, t_clip, _, _, _ = clip.size()
        t_trg = t_clip - 2 * t_src

        # CNN encoder: encode both source and target
        # B, T, C, H, W
        clip = self.leaky(self.bn_1(self.conv_1(clip.view(-1, clip.size(2), clip.size(3), clip.size(4)))))

        # layer-interactive ConvLSTM bridge (Part 1)
        clip_lstm = clip.view(b_clip, t_clip, clip.size(1), clip.size(2), clip.size(3)).transpose(0, 1) # t, b, c, h, w
        # forward pipeline
        trg_lstm_f = clip_lstm[(t_src-1):(t_src+t_trg-1),...]
        _, hidden_f = self.convlstm_1f(trg_lstm_f, seq_len=t_trg, hidden_list=None)
        # backward pipeline
        trg_lstm_b = clip_lstm[(1-t_src-t_trg):(1-t_src),...]
        trg_lstm_b = torch.flip(trg_lstm_b, [0,]) # flip the temporal order
        _, hidden_b = self.convlstm_1b(trg_lstm_b, seq_len=t_trg, hidden_list=None)

        # CNN encoder (continue)
        clip = self.pl(self.leaky(self.bn_2(self.conv_2(clip))))
        clip = self.pl(self.leaky(self.bn_3(self.conv_3(clip))))
        clip = clip.view(b_clip, t_clip, clip.size(1), clip.size(2), clip.size(3))

        src = torch.cat([clip[:, :t_src,...], clip[:, -t_src:,...]], 1)
        trg_f = clip[:, (t_src-1):(t_src+t_trg-1),...]
        trg_b = clip[:, (1-t_src-t_trg):(1-t_src),...]
        trg_b = torch.flip(trg_b, [1,]) # flip the temporal order

        # transformer encoder
        src = self.conv_tsfm_en_1(src, src_mask)
        src = self.conv_tsfm_en_2(src, src_mask)
        src = self.conv_tsfm_en_3(src, src_mask)
        src = self.conv_tsfm_en_4(src, src_mask)
        src = self.conv_tsfm_en_5(src, src_mask)

        # transformer encoder norm
        src = self.bn_4(src.view(-1, src.size(2), src.size(3), src.size(4)))
        en_output = src.view(b_clip, 2 * t_src, src.size(1), src.size(2), src.size(3))

        # transformer decoder in the forward pipeline
        pred_f = self.conv_tsfm_de_1_f(trg_f, en_output, src_mask, trg_mask)
        pred_f = self.conv_tsfm_de_2_f(pred_f, en_output, src_mask, trg_mask)
        pred_f = self.conv_tsfm_de_3_f(pred_f, en_output, src_mask, trg_mask)
        pred_f = self.conv_tsfm_de_4_f(pred_f, en_output, src_mask, trg_mask)
        pred_f = self.conv_tsfm_de_5_f(pred_f, en_output, src_mask, trg_mask)
        # transformer decoder in the backward pipeline
        pred_b = self.conv_tsfm_de_1_b(trg_b, en_output, src_mask, trg_mask)
        pred_b = self.conv_tsfm_de_2_b(pred_b, en_output, src_mask, trg_mask)
        pred_b = self.conv_tsfm_de_3_b(pred_b, en_output, src_mask, trg_mask)
        pred_b = self.conv_tsfm_de_4_b(pred_b, en_output, src_mask, trg_mask)
        pred_b = self.conv_tsfm_de_5_b(pred_b, en_output, src_mask, trg_mask) # still flipped

        # transformer decoder norm
        pred_f = self.bn_5(pred_f.view(-1, pred_f.size(2), pred_f.size(3), pred_f.size(4)))
        pred_b = self.bn_5(pred_b.view(-1, pred_b.size(2), pred_b.size(3), pred_b.size(4)))

        # CNN decoder
        pred_f = self.leaky(self.bn_6(self.deconv_1(pred_f)))
        pred_f = self.leaky(self.bn_7(self.deconv_2(pred_f)))

        pred_b = self.leaky(self.bn_6(self.deconv_1(pred_b)))
        pred_b = self.leaky(self.bn_7(self.deconv_2(pred_b)))

        # layer-interactive ConvLSTM bridge (Part 2)
        # forward pipeline
        pred_f = pred_f.reshape(b_clip, t_trg, pred_f.size(1), pred_f.size(2), pred_f.size(3)).transpose(0, 1) # t, b, c, h, w
        pred_f, _ = self.convlstm_2f(pred_f, seq_len=t_trg, hidden_list=hidden_f)
        pred_f = pred_f.transpose(0, 1) # .reshape(-1, pred_f.size(2), pred_f.size(3), pred_f.size(4)) # b, t, c, h, w
        # backward pipeline
        pred_b = pred_b.reshape(b_clip, t_trg, pred_b.size(1), pred_b.size(2), pred_b.size(3)).transpose(0, 1) # t, b, c, h, w
        pred_b, _ = self.convlstm_2b(pred_b, seq_len=t_trg, hidden_list=hidden_b)
        pred_b = pred_b.transpose(0, 1) # .reshape(-1, pred_b.size(2), pred_b.size(3), pred_b.size(4)) # b, t, c, h, w

        # merge forward and backward predictions
        pred_b = torch.flip(pred_b, [1,]) # flip back
        pred = self.lamda * pred_f + (1.0 - self.lamda) * pred_b
        pred = pred.reshape(-1, pred.size(2), pred.size(3), pred.size(4))

        pred = self.tanh(self.conv_4(pred))
        output = pred.view(b_clip, t_trg, pred.size(1), pred.size(2), pred.size(3))

        return output
