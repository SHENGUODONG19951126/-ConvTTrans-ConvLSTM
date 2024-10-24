import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models import convttrans_lstm
from dataset.data_loader import VAD_train_set
from convttrans_mask import create_convttrans_mask
from tqdm import tqdm
from early_stop import EarlyStopping
import numpy as np
import os
from loss import SSIM_L1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TIMESTAMP = "2024-10-24"


def get_model(opt):

    model = convttrans_lstm(opt.d_model, opt.heads, opt.dropout, opt.lamda)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if opt.device == 0:
        model = model.cuda()

    return model


def train_model(model, opt):

    avg_train_losses, avg_val_losses = [], []

    print("training model...")
    for epoch in range(opt.cur_epoch, opt.epochs):
        train_total_loss, val_total_loss = [], []
        model.train()

        train_tqdm = tqdm(opt.trainLoader, leave=False, total=len(opt.trainLoader), bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}')
        for batch_idx, (idx, imgs, _) in enumerate(train_tqdm):
            imgs = imgs[:, ::opt.skip, ...]
            if opt.device == 0:
                imgs = imgs.cuda()

            src1, src2, trg = imgs[:, :opt.slice_len, ...], imgs[:, -opt.slice_len:, ...], imgs[:, opt.slice_len:-opt.slice_len, ...]
            src = torch.cat([src1, src2], 1)
            src_mask, trg_mask = create_convttrans_mask(src[:, :, 0, 0 ,0], trg[:, :, 0, 0 ,0], opt)  # generate masks for source and target

            pred = model(imgs, opt.slice_len, src_mask, trg_mask)

            C, H, W = trg.size(2), trg.size(3), trg.size(4)
            Label, Pred = torch.reshape(trg, (-1, C, H, W)), torch.reshape(pred, (-1, C, H, W))
            Label, Pred = (Label + 1.0) * 0.5, (Pred + 1.0) * 0.5  # Scale to [0, 1]

            opt.optimizer.zero_grad()
            loss = opt.ssim_l1(Label, Pred)
            loss.backward()
            opt.optimizer.step()
            
            train_total_loss.append(loss.item())
            train_tqdm.set_postfix({
                'trainloss': '{:.6f}'.format(loss.item()),
                'epoch': '{:02d}'.format(epoch)
            })


        model.eval()
        with torch.no_grad():
            val_tqdm = tqdm(opt.validLoader, leave=False, total=len(opt.validLoader), bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}')
            for batch_idx, (idx, imgs, _) in enumerate(val_tqdm):
                imgs = imgs[:, ::opt.skip, ...]
                if opt.device == 0:
                    imgs = imgs.cuda()

                src1, src2, trg = imgs[:, :opt.slice_len, ...], imgs[:, -opt.slice_len:, ...], imgs[:, opt.slice_len:-opt.slice_len, ...]
                src = torch.cat([src1, src2], 1)
                src_mask, trg_mask = create_convttrans_mask(src[:, :, 0, 0, 0], trg[:, :, 0, 0, 0], opt)

                pred = model(imgs, opt.slice_len, src_mask, trg_mask)

                Label, Pred = torch.reshape(trg, (-1, C, H, W)), torch.reshape(pred, (-1, C, H, W))
                Label, Pred = (Label + 1.0) * 0.5, (Pred + 1.0) * 0.5

                loss = opt.ssim_l1(Label, Pred)
                val_total_loss.append(loss.item())

                val_tqdm.set_postfix({
                    'validloss': '{:.6f}'.format(loss.item()),
                    'epoch': '{:02d}'.format(epoch)
                })

        # track average losses
        avg_train_losses.append(np.average(train_total_loss))
        avg_val_losses.append(np.average(val_total_loss))

        print(f'[{epoch:>{len(str(opt.epochs))}}/{opt.epochs}] train_loss: {avg_train_losses[-1]:.6f}, valid_loss: {avg_val_losses[-1]:.6f}')

        # change learning rate
        opt.pla_lr_scheduler.step(avg_val_losses[-1])

        # save checkpoint
        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': opt.optimizer.state_dict()
        }


        if not os.path.isdir(opt.save_path):
            os.makedirs(opt.save_path)

        opt.early_stopping(avg_val_losses[-1], model_dict, epoch, opt.save_path)
        if opt.early_stopping.early_stop:
           print("Early stopping")
           break

    # logging
    with open(opt.save_path+'avg_train_losses.txt', 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open(opt.save_path+'avg_valid_losses.txt', 'wt') as f:
        for i in avg_val_losses:
           print(i, file=f)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=[1,256,256])   # model dimensions, AKA, input size [channels, height, width]
    parser.add_argument('-heads', type=int, default=8) # number of attention heads in the transformer
    parser.add_argument('-dropout', type=int, default=0.1) # dropout rate
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-clip_len', type=int, default=25) # total length of the video clip (before skipping frames)
    parser.add_argument('-skip', type=int, default=3)  # frame sampling interval (every nth frame from the clip)
    parser.add_argument('-slice_len', type=int, default=3)  # number of frames to slice after skipping
    parser.add_argument('-lamda', type=float, default=0.85)  # weight to merge forward and backward predictions
    parser.add_argument('-lr', type=float, default=0.0010)   # learning rate

    parser.add_argument('-data_path', type=str, default='./UCSDped2/training_set/')
    parser.add_argument('-save_path', type=str, default='./ckps/') # path to save model checkpoints
    parser.add_argument('-exist_ckp_path', type=str, default='') # path to an existing checkpoint (if available)

    opt = parser.parse_args()

    # Extract channel and imgsize from d_model
    opt.channel = opt.d_model[0]
    opt.imgsize = opt.d_model[1]

    opt.device = 0 if opt.no_cuda is False else -1

    if opt.device == 0:
        assert torch.cuda.is_available()

    # load video data
    trainvalFolder = VAD_train_set(root= opt.data_path,
                                  clip_len=opt.clip_len,  input_channel=opt.channel,
                                  resize_height=opt.imgsize, resize_width=opt.imgsize)

    train_size = int(0.9 * len(trainvalFolder))
    trainFolder, validFolder = torch.utils.data.random_split(trainvalFolder, [train_size, len(trainvalFolder) - train_size])

    opt.trainLoader = torch.utils.data.DataLoader(trainFolder,
                                              batch_size=opt.batch_size,
                                              shuffle=True)
    opt.validLoader = torch.utils.data.DataLoader(validFolder,
                                              batch_size=opt.batch_size,
                                              shuffle=True)

    model = get_model(opt)

    if os.path.exists(opt.exist_ckp_path):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(opt.exist_ckp_path)
        model.load_state_dict(model_info['state_dict'])
        opt.optimizer = torch.optim.Adam(model.parameters())
        opt.optimizer.load_state_dict(model_info['optimizer'])
        opt.cur_epoch = model_info['epoch'] + 1
    else:
        opt.cur_epoch = 0
        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    opt.pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(opt.optimizer, factor=0.5, patience=2, verbose=True)
    opt.ssim_l1 = SSIM_L1(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=opt.channel, alpha=0.5)
    opt.early_stopping = EarlyStopping(patience=10, verbose=True)

    train_model(model, opt)

if __name__ == "__main__":
    main()
