import numpy as np
import torch
import os


class EarlyStopping:
    # early stop the training if validation loss doesn't improve after a given patience
    def __init__(self, patience=10, verbose=True):

    # patience: epochs to wait after last time validation loss improved
    # verbose: If True, print a message for each validation loss improvement

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):

        score = -val_loss # smaller validation errors lead to higher scores

        if self.best_score is None:
            self.best_score = score
            self.save_ckp(val_loss, model, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_ckp(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_ckp(self, val_loss, model, epoch, save_path):
        # save the checkpoint when validation loss decreases
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(
            model, save_path +
            "checkpoint_{}_{:.6f}.pth.tar".format(epoch,val_loss))
        self.val_loss_min = val_loss
