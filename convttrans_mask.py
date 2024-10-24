import torch
import numpy as np
from torch.autograd import Variable
import random
random.seed(0)
np.random.seed(0)

def nopeak_mask(size, device):
    # generate a no-peak mask to prevent attending to future frames in a sequence.

    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = torch.from_numpy(np_mask) == 0
    np_mask = np_mask.to(device)
    return np_mask


def create_convttrans_mask(src, trg, opt):
    # create source and target masks for the input sequences
    # src: the source sequence
    # trg: the target sequence

    src_mask = (src != 1).unsqueeze(-2)

    trg_mask = None
    if trg is not None:
        trg_mask = (trg != 1).unsqueeze(-2)
        size = trg.size(1)  # sequence length for the no-peak mask
        np_mask = nopeak_mask(size, opt.device)
        trg_mask = trg_mask & np_mask

    return src_mask, trg_mask
