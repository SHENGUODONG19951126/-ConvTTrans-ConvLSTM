import warnings
import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    # create one-dimensional gaussian kernel
    # size: the size of gaussian kernel
    # sigma: standard deviation of normal distribution

    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    # blur input with one-dimensional kernel

    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    # calculate ssim for images X and Y
    # data_range: value range of input images. (usually 1.0 or 255)

    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    # print(mu1.shape)  no padding, image becomes smaller

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def _gauss_l1(X, Y, win):
    # calculate mean absolute error (mae, l1) loss for images X and Y
    # the same gaussian kernel is applied to the l1 loss for better consistency with ssim loss

    win = win.to(X.device, dtype=X.dtype)
    diff = torch.abs(X-Y)
    gaussian_diff = gaussian_filter(diff, win)
    gaussian_l1 = torch.mean(gaussian_diff)

    return gaussian_l1


def ssim_l1(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
    alpha=0.84
):
    # interface of ssim + l1
    # X, Y: a batch of input images X and Y, each (N,C,H,W)
    # data_range: value range of input images. (usually 1.0 or 255)
    # size_average: if size_average=True, ssim of all images will be averaged as a scalar
    # win_size: the size of gaussian kernel
    # win_sigma: standard deviation of normal distribution
    # win: one-dimensional gaussian kernel. if None, a new kernel will be created according to win_size and win_sigma
    # K: scalar constants (K1, K2).
    # nonnegative_ssim: force the ssim response to be nonnegative with relu
    # alpha: coefficient for combining ssim and l1

    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, K=K)

    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        loss_ssim = ssim_per_channel.mean()
    else:
        loss_ssim = ssim_per_channel.mean(1)

    # Gaussian-blurred L1 loss
    loss_L1 = _gauss_l1(X, Y,win=win)
    # combine ssim and l1 losses based on coefficient alpha
    loss_ssim_l1 = (1.0-loss_ssim) * alpha + loss_L1 * (1 - alpha)

    return loss_ssim_l1



class SSIM_L1(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
        alpha=0.5
    ):
        # class for ssim + mae (l1)

        super(SSIM_L1, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims) # (self.win=11)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.alpha = alpha

    def forward(self, X, Y):
        return ssim_l1(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
            alpha=self.alpha
        )
