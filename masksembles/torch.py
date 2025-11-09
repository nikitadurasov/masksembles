import torch
from torch import nn

from . import common


class Masksembles2D(nn.Module):
    """
    :class:`Masksembles2D` is high-level class that implements Masksembles approach
    for 2-dimensional inputs (similar to :class:`torch.nn.Dropout2d`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C, H, W)
        * Output: (N, C, H, W) (same shape as input)

    Examples:

    >>> m = Masksembles2D(16, 4, 2.0)
    >>> input = torch.ones([4, 16, 28, 28])
    >>> output = m(input)

    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """
    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()
        self.channels, self.n, self.scale = channels, n, scale
        masks_np = common.generation_wrapper(channels, n, scale)  # numpy float64 by default
        masks = torch.as_tensor(masks_np, dtype=torch.float32)    # make float32 here
        self.register_buffer('masks', masks, persistent=False)    # not trainable, moves with .to()

    def forward(self, inputs):
        # make sure masks match dtype/device (usually already true because of buffer, but safe)
        masks = self.masks.to(dtype=inputs.dtype, device=inputs.device)

        batch = inputs.shape[0]
        # safer split even if batch % n != 0
        chunks = torch.chunk(inputs.unsqueeze(1), self.n, dim=0)  # returns nearly equal chunks
        x = torch.cat(chunks, dim=1).permute(1, 0, 2, 3, 4)       # [n, ?, C, H, W]
        x = x * masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)    # broadcast masks
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return x.squeeze(0)



class Masksembles1D(nn.Module):
    """
    :class:`Masksembles1D` is high-level class that implements Masksembles approach
    for 1-dimensional inputs (similar to :class:`torch.nn.Dropout`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C)
        * Output: (N, C) (same shape as input)

    Examples:

    >>> m = Masksembles1D(16, 4, 2.0)
    >>> input = torch.ones([4, 16])
    >>> output = m(input)


    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()
        self.channels, self.n, self.scale = channels, n, scale
        masks_np = common.generation_wrapper(channels, n, scale)
        masks = torch.as_tensor(masks_np, dtype=torch.float32)
        self.register_buffer('masks', masks, persistent=False)

    def forward(self, inputs):
        masks = self.masks.to(dtype=inputs.dtype, device=inputs.device)

        batch = inputs.shape[0]
        chunks = torch.chunk(inputs.unsqueeze(1), self.n, dim=0)
        x = torch.cat(chunks, dim=1).permute(1, 0, 2)             # [n, ?, C]
        x = x * masks.unsqueeze(1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return x.squeeze(0)
