# utils/masking.py - Extract masking functionality

import torch


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.

    Args:
        x (torch.Tensor): Input tensor of shape [N, L, D]
        mask_ratio (float): Ratio of tokens to mask

    Returns:
        tuple: (masked tensor, mask, ids_restore)
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    len_keep = max(1, len_keep)  # Keep at least one token

    # Generate random noise and use it to shuffle indices
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first len_keep tokens
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    # Generate binary mask (1 is masked, 0 is keep)
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore