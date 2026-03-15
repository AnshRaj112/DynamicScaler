import torch
import torch.nn.functional as F

def padding_latents_at_front(source_latents, front_padding_num):
    latents_list = []
    for i in range(front_padding_num):
        latents_list.append(source_latents[:, :, [0]])
    latents_list.append(source_latents)
    padded_latents = torch.cat(latents_list, dim=2)
    return padded_latents


def padding_latents_at_end(source_latents, end_padding_num):
    latents_list = [source_latents]
    for i in range(end_padding_num):
        latents_list.append(source_latents[:, :, [-1]])
    padded_latents = torch.cat(latents_list, dim=2)
    return padded_latents


def resize_video_latent(input_latent, target_height, target_width, mode='bilinear', align_corners=False, gaussian_sigma=0.0):
    """Resize video latent. Use bilinear/bicubic for upscaling to avoid blockiness. Optional gaussian_sigma applies
    light blur to upsampled latent before return to suppress high-frequency artifacts (e.g. 0.5--1.0)."""
    if mode == 'nearest':
        align_corners = None

    batch, channel, frame, h, w = input_latent.shape

    input_latent = input_latent.permute(0, 2, 1, 3, 4)
    video_reshaped = input_latent.view(batch * frame, channel, h, w)
    upsampled = F.interpolate(video_reshaped, size=(target_height, target_width), mode=mode, align_corners=align_corners)
    upsampled = upsampled.view(batch, frame, channel, target_height, target_width)
    upsampled = upsampled.permute(0, 2, 1, 3, 4)

    if gaussian_sigma > 0:
        # Light Gaussian blur along spatial dims to suppress high-freq artifacts (e.g. before re-noising)
        kernel_size = int(2 * max(1, round(2 * gaussian_sigma)) + 1)
        if kernel_size >= 3:
            sigma = [gaussian_sigma] * 2
            upsampled = _gaussian_smooth_5d(upsampled, kernel_size, sigma)

    return upsampled


def _gaussian_smooth_5d(x, kernel_size, sigma):
    """Apply 2D Gaussian blur per frame (B, C, T, H, W)."""
    from torch.nn import Conv2d
    from torch.nn.functional import pad
    B, C, T, H, W = x.shape
    x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    kh, kw = kernel_size, kernel_size
    pad_w = (kw - 1) // 2
    pad_h = (kh - 1) // 2
    g = _gaussian_kernel2d(kernel_size, sigma, x.dtype, x.device)
    g = g.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)  # C, 1, kh, kw
    x_flat = pad(x_flat, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    x_flat = F.conv2d(x_flat, g, padding=0, groups=C)
    x_flat = x_flat.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return x_flat


def _gaussian_kernel2d(kernel_size, sigma, dtype, device):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device, dtype=dtype)
    gx = torch.exp(-ax ** 2 / (2 * sigma[0] ** 2))
    gy = torch.exp(-ax ** 2 / (2 * sigma[1] ** 2))
    g = gx.unsqueeze(1) * gy.unsqueeze(0)
    return g / g.sum()



