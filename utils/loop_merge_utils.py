import os

import torch
import torch.nn.functional as F
import imageio
import numpy as np
from PIL import Image


def _temporal_gaussian_smooth(x, sigma):
    """1D Gaussian smoothing along time (dim=2). x: (B, C, T, H, W)."""
    B, C, T, H, W = x.shape
    if T < 3 or sigma <= 0:
        return x
    kernel_size = min(int(2 * round(2 * sigma) + 1), T)
    if kernel_size < 3:
        return x
    t = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-t ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    # conv3d weight: (out_ch, in_ch/groups, kT, kH, kW); 1D along T -> (C, 1, kT, 1, 1), groups=C
    kernel = kernel.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
    pad = kernel_size // 2
    x_pad = F.pad(x, (0, 0, 0, 0, pad, pad), mode='replicate')
    out = F.conv3d(x_pad, kernel, padding=0, groups=C)
    return out


def tensor2image(batch_tensors):
    img_tensor = torch.squeeze(batch_tensors)  # c,h,w

    image = img_tensor.detach().cpu()
    image = torch.clamp(image.float(), -1., 1.)

    image = (image + 1.0) / 2.0
    image = (image * 255).to(torch.uint8).permute(1, 2, 0)  # h,w,c
    image = image.numpy()
    image = Image.fromarray(image)

    return image


def save_decoded_video_latents(decoded_video_latents, output_path, output_name, fps, save_mp4=True, save_gif=True,
                               temporal_smooth_sigma=0.0, batch_decode_size=8):
    """Decode latents to video frames. Optionally apply 1D temporal smoothing to reduce inter-frame flicker.
    Decodes in mini-batches for efficiency."""
    num_frames = decoded_video_latents.shape[2]
    video_frames_tensor_list = []

    for start in range(0, num_frames, batch_decode_size):
        end = min(start + batch_decode_size, num_frames)
        batch_latent = decoded_video_latents[:, :, start:end]  # B, C, batch_frames
        # Decode batch: 2DAE expects [B, C, T, H, W]; single-frame decode per slice
        for k in range(batch_latent.shape[2]):
            frame_tensor = batch_latent[:, :, [k]]
            video_frames_tensor_list.append(frame_tensor)

    # Optional 1D temporal smoothing over decoded frames (Gaussian along time)
    if temporal_smooth_sigma > 0 and len(video_frames_tensor_list) >= 3:
        stacked = torch.cat(video_frames_tensor_list, dim=2)  # B, C, T, H, W
        stacked = _temporal_gaussian_smooth(stacked, temporal_smooth_sigma)
        video_frames_tensor_list = [stacked[:, :, [t]] for t in range(stacked.shape[2])]

    video_frames_img_list = []
    for frame_tensor in video_frames_tensor_list:
        image = tensor2image(frame_tensor)
        video_frames_img_list.append(image)

    print(f"converted {len(video_frames_img_list)} frame tensors")

    if save_mp4:
        mp4_save_path = os.path.join(output_path, f"{output_name}.mp4")
        imageio.mimsave(mp4_save_path, video_frames_img_list, fps=fps)
        print(f"pano video saved to -> {mp4_save_path}")
