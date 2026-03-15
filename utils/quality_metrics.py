"""
Optional quality metrics for DynamicScaler: Q-Align (I/V) and CLIP Score.
Integrate external Q-Align APIs or local models for image/video quality scoring and prompt-frame alignment.
"""

import torch
from typing import List, Optional, Union
import numpy as np


def compute_qalign_image_score(frames: Union[torch.Tensor, List], model_or_api=None) -> Optional[float]:
    """
    Q-Align (I) -- Image quality score for generated frames.
    Purpose: Align with perceived image quality (resolution, sharpness, aesthetics).
    Use for: best-of-k selection or logging.
    Integration: Pass decoded frames (or downsampled subset); call external Q-Align (I) API or local model.
    """
    if model_or_api is None:
        return None  # No scorer configured
    # Placeholder: implement call to Q-Align image scorer
    # return model_or_api.score(frames)
    return None


def compute_qalign_video_score(video_tensor_or_path, model_or_api=None) -> Optional[float]:
    """
    Q-Align (V) -- Video quality score (temporal consistency, motion naturalness).
    Purpose: Score full decoded video for reporting and ablation.
    Integration: Run on saved mp4 or video tensor; return scalar score.
    """
    if model_or_api is None:
        return None
    # Placeholder: implement call to Q-Align video scorer
    return None


def compute_clip_score(
    frames: torch.Tensor,
    prompt: str,
    clip_model=None,
    text_encoder=None,
    image_encoder=None,
    device: Optional[torch.device] = None,
) -> Optional[float]:
    """
    CLIP Score: cosine similarity between text embedding of prompt and (averaged) image embeddings of frames.
    Use for: tuning guidance scale and prompts; monitoring prompt-frame alignment.
    """
    if clip_model is None and (text_encoder is None or image_encoder is None):
        return None
    device = device or (frames.device if isinstance(frames, torch.Tensor) else None)
    if device is None:
        return None
    # Placeholder: encode prompt -> text_emb; encode frames -> image_embs; cosine(text_emb, mean(image_embs))
    # If using OpenCLIP: text_emb = clip.encode_text(...); image_emb = clip.encode_image(...); score = F.cosine_similarity(...)
    return None
