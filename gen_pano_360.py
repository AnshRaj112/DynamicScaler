
import math
import os
import shutil
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Literal, Tuple

# Set to True to skip 2x upscale stage and avoid VRAM OOM on 16GB GPUs (e.g. Tesla P100).
# The 2x decode path allocates ~5GB+ during VAE decode; set to False only when you have enough VRAM.
SKIP_2X_UPSCALE_TO_AVOID_OOM = True


@dataclass
class VArgs:

    # ============ CONFIGS ============= #
    seed: int = 2333333
    gpu_id: int = 0
    mode: Literal["static", "static_wrap", "static_model", "dynamic"] = "static"
    # main input panorama image
    pano_image_path: str = "./input/geometric_pattern_1.jpg"

    # ============ STATIC PANORAMA CONFIGS ============= #
    # These are used when mode="static" to write a single panorama image and exit.
    static_out_dir: str = "./results/static_pano"
    static_name: Optional[str] = None  # default: derived from input filename + timestamp
    static_format: Literal["png", "jpg"] = "png"
    static_width: int = 2048
    static_height: int = 1024
    static_fit: Literal["crop", "pad", "stretch"] = "crop"  # how to enforce 2:1 aspect ratio
    write_viewer_html: bool = True  # writes a minimal HTML viewer (CDN-based) alongside the image
    static_model_blend: Literal["mean", "last", "first"] = "mean"
    # Used when mode="static_wrap": make a simple wrap-around panorama from any image (no diffusion).
    wrap_overlap_px: int = 64  # seam blending overlap in pixels (in output image space)
    wrap_mirror_edges: bool = True  # mirror edges before blending to hide harsh seams
    auto_phi_prompts: bool = True  # auto-generate phi_prompt_dict from input image + base prompt
    auto_phi_prompt_mode: Literal["uniform", "heuristic_bands"] = "heuristic_bands"
    launch_ui: bool = False  # launch a simple upload UI (Gradio) instead of CLI run
    ui_share: bool = False  # Gradio share link (useful on Kaggle)
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860

    # Default prompt & phi_prompt_dict used by the dynamic pipeline; safe via default_factory.
    prompt: str = (
        "Clean high-contrast geometric tessellation pattern, sharp lines, crisp dots, "
        "uniform spacing, seamless repeating texture, flat lighting, minimal noise"
    )
    phi_prompt_dict: Dict[int, str] = field(
        default_factory=lambda: {
            90: "Clean geometric pattern, bright white background, sharp black lines",
            75: "Clean geometric pattern, sharp linework, uniform spacing",
            60: "Geometric tessellation, crisp edges, high contrast",
            45: "Seamless geometric repeating pattern, sharp lines and dots",
            0: "Seamless geometric repeating pattern, sharp lines and dots, high contrast",
            -45: "Geometric pattern, crisp edges, uniform spacing, minimal blur",
            -60: "Clean geometric pattern, sharp lines, high contrast",
            -75: "Clean geometric pattern, bright background, sharp black linework",
            -90: "Clean geometric pattern, bright background, sharp black linework",
        }
    )

    total_f: int = 16
    do_upscale: bool = True
    upscale_factor: int = 2

    # ============ ADVANCED CONFIGS ============= #
    phi_num:            int = 6     # 6
    view_fov: int           = 120
    denoise_to_step:    int = 15    # 5
    skip_time_step:     int = -1
    loop_step_theta:    int = 10
    predenoised_SP_latent_path: str = None 
    predenoised_SW_1x_latent_path: str = None
    dock_at_f: bool = True
    loop_step_frame: int = 8
    skip_1x = False
    loop_step_hw: int = 16
    merge_renoised_overlap_latent_ratio =  1 
    merge_denoised = True
    max_merge_denoised_overlap_latent_ratio = 0.5 
    _merge_prev_step = 20
    temporal_guidance_scale: Optional[float] = None  # set to e.g. 0.5 for temporal CFG in sphere pipeline
    low_memory: bool = False  # empty_cache between stages for P100/16GB


    @classmethod
    def from_args(cls):
        from dataclasses import MISSING

        parser = argparse.ArgumentParser()
        # IMPORTANT: fields using default_factory (e.g. dict/list) must not be registered with argparse
        # with a MISSING default, otherwise argparse will set them to a sentinel and override the factory.
        skipped_factory_fields = set()
        for field_name, field_def in cls.__dataclass_fields__.items():
            default = field_def.default
            if default is MISSING and field_def.default_factory is not MISSING:
                skipped_factory_fields.add(field_name)
                continue
            # bool flags need special handling; argparse's bool("False") == True is a common pitfall.
            if isinstance(default, bool):
                parser.add_argument(
                    f"--{field_name}",
                    action=argparse.BooleanOptionalAction,
                    default=default,
                    help=f"{field_name} (default: {default})",
                )
            else:
                parser.add_argument(
                    f"--{field_name}",
                    type=type(default) if default is not None else str,
                    default=default,
                    help=f"{field_name} (default: {default})",
                )
        args = parser.parse_args()
        kwargs = vars(args)
        for k in skipped_factory_fields:
            kwargs.pop(k, None)
        return cls(**kwargs)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _fit_to_aspect_2_to_1(
    img,
    fit: Literal["crop", "pad", "stretch"] = "crop",
    pad_color: Tuple[int, int, int] = (0, 0, 0),
):
    """
    Enforce 2:1 aspect ratio for equirect panoramas.

    - crop: center-crop to 2:1 without distortion
    - pad: letterbox/pillarbox to 2:1 without distortion
    - stretch: resize to 2:1 with distortion
    """
    import numpy as np

    h, w = img.shape[:2]
    target_aspect = 2.0
    curr_aspect = w / max(1, h)

    if abs(curr_aspect - target_aspect) < 1e-3:
        return img

    if fit == "stretch":
        # just handle at resize stage later
        return img

    if fit == "crop":
        if curr_aspect > target_aspect:
            # too wide -> crop width
            new_w = int(round(h * target_aspect))
            x0 = max(0, (w - new_w) // 2)
            return img[:, x0 : x0 + new_w]
        else:
            # too tall -> crop height
            new_h = int(round(w / target_aspect))
            y0 = max(0, (h - new_h) // 2)
            return img[y0 : y0 + new_h, :]

    if fit == "pad":
        if curr_aspect > target_aspect:
            # too wide -> pad height
            new_h = int(round(w / target_aspect))
            canvas = np.empty((new_h, w, 3), dtype=img.dtype)
            canvas[...] = np.array(pad_color, dtype=img.dtype)
            y0 = max(0, (new_h - h) // 2)
            canvas[y0 : y0 + h, :, :] = img
            return canvas
        else:
            # too tall -> pad width
            new_w = int(round(h * target_aspect))
            canvas = np.empty((h, new_w, 3), dtype=img.dtype)
            canvas[...] = np.array(pad_color, dtype=img.dtype)
            x0 = max(0, (new_w - w) // 2)
            canvas[:, x0 : x0 + w, :] = img
            return canvas

    raise ValueError(f"Unknown fit mode: {fit}")


def _write_static_viewer_html(output_dir: str, image_filename: str, title: str = "Static 360 Panorama") -> str:
    """
    Writes a minimal pannellum-based viewer HTML (uses CDN).
    """
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css" />
    <style>
      html, body {{ height: 100%; margin: 0; }}
      #pano {{ width: 100%; height: 100%; }}
    </style>
  </head>
  <body>
    <div id="pano"></div>
    <script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
    <script>
      pannellum.viewer('pano', {{
        "type": "equirectangular",
        "panorama": "{image_filename}",
        "autoLoad": true,
        "showZoomCtrl": true
      }});
    </script>
  </body>
</html>
"""
    out_path = os.path.join(output_dir, "viewer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def _simple_scene_tokens_from_band(mean_rgb):
    """
    Extremely lightweight heuristic to turn a mean RGB into coarse descriptors.
    This is intentionally simple (works offline, no extra deps).
    """
    r, g, b = [float(x) for x in mean_rgb]
    tokens = []
    # brightness
    v = (r + g + b) / 3.0
    if v > 200:
        tokens.append("bright")
    elif v < 60:
        tokens.append("dark")
    else:
        tokens.append("well-lit")
    # dominant hue-ish
    if b > r + 20 and b > g + 20:
        tokens.append("blue")
        if v > 140:
            tokens.append("sky-like")
    elif g > r + 20 and g > b + 20:
        tokens.append("green")
        tokens.append("foliage-like")
    elif r > g + 20 and r > b + 20:
        tokens.append("warm")
    # saturation-ish
    spread = max(r, g, b) - min(r, g, b)
    if spread < 18:
        tokens.append("low-saturation")
    else:
        tokens.append("high-contrast")
    return tokens


def build_phi_prompt_dict(
    image_path: str,
    base_prompt: str,
    mode: Literal["uniform", "heuristic_bands"] = "heuristic_bands",
) -> Dict[int, str]:
    """
    Build phi-specific prompts at runtime so users don't edit code.

    - uniform: use the same base prompt for all phi
    - heuristic_bands: analyze top/middle/bottom horizontal bands and produce
      slightly different prompts for sky/equator/ground latitudes.
    """
    if mode == "uniform":
        return {90: base_prompt, 75: base_prompt, 60: base_prompt, 45: base_prompt, 0: base_prompt,
                -45: base_prompt, -60: base_prompt, -75: base_prompt, -90: base_prompt}

    import numpy as np
    import imageio.v3 as iio

    img = iio.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.float32)
    h = img.shape[0]
    # bands: top 25%, middle 50%, bottom 25%
    top = img[: max(1, h // 4)]
    mid = img[max(1, h // 4): max(2, (3 * h) // 4)]
    bot = img[max(2, (3 * h) // 4):]

    top_tokens = _simple_scene_tokens_from_band(top.reshape(-1, 3).mean(axis=0))
    mid_tokens = _simple_scene_tokens_from_band(mid.reshape(-1, 3).mean(axis=0))
    bot_tokens = _simple_scene_tokens_from_band(bot.reshape(-1, 3).mean(axis=0))

    def enrich(prompt, tokens):
        # Keep user prompt primary; add a few coarse stabilizers.
        extra = ", ".join(tokens[:4])
        return f"{prompt}, {extra}" if extra else prompt

    top_prompt = enrich(base_prompt, ["upper hemisphere"] + top_tokens)
    mid_prompt = enrich(base_prompt, ["horizon band"] + mid_tokens)
    bot_prompt = enrich(base_prompt, ["lower hemisphere"] + bot_tokens)

    return {
        90: top_prompt,
        75: top_prompt,
        60: top_prompt,
        45: mid_prompt,
        0: mid_prompt,
        -45: bot_prompt,
        -60: bot_prompt,
        -75: bot_prompt,
        -90: bot_prompt,
    }


def run_static_panorama(vargs: "VArgs") -> str:
    import numpy as np
    import imageio.v3 as iio

    in_path = vargs.pano_image_path
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input image not found: {in_path}")

    out_dir = _ensure_dir(vargs.static_out_dir)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = vargs.static_name or f"{stem}_pano_{ts}"
    ext = "jpg" if vargs.static_format == "jpg" else "png"
    out_img = os.path.join(out_dir, f"{name}.{ext}")

    img = iio.imread(in_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    img = _fit_to_aspect_2_to_1(img, fit=vargs.static_fit)

    if vargs.static_fit == "stretch":
        # enforce 2:1 at resize time
        pass

    # Resize to requested output size
    try:
        import cv2

        img_resized = cv2.resize(img, (int(vargs.static_width), int(vargs.static_height)), interpolation=cv2.INTER_LANCZOS4)
    except Exception:
        # Fallback to numpy-based nearest if cv2 isn't available
        img_resized = img
        if (img.shape[1], img.shape[0]) != (vargs.static_width, vargs.static_height):
            yy = (np.linspace(0, img.shape[0] - 1, vargs.static_height)).astype(np.int64)
            xx = (np.linspace(0, img.shape[1] - 1, vargs.static_width)).astype(np.int64)
            img_resized = img[yy][:, xx]

    iio.imwrite(out_img, img_resized)

    if vargs.write_viewer_html:
        _write_static_viewer_html(out_dir, os.path.basename(out_img), title=os.path.basename(out_img))

    print(f"[static] wrote panorama image: {out_img}")
    return out_img


def run_static_wrap_panorama(vargs: "VArgs") -> str:
    """
    Create a simple, understandable equirect panorama from a single image by:
    - enforcing 2:1 aspect output (WxH)
    - resizing input to output height
    - tiling horizontally to fill output width (wrap-around)
    - blending the left/right seam over wrap_overlap_px

    This does NOT use any diffusion model and is designed to work reliably on low-VRAM GPUs/CPU.
    """
    import numpy as np
    import imageio.v3 as iio

    in_path = vargs.pano_image_path
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input image not found: {in_path}")

    out_dir = _ensure_dir(vargs.static_out_dir)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = vargs.static_name or f"{stem}_wrap_{ts}"
    ext = "jpg" if vargs.static_format == "jpg" else "png"
    out_img = os.path.join(out_dir, f"{name}.{ext}")

    img = iio.imread(in_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    # Ensure uint8 for predictable blending
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    H_out = int(vargs.static_height)
    W_out = int(vargs.static_width)
    if W_out != 2 * H_out:
        # equirect should be 2:1; force it for predictable viewing
        W_out = 2 * H_out

    # Resize input to output height, keep aspect
    h, w = img.shape[:2]
    scale = H_out / max(1, h)
    new_w = max(1, int(round(w * scale)))
    try:
        import cv2

        img_rs = cv2.resize(img, (new_w, H_out), interpolation=cv2.INTER_LANCZOS4)
    except Exception:
        yy = (np.linspace(0, h - 1, H_out)).astype(np.int64)
        xx = (np.linspace(0, w - 1, new_w)).astype(np.int64)
        img_rs = img[yy][:, xx]

    # Tile horizontally to fill width
    reps = int(math.ceil(W_out / img_rs.shape[1])) + 2
    tiled = np.concatenate([img_rs] * reps, axis=1)
    start_x = (tiled.shape[1] - W_out) // 2
    pano = tiled[:, start_x : start_x + W_out].copy()

    # Blend seam between left and right edges
    overlap = max(0, int(vargs.wrap_overlap_px))
    overlap = min(overlap, W_out // 4)
    if overlap >= 4:
        left = pano[:, :overlap].astype(np.float32)
        right = pano[:, -overlap:].astype(np.float32)

        if vargs.wrap_mirror_edges:
            # Mirror to reduce abrupt transitions
            left_src = left[:, ::-1]
            right_src = right[:, ::-1]
        else:
            left_src = left
            right_src = right

        alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[None, :, None]
        # Make left edge gradually become right content (and vice versa) to enforce cyclic consistency
        pano[:, :overlap] = (left_src * (1 - alpha) + right_src * alpha).astype(np.uint8)
        pano[:, -overlap:] = (right_src * (1 - alpha[:, ::-1]) + left_src * alpha[:, ::-1]).astype(np.uint8)

    iio.imwrite(out_img, pano)
    if vargs.write_viewer_html:
        _write_static_viewer_html(out_dir, os.path.basename(out_img), title=os.path.basename(out_img))

    print(f"[static_wrap] wrote panorama image: {out_img}")
    return out_img


def launch_upload_ui(default_out_dir: str = "./results/static_pano"):
    """
    Minimal runtime UI for uploading an image and generating a static panorama.
    Designed to work on low-memory GPUs/CPU by defaulting to mode="static_wrap".
    """
    try:
        import gradio as gr
    except Exception as e:
        msg = str(e)
        if "ImageClassificationOutputElement" in msg or "huggingface_hub" in msg:
            raise RuntimeError(
                "Gradio failed to import due to a version mismatch with `huggingface_hub`.\n"
                "On Kaggle/Colab, fix it by upgrading both packages:\n\n"
                "  pip install -U gradio huggingface_hub\n\n"
                "Then re-run:\n"
                "  python gen_pano_360.py --launch_ui --ui_share\n"
            ) from e
        raise RuntimeError(
            "Gradio is required for --launch_ui. Install it with:\n\n"
            "  pip install -U gradio\n"
        ) from e

    import tempfile
    import imageio.v3 as iio

    def _run(image, mode, prompt, out_dir, height, overlap_px, mirror_edges, write_viewer):
        if image is None:
            raise gr.Error("Please upload an image.")

        os.makedirs(out_dir, exist_ok=True)
        # Save uploaded image to a temp file inside out_dir so viewer.html can reference it if needed.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_path = os.path.join(out_dir, f"uploaded_{ts}.png")
        # gradio gives numpy array (H,W,3) uint8 typically
        iio.imwrite(tmp_path, image)

        # Build args for the run
        va = VArgs()
        va.mode = mode
        va.pano_image_path = tmp_path
        va.static_out_dir = out_dir
        va.static_height = int(height)
        va.static_width = int(2 * int(height))
        va.wrap_overlap_px = int(overlap_px)
        va.wrap_mirror_edges = bool(mirror_edges)
        va.write_viewer_html = bool(write_viewer)
        if prompt:
            va.prompt = prompt

        if va.mode == "static":
            out_img = run_static_panorama(va)
        elif va.mode == "static_wrap":
            out_img = run_static_wrap_panorama(va)
        else:
            raise gr.Error("UI currently supports only static/static_wrap (no model).")

        viewer_path = os.path.join(out_dir, "viewer.html")
        viewer_note = viewer_path if (va.write_viewer_html and os.path.exists(viewer_path)) else ""
        return out_img, viewer_note

    with gr.Blocks(title="DynamicScaler — Static Panorama") as demo:
        gr.Markdown("## Static panorama generator\nUpload an image and generate an equirectangular (2:1) panorama.\n\nRecommended on Kaggle/P100: **static_wrap** (fast, deterministic).")
        with gr.Row():
            inp = gr.Image(label="Upload image", type="numpy")
            out = gr.Image(label="Output panorama", type="filepath")

        with gr.Row():
            mode = gr.Dropdown(choices=["static_wrap", "static"], value="static_wrap", label="Mode")
            height = gr.Slider(256, 2048, value=1024, step=64, label="Panorama height (width = 2×height)")

        with gr.Row():
            overlap_px = gr.Slider(0, 512, value=128, step=8, label="Seam blend overlap (px)")
            mirror_edges = gr.Checkbox(value=True, label="Mirror edges before blending")

        with gr.Row():
            out_dir = gr.Textbox(value=default_out_dir, label="Output folder")
            write_viewer = gr.Checkbox(value=True, label="Write viewer.html (Pannellum)")

        prompt = gr.Textbox(value="", label="Optional prompt (stored, not used by static_wrap)")
        btn = gr.Button("Generate panorama")
        viewer = gr.Textbox(value="", label="viewer.html path (open in browser)", interactive=False)

        btn.click(
            _run,
            inputs=[inp, mode, prompt, out_dir, height, overlap_px, mirror_edges, write_viewer],
            outputs=[out, viewer],
        )

    return demo

@dataclass
class RunArgs:
    config: str = "configs/inference_i2v_512_v1.0.yaml"
    base_ckpt_path: str = "./videocrafter_models/i2v_512_v1/model.ckpt"
    seed: int = 2333
    num_inference_steps: int = None
    total_video_length: int = 64
    num_processes: int = 1
    rank: int = 0
    height: int = 320
    width: int = 512
    save_frames: bool = True
    fps: int = 8
    unconditional_guidance_scale: float = 7.5
    lookahead_denoising: bool = False
    eta: float = 1.0
    output_dir: Optional[str] = None
    use_mp4: bool = True


def run_dynamic_video_generation(vargs: "VArgs"):
    # Heavy imports are intentionally inside this function so mode="static" stays lightweight.
    import gc
    from collections import OrderedDict

    import torch
    from pytorch_lightning import seed_everything
    from omegaconf import OmegaConf

    from scripts.evaluation.funcs import load_model_checkpoint
    from utils.utils import instantiate_from_config, create_dir
    from utils.loop_merge_utils import save_decoded_video_latents, tensor2image
    from utils.diffusion_utils import resize_video_latent
    from pipeline.i2v_sphere_panorama_pipeline import VC2_Pipeline_I2V_SpherePano
    from pipeline.scheduler import lvdm_DDIM_Scheduler

    def main(run_args: RunArgs, prompt, image_path, image_folder,
         use_fp16=False, save_latents=False,
         pano_image_path=None,
         loop_step=None,
         num_windows_h=None,
         num_windows_w=None,
         num_windows_f=None,
         use_skip_time=False,
         skip_time_step_idx=0,
         progressive_skip=False,
         equirect_width=None,
         equirect_height=None,
         phi_theta_dict=None,
         phi_prompt_dict: dict = None,
         view_fov=None,
         loop_step_theta=None,
         merge_renoised_overlap_latent_ratio=None,
         paste_on_static=None,
         downsample_factor_before_vae_decode=None,
         view_get_scale_factor=None,
         view_set_scale_factor=None,
         denoise_to_step=None,
         num_windows_h_2=None,
         num_windows_w_2=None,
         total_f=None,
         dock_at_f=None,
         loop_step_frame=None,
         overlap_ratio_list_1_f=None,
         overlap_ratio_list_2_f=None,
         upscale_factor=None,
         merge_prev_denoised_ratio_list=None,
         temporal_guidance_scale=None,
         project_name="",
         project_folder=None):

        print(f"==========================\n"
          f"CURR GPU: {os.environ['CUDA_VISIBLE_DEVICES']}, SEED: {run_args.seed}\n"
          f"==========================\n")

        seed_everything(run_args.seed)

        output_dir, tmp_dir = create_dir(project_id=project_name, project_folder=project_folder)


        source_file_path = __file__
        destination_file_path = os.path.join(output_dir, "_src_script.py")
        shutil.copy(source_file_path, destination_file_path)

        src_path = os.path.join(output_dir, 'src')
        os.makedirs(src_path)

        src_dirs_list = ["./utils", "./pipeline"]

        for src_dir in src_dirs_list:
            src_dir_abs = os.path.abspath(src_dir)
            target_save_dir = os.path.join(src_path, src_dir)
            target_save_dir_abs = os.path.abspath(target_save_dir)
            shutil.copytree(src_dir_abs, target_save_dir_abs, ignore=shutil.ignore_patterns('*.pyc'))

        config = OmegaConf.load(run_args.config)
        model_config = config.pop("model", OmegaConf.create())

        if use_fp16: 
            model_config['params']['unet_config']['params']['use_fp16'] = True

        model = instantiate_from_config(model_config)
        model = model.cuda()
        assert os.path.exists(run_args.base_ckpt_path), f"Error: checkpoint [{run_args.base_ckpt_path}] Not Found!"

        model = load_model_checkpoint(model, run_args.base_ckpt_path)
        model.eval()

        if use_fp16:
            model.to(torch.float16)


        scheduler = lvdm_DDIM_Scheduler(model=model)

        pipeline = VC2_Pipeline_I2V_SpherePano(pretrained_t2v=model,
                                           scheduler=scheduler,
                                           model_config=model_config)
        pipeline.to(model.device)

        if use_fp16:
            pipeline.to(model.device, torch_dtype=torch.float16)

    # sample shape
        assert (run_args.height % 16 == 0) and (run_args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"

        channels = model.channels

        batch_size = 1
        prompts = [prompt]
        img_cond_path = [pano_image_path]

        print("==== Sphere Panorama Shift Windows Sample ====")

        if vargs.predenoised_SP_latent_path is None:
            sphere_SW_latent, sphere_SW_denoised = pipeline.basic_sample_shift_shpere_panorama(
            prompt=prompts,
            img_cond_path=img_cond_path,
            height=run_args.height,
            width=run_args.width,
            frames=16, 
            fps=run_args.fps,
            guidance_scale=run_args.unconditional_guidance_scale,

            init_panorama_latent=None,
            use_skip_time=use_skip_time,
            skip_time_step_idx=skip_time_step_idx,
            progressive_skip=progressive_skip,

            loop_step=loop_step,
            pano_image_path=pano_image_path,

            total_f=total_f,
            dock_at_f=dock_at_f,
            overlap_ratio_list_f=overlap_ratio_list_1_f,
            loop_step_frame=loop_step_frame,

            equirect_width=equirect_width * upscale_factor if vargs.skip_1x else equirect_width * 2,    # 避免过大导致motion偏小?
            equirect_height=equirect_height * upscale_factor if vargs.skip_1x else equirect_height * 2,
            phi_theta_dict=phi_theta_dict,
            phi_prompt_dict=phi_prompt_dict,
            view_fov=view_fov,
            loop_step_theta=loop_step_theta,
            merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,

            paste_on_static=paste_on_static,

            view_get_scale_factor=view_get_scale_factor,
            view_set_scale_factor=view_set_scale_factor,

            denoise_to_step=denoise_to_step,
            merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,
            temporal_guidance_scale=temporal_guidance_scale,

            downsample_factor_before_vae_decode=downsample_factor_before_vae_decode,
            latents=None,
            num_inference_steps=run_args.num_inference_steps,
            num_videos_per_prompt=1,
            generator_seed=run_args.seed,

            output_type = "latent"
            )

            if save_latents:
                torch.save(sphere_SW_latent, os.path.join(output_dir, "sphere_SW_latent.pt"))
            if getattr(vargs, 'low_memory', False):
                torch.cuda.empty_cache()
                gc.collect()
        else:
            print(f"loading SW latent from {vargs.predenoised_SP_latent_path}")
            sphere_SW_latent = torch.load(vargs.predenoised_SP_latent_path)

        # ----- STATIC MODEL-BASED PANORAMA BRANCH -----
        if vargs.mode == "static_model":
            print("==== Static sphere panorama decode (no video) ====")
            # Decode denoised sphere latents to RGB frames: B, C, T, H, W
            decoded = model.decode_first_stage_2DAE(sphere_SW_denoised)
            if vargs.static_model_blend == "mean":
                blended = decoded.mean(dim=2, keepdim=True)  # B, C, 1, H, W
                pano_img = tensor2image(blended)
            elif vargs.static_model_blend == "last":
                pano_img = tensor2image(decoded[:, :, [-1]])
            else:
                pano_img = tensor2image(decoded[:, :, [0]])
            out_name = "sphere_pano_static.png"
            out_path = os.path.join(output_dir, out_name)
            pano_img.save(out_path)
            print(f"[static_model] wrote panorama image: {out_path}")
            return

        print("==== Normal Plane Shift Windows Sample ====")

        if not vargs.skip_1x:

            if vargs.predenoised_SW_1x_latent_path is None:

                downsampled_sphere_SW_latent = resize_video_latent(input_latent=sphere_SW_latent.clone(), mode="bilinear",
                                                               target_height=int(equirect_height // downsample_factor_before_vae_decode // 8),
                                                               target_width=int(equirect_width // downsample_factor_before_vae_decode // 8))

                basic_SW_video_frames, basic_SW_latent = pipeline.basic_sample_shift_multi_windows(
                prompt=prompts,
                img_cond_path=img_cond_path,
                height=run_args.height,
                width=run_args.width,
                frames=16, 
                fps=run_args.fps,
                guidance_scale=run_args.unconditional_guidance_scale,

                init_panorama_latent=downsampled_sphere_SW_latent,
                use_skip_time=True,
                skip_time_step_idx=denoise_to_step,
                progressive_skip=False,
                total_h=int(equirect_height // downsample_factor_before_vae_decode),
                total_w=int(equirect_width // downsample_factor_before_vae_decode),
                num_windows_h=num_windows_h_2,
                num_windows_w=num_windows_w_2,
                num_windows_f=num_windows_f,
                loop_step=loop_step,
                pano_image_path=pano_image_path,

                total_f=total_f,
                dock_at_f=dock_at_f,
                overlap_ratio_list_f=overlap_ratio_list_1_f,
                loop_step_frame=loop_step_frame,

                merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,

                latents=None,
                num_inference_steps=run_args.num_inference_steps,
                num_videos_per_prompt=1,
                generator_seed=run_args.seed,
                )

                if save_latents:
                    torch.save(basic_SW_latent, os.path.join(output_dir, f"basic_SW_latent-{project_name}.pt"))
                    torch.save(basic_SW_video_frames, os.path.join(output_dir, f"basic_SW_video_frames-{project_name}.pt"))

                save_decoded_video_latents(decoded_video_latents=basic_SW_video_frames,
                                       output_path=output_dir,
                                       output_name="shift_windows",
                                       fps=run_args.fps)
                if getattr(vargs, 'low_memory', False):
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                print(f"loading basic_SW_latent from : {vargs.predenoised_SW_1x_latent_path}")
                basic_SW_latent = torch.load(vargs.predenoised_SW_1x_latent_path)


        if vargs.do_upscale and not SKIP_2X_UPSCALE_TO_AVOID_OOM:
            print("==== Upscale Shift Windows Sample ====")

            if vargs.skip_1x:
                mixed_upscale_latent = sphere_SW_latent
            else:

                upsampled_SW_latent = resize_video_latent(input_latent=basic_SW_latent.clone(), mode="bicubic",
                                                      target_height=int(equirect_height // downsample_factor_before_vae_decode // 8 * upscale_factor),
                                                      target_width=int(equirect_width // downsample_factor_before_vae_decode // 8 * upscale_factor))
                pipeline.scheduler.make_schedule(run_args.num_inference_steps)
                renoised_basic_SW_latent = pipeline.scheduler.re_noise(x_a=upsampled_SW_latent,
                                                                   step_a=0,
                                                                   step_b=run_args.num_inference_steps-denoise_to_step)

                mixed_upscale_latent = renoised_basic_SW_latent 

            basic_SW_video_frames_2x, basic_SW_latent_2x = pipeline.basic_sample_shift_multi_windows(
            prompt=prompts,
            img_cond_path=img_cond_path,
            height=run_args.height,
            width=run_args.width,
            frames=16, 
            fps=run_args.fps,
            guidance_scale=run_args.unconditional_guidance_scale,

            init_panorama_latent=mixed_upscale_latent,
            use_skip_time=True,
            skip_time_step_idx=denoise_to_step,
            progressive_skip=False,
            total_h=int(equirect_height // downsample_factor_before_vae_decode * upscale_factor),
            total_w=int(equirect_width // downsample_factor_before_vae_decode * upscale_factor),
            num_windows_h=num_windows_h_2 * upscale_factor,
            num_windows_w=num_windows_w_2 * upscale_factor,
            num_windows_f=num_windows_f,
            loop_step=loop_step,
            pano_image_path=pano_image_path,

            merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,

            total_f=total_f,
            dock_at_f=dock_at_f,
            overlap_ratio_list_f=overlap_ratio_list_2_f,
            loop_step_frame=loop_step_frame,

            latents=None,
            num_inference_steps=run_args.num_inference_steps,
            num_videos_per_prompt=1,
            generator_seed=run_args.seed,
            )

            if save_latents:
                torch.save(basic_SW_latent_2x, os.path.join(output_dir, f"denoised_latent2x-{project_name}.pt"))

            save_decoded_video_latents(decoded_video_latents=basic_SW_video_frames_2x,
                                   output_path=output_dir,
                                   output_name=f"SW_2X_{project_name}",
                                   fps=run_args.fps)
        elif vargs.do_upscale and SKIP_2X_UPSCALE_TO_AVOID_OOM:
            print("==== 2x upscale skipped (SKIP_2X_UPSCALE_TO_AVOID_OOM=True) to avoid VRAM OOM on 16GB GPUs ====")

    # ---- original script entrypoint logic (kept intact) ----
    run_args = RunArgs()
    run_args.seed = vargs.seed
    run_args.base_ckpt_path = "./videocrafter_models/i2v_512_v1/model.ckpt"
    # Fewer steps and smaller UNet resolution in low-memory mode to avoid OOM on 16GB GPUs
    if vargs.low_memory:
        run_args.num_inference_steps = 24
        run_args.height = 256
        run_args.width = 384
    else:
        run_args.num_inference_steps = 48
    run_args.fps = 8
    prompt = vargs.prompt

    image_path = None
    image_folder = None
    pano_image_path = vargs.pano_image_path

    loop_step = vargs.loop_step_hw
    num_windows_h = None
    num_windows_w = None
    num_windows_f = 1

    # for Sphere Pano Denoising Phrase
    skip_time_step_idx = vargs.skip_time_step
    if skip_time_step_idx >= 0:
        use_skip_time = True
        progressive_skip = True
    else:
        use_skip_time = False
        skip_time_step_idx = 0
        progressive_skip = False

    denoise_to_step = vargs.denoise_to_step

    # Sphere Pano Basic
    downsample_factor_before_vae_decode = 1
    # Use a smaller base equirect size in low-memory mode so the sphere stage (which uses *2) fits in 16GB.
    if vargs.low_memory:
        equirect_width = 512
        equirect_height = 256
    else:
        equirect_width = int(1024 * downsample_factor_before_vae_decode)
        equirect_height = int(512 * downsample_factor_before_vae_decode)

    view_fov = vargs.view_fov

    phi_0_first = False
    phi_num = vargs.phi_num

    # Latitude-adaptive theta count: N_theta(phi) = max(1, floor(N_0 * cos(phi * pi/180))) for uniform solid-angle coverage
    def n_theta_for_phi(phi_deg, n0):
        n = max(1, int(n0 * math.cos(phi_deg * math.pi / 180.0)))
        return max(1, n)

    phi_theta_dict = {
        90: [0],
        -90: [0],
        75: [360 * t // n_theta_for_phi(75, phi_num) for t in range(n_theta_for_phi(75, phi_num))],
        -75: [360 * t // n_theta_for_phi(-75, phi_num) for t in range(n_theta_for_phi(-75, phi_num))],
        60: [360 * t // n_theta_for_phi(60, phi_num) for t in range(n_theta_for_phi(60, phi_num))],
        -60: [360 * t // n_theta_for_phi(-60, phi_num) for t in range(n_theta_for_phi(-60, phi_num))],
        45: [360 * t // n_theta_for_phi(45, phi_num) for t in range(n_theta_for_phi(45, phi_num))],
        -45: [360 * t // n_theta_for_phi(-45, phi_num) for t in range(n_theta_for_phi(-45, phi_num))],
        0: [360 * t // phi_num for t in range(phi_num)],  # equator: full N0
    }

    if phi_0_first:
        phi_theta_dict = OrderedDict(reversed(list(phi_theta_dict.items())))
    phi_prompt_dict = vargs.phi_prompt_dict

    paste_on_static = True
    loop_step_theta = vargs.loop_step_theta  # Sphere Pano SW
    merge_renoised_overlap_latent_ratio = vargs.merge_renoised_overlap_latent_ratio

    view_get_scale_factor = 1
    view_set_scale_factor = 1

    num_windows_h_2 = 2
    num_windows_w_2 = 2

    # total_f must be >= frames (16) for the sphere sampler; clamp to avoid ValueError.
    total_f = max(int(vargs.total_f), 16)
    dock_at_f = vargs.dock_at_f
    loop_step_frame = vargs.loop_step_frame

    # Smooth cosine annealing for overlap ratio (reduces temporal seam/flicker at transition)
    r_max, r_min = 0.75, 0.5
    T_steps = run_args.num_inference_steps
    overlap_ratio_list_1_f = [
        r_max - (r_max - r_min) / 2 * (1 - math.cos(math.pi * i / max(1, T_steps - 1)))
        for i in range(T_steps)
    ]
    print(f"overlap_ratio_list for 1x F (cosine): {overlap_ratio_list_1_f[:3]}...{overlap_ratio_list_1_f[-3:]}")

    overlap_ratio_list_2_f = [
        r_max - (r_max - r_min) / 2 * (1 - math.cos(math.pi * i / max(1, T_steps - 1)))
        for i in range(T_steps)
    ]
    print(f"overlap_ratio_list for 2x F (cosine): {overlap_ratio_list_2_f[:3]}...{overlap_ratio_list_2_f[-3:]}")

    # Cosine decay for merge_prev_denoised (strong early, drops in refinement phase)
    if vargs.merge_denoised:
        merge_prev_denoised_ratio_list = [
            vargs.max_merge_denoised_overlap_latent_ratio / 2 * (1 + math.cos(math.pi * t / max(1, vargs._merge_prev_step)))
            for t in range(vargs._merge_prev_step)
        ] + [0.0] * (run_args.num_inference_steps - vargs._merge_prev_step)
        print(
            f"merge_prev_denoised_ratio_list (cosine decay): "
            f"{merge_prev_denoised_ratio_list[:3]}...{merge_prev_denoised_ratio_list[vargs._merge_prev_step-3:vargs._merge_prev_step]}"
        )
    else:
        merge_prev_denoised_ratio_list = None

    upscale_factor = vargs.upscale_factor

    PROJECT_FOLDER = "TEST_1"
    PROJECT_NOTE = f"s-{vargs.seed}"
    PROJECT_NAME = f"{PROJECT_NOTE}"

    save_latents = True

    main(
        run_args,
        prompt,
        image_path,
        image_folder,
        project_name=PROJECT_NAME,
        project_folder=PROJECT_FOLDER,
        pano_image_path=pano_image_path,
        loop_step=loop_step,
        num_windows_h=num_windows_h,
        num_windows_w=num_windows_w,
        num_windows_f=num_windows_f,
        use_skip_time=use_skip_time,
        skip_time_step_idx=skip_time_step_idx,
        progressive_skip=progressive_skip,
        equirect_width=equirect_width,
        equirect_height=equirect_height,
        phi_theta_dict=phi_theta_dict,
        phi_prompt_dict=phi_prompt_dict,
        view_fov=view_fov,
        loop_step_theta=loop_step_theta,
        merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,
        paste_on_static=paste_on_static,
        view_get_scale_factor=view_get_scale_factor,
        view_set_scale_factor=view_set_scale_factor,
        downsample_factor_before_vae_decode=downsample_factor_before_vae_decode,
        denoise_to_step=denoise_to_step,
        num_windows_h_2=num_windows_h_2,
        num_windows_w_2=num_windows_w_2,
        dock_at_f=dock_at_f,
        overlap_ratio_list_1_f=overlap_ratio_list_1_f,
        overlap_ratio_list_2_f=overlap_ratio_list_2_f,
        loop_step_frame=loop_step_frame,
        total_f=total_f,
        upscale_factor=upscale_factor,
        merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,
        temporal_guidance_scale=getattr(vargs, "temporal_guidance_scale", None),
        save_latents=save_latents,
    )



if __name__ == "__main__":
    vargs = VArgs.from_args()
    print(vargs)

    if getattr(vargs, "launch_ui", False):
        demo = launch_upload_ui(default_out_dir=vargs.static_out_dir)
        demo.launch(server_name=vargs.ui_host, server_port=int(vargs.ui_port), share=bool(vargs.ui_share))
        raise SystemExit(0)

    if vargs.mode == "static":
        run_static_panorama(vargs)
        raise SystemExit(0)
    if vargs.mode == "static_wrap":
        run_static_wrap_panorama(vargs)
        raise SystemExit(0)

    # dynamic / legacy behavior
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{vargs.gpu_id}"
    os.environ["WORLD_SIZE"] = "1"

    # Auto-generate phi_prompt_dict at runtime (no code edits needed per image)
    if getattr(vargs, "auto_phi_prompts", False):
        vargs.phi_prompt_dict = build_phi_prompt_dict(
            image_path=vargs.pano_image_path,
            base_prompt=vargs.prompt,
            mode=getattr(vargs, "auto_phi_prompt_mode", "heuristic_bands"),
        )

    run_dynamic_video_generation(vargs)
