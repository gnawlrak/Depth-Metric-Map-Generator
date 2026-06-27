"""
Local inference backend for Depth Anything V2, Metric3D, ZoeDepth and Depth Anything V2 Metric.
Runs as a FastAPI service that the Node.js server can proxy to.
"""

import io
import os
import sys
import base64
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags, ImageOps
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("depth-backend")

app = FastAPI(title="DepthViz Local Inference Backend")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Device preference: use CUDA if available, otherwise CPU
_DEVICE: Optional[str] = None


def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        try:
            import torch

            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _DEVICE = "cpu"
    return _DEVICE


# Model IDs
DA2_MODELS = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
}

METRIC3D_ONNX_MODELS = {
    "metric3d_vit_small": "onnx-community/metric3d-vit-small",
    "metric3d_vit_large": "onnx-community/metric3d-vit-large",
    "metric3d_vit_giant2": "onnx-community/metric3d-vit-giant2",
}

ZOEDEPTH_MODELS = {
    "zoedepth_n": "ZoeD_N",
    "zoedepth_k": "ZoeD_K",
    "zoedepth_nk": "ZoeD_NK",
}

# (model_id: (encoder, features, out_channels, max_depth, checkpoint_filename, hf_repo_id))
DA2_METRIC_MODELS = {
    # Indoor (Hypersim) — max_depth=20
    "da2_metric_indoor_small": ("vits", 64, [48, 96, 192, 384], 20, "depth_anything_v2_metric_hypersim_vits.pth", "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small"),
    "da2_metric_indoor_base": ("vitb", 128, [96, 192, 384, 768], 20, "depth_anything_v2_metric_hypersim_vitb.pth", "depth-anything/Depth-Anything-V2-Metric-Hypersim-Base"),
    "da2_metric_indoor_large": ("vitl", 256, [256, 512, 1024, 1024], 20, "depth_anything_v2_metric_hypersim_vitl.pth", "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"),
    # Outdoor (VKITTI) — max_depth=80
    "da2_metric_outdoor_small": ("vits", 64, [48, 96, 192, 384], 80, "depth_anything_v2_metric_vkitti_vits.pth", "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small"),
    "da2_metric_outdoor_base": ("vitb", 128, [96, 192, 384, 768], 80, "depth_anything_v2_metric_vkitti_vitb.pth", "depth-anything/Depth-Anything-V2-Metric-VKITTI-Base"),
    "da2_metric_outdoor_large": ("vitl", 256, [256, 512, 1024, 1024], 80, "depth_anything_v2_metric_vkitti_vitl.pth", "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large"),
}

ALL_MODELS = {**DA2_MODELS, **METRIC3D_ONNX_MODELS, **ZOEDEPTH_MODELS, **DA2_METRIC_MODELS}

# Empirical depth ranges (min_m, max_m) from each model's pretraining data.
# Used for outlier clipping and accurate metric_stats.
MODEL_DEPTH_RANGES: Dict[str, Tuple[float, float]] = {
    # Metric3D: trained on NYUv2 (indoor) + KITTI (outdoor)
    "metric3d_vit_small":   (0.3, 80.0),
    "metric3d_vit_large":   (0.3, 80.0),
    "metric3d_vit_giant2":  (0.3, 80.0),
    # ZoeDepth per-variant
    "zoedepth_n":  (0.3, 20.0),    # NYUv2 indoor
    "zoedepth_k":  (1.0, 80.0),    # KITTI outdoor
    "zoedepth_nk": (0.3, 80.0),    # NYU + KITTI mixed
    # DA2 Metric Indoor (Hypersim, max_depth=20)
    "da2_metric_indoor_small":  (0.1, 20.0),
    "da2_metric_indoor_base":   (0.1, 20.0),
    "da2_metric_indoor_large":  (0.1, 20.0),
    # DA2 Metric Outdoor (Virtual KITTI 2, max_depth=80)
    "da2_metric_outdoor_small": (0.1, 80.0),
    "da2_metric_outdoor_base":  (0.1, 80.0),
    "da2_metric_outdoor_large": (0.1, 80.0),
}

MODEL_DISPLAY_NAMES = {
    "depth_anything_v2_small": "Depth Anything V2 Small",
    "depth_anything_v2_base": "Depth Anything V2 Basic",
    "depth_anything_v2_large": "Depth Anything V2 Large",
    "metric3d_vit_small": "Metric3D ViT Small",
    "metric3d_vit_large": "Metric3D ViT Large",
    "metric3d_vit_giant2": "Metric3D ViT Giant2",
    "zoedepth_n": "ZoeDepth N (Indoor)",
    "zoedepth_k": "ZoeDepth K (Outdoor)",
    "zoedepth_nk": "ZoeDepth NK (Mixed)",
    "da2_metric_indoor_small": "DA2 Metric Indoor Small",
    "da2_metric_indoor_base": "DA2 Metric Indoor Basic",
    "da2_metric_indoor_large": "DA2 Metric Indoor Large",
    "da2_metric_outdoor_small": "DA2 Metric Outdoor Small",
    "da2_metric_outdoor_base": "DA2 Metric Outdoor Basic",
    "da2_metric_outdoor_large": "DA2 Metric Outdoor Large",
}

# ---------------------------------------------------------------------------
# Colormap (matches the Node.js turbo-like approximation)
# ---------------------------------------------------------------------------


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert a single-channel depth map (H, W) to an RGB image (H, W, 3)."""
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = (depth - depth_min) / (depth_max - depth_min)

    x = normalized.astype(np.float32)
    r = np.clip(255 * (3.11 * x - 2.11 * x * x), 0, 255).astype(np.uint8)
    g = np.clip(255 * (2.0 * x - x * x), 0, 255).astype(np.uint8)
    b = np.clip(255 * (1.5 * x), 0, 255).astype(np.uint8)

    colored = np.stack([r, g, b], axis=-1)
    return colored


def encode_image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_raw_depth(depth: np.ndarray) -> Dict[str, Any]:
    """Pack a float32 depth array into base64 bytes along with shape/dtype."""
    depth = depth.astype(np.float32)
    return {
        "raw_depth": base64.b64encode(depth.tobytes()).decode("utf-8"),
        "raw_shape": depth.shape,
        "raw_dtype": str(depth.dtype),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_pipeline_cache: Dict[str, Any] = {}
_onnx_session_cache: Dict[str, Any] = {}
_torch_model_cache: Dict[str, Any] = {}


def _disable_tqdm() -> None:
    """Disable tqdm progress bars; they crash on non-TTY stderr."""
    os.environ["DISABLE_TQDM"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from tqdm import tqdm
        tqdm.disable = True
    except Exception:
        pass


def load_depth_anything_v2(model_name: str) -> Any:
    """Lazy-load a Depth Anything V2 transformers pipeline."""
    if model_name in _pipeline_cache:
        return _pipeline_cache[model_name]

    from transformers import pipeline

    _disable_tqdm()

    hf_id = DA2_MODELS[model_name]
    logger.info(f"Loading Depth Anything V2 model: {hf_id}")
    device_idx = 0 if get_device() == "cuda" else -1
    pipe = pipeline(
        task="depth-estimation",
        model=hf_id,
        device=device_idx,
    )
    _pipeline_cache[model_name] = pipe
    logger.info(f"Loaded {hf_id} on {get_device()}")
    return pipe


def load_metric3d_onnx(model_name: str) -> Any:
    """Lazy-load a Metric3D ONNX model via huggingface_hub + onnxruntime."""
    if model_name in _onnx_session_cache:
        return _onnx_session_cache[model_name]

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    hf_id = METRIC3D_ONNX_MODELS[model_name]
    logger.info(f"Loading Metric3D ONNX model: {hf_id}")

    # The onnx-community repos typically contain onnx/model.onnx
    try:
        model_path = hf_hub_download(repo_id=hf_id, filename="onnx/model.onnx")
    except Exception:
        # Fallback: try root-level model.onnx
        try:
            model_path = hf_hub_download(repo_id=hf_id, filename="model.onnx")
        except Exception:
            # Last resort: repo name as the file at root
            model_path = hf_hub_download(repo_id=hf_id, filename=hf_id.split("/")[-1] + ".onnx")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    _onnx_session_cache[model_name] = session
    logger.info(f"Loaded {hf_id}")
    return session


def load_zoedepth(model_name: str) -> Any:
    """Lazy-load a ZoeDepth model via torch.hub."""
    if model_name in _torch_model_cache:
        return _torch_model_cache[model_name]

    import torch

    _disable_tqdm()
    hub_name = ZOEDEPTH_MODELS[model_name]
    logger.info(f"Loading ZoeDepth model: {hub_name}")

    # ZoeDepth internally depends on the MiDaS repo; pre-fetch it so the error
    # surface is easier to understand if the network is unavailable.
    try:
        torch.hub.help("isl-org/MiDaS", "DPT_BEiT_L_384", force_reload=False, trust_repo=True)
    except Exception as exc:
        logger.warning(f"Could not pre-fetch MiDaS dependency for ZoeDepth: {exc}")

    model = torch.hub.load("isl-org/ZoeDepth", hub_name, pretrained=True, trust_repo=True)
    model = model.to(get_device())
    model.eval()
    _torch_model_cache[model_name] = model
    logger.info(f"Loaded {hub_name} on {get_device()}")
    return model


_DA2_REPO_CLONED = False


def _ensure_da2_metric_repo() -> str:
    """Clone the official Depth-Anything-V2 repo (if needed) and add metric_depth to sys.path.

    The repo has no setup.py so it cannot be pip-installed. We clone it locally
    and insert metric_depth/ onto sys.path so that ``from depth_anything_v2.dpt
    import DepthAnythingV2`` resolves to the metric-capable version.
    Returns the metric_depth directory path.
    """
    global _DA2_REPO_CLONED
    repo_dir = os.path.join(os.path.dirname(__file__), "Depth-Anything-V2")
    metric_depth_dir = os.path.join(repo_dir, "metric_depth")
    dpt_path = os.path.join(metric_depth_dir, "depth_anything_v2", "dpt.py")

    if not os.path.exists(dpt_path):
        logger.info("Cloning Depth-Anything-V2 repository for metric depth support...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/DepthAnything/Depth-Anything-V2.git", repo_dir],
                check=True, capture_output=True, timeout=300,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to clone Depth-Anything-V2 repo: {exc}. "
                f"Please clone manually: git clone https://github.com/DepthAnything/Depth-Anything-V2.git {repo_dir}"
            )

    if metric_depth_dir not in sys.path:
        sys.path.insert(0, metric_depth_dir)

    _DA2_REPO_CLONED = True
    return metric_depth_dir


def load_da2_metric(model_name: str) -> Any:
    """Lazy-load a Depth Anything V2 Metric PyTorch checkpoint."""
    if model_name in _torch_model_cache:
        return _torch_model_cache[model_name]

    import torch
    from huggingface_hub import hf_hub_download

    encoder, features, out_channels, max_depth, checkpoint_name, hf_repo = DA2_METRIC_MODELS[model_name]
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    logger.info(f"Loading {display_name} (max_depth={max_depth})")

    # Ensure the metric_depth version of depth_anything_v2 is importable
    _ensure_da2_metric_repo()
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder=encoder, features=features, out_channels=out_channels, max_depth=max_depth,
    )

    # Resolve checkpoint: env var -> Hugging Face -> local checkpoints dir
    checkpoint_path = os.environ.get(f"DA2_METRIC_{model_name.upper()}_CHECKPOINT")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = hf_hub_download(repo_id=hf_repo, filename=checkpoint_name)
        except Exception as exc:
            logger.warning(f"HF download failed for {hf_repo}/{checkpoint_name}: {exc}")
            local_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            checkpoint_path = os.path.join(local_dir, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"DA2 Metric checkpoint not found: {checkpoint_path}. "
            f"Download {checkpoint_name} from {hf_repo} and place it in "
            f"python_backend/checkpoints/, or set DA2_METRIC_{model_name.upper()}_CHECKPOINT."
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(get_device())
    model.eval()
    _torch_model_cache[model_name] = model
    logger.info(f"Loaded {display_name} from {checkpoint_path} on {get_device()}")
    return model


# ---------------------------------------------------------------------------
# Metric3D preprocessing / postprocessing
# ---------------------------------------------------------------------------

METRIC3D_VIT_INPUT_SIZE = (616, 1064)
METRIC3D_CONV_INPUT_SIZE = (544, 1216)
METRIC3D_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
METRIC3D_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def metric3d_preprocess(
    pil_img: Image.Image, model_name: str
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Resize and pad image for Metric3D ONNX inference.
    Returns (preprocessed RGB array NHWC, pad_info, original_size).
    """
    if "vit" in model_name:
        input_size = METRIC3D_VIT_INPUT_SIZE
    else:
        input_size = METRIC3D_CONV_INPUT_SIZE

    # Convert to RGB numpy array
    rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    orig_h, orig_w = rgb.shape[:2]

    # Keep-ratio resize
    scale = min(input_size[0] / orig_h, input_size[1] / orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    resized = np.array(
        Image.fromarray(rgb.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)
    )

    # Pad to input_size with the mean values so that after internal normalization they become zero
    pad_h = input_size[0] - new_h
    pad_w = input_size[1] - new_w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    padded = np.pad(
        resized,
        ((pad_h_half, pad_h - pad_h_half), (pad_w_half, pad_w - pad_w_half), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # Fill padding regions with ImageNet mean
    padded[:pad_h_half, :, :] = METRIC3D_MEAN
    padded[-(pad_h - pad_h_half):, :, :] = METRIC3D_MEAN
    padded[:, :pad_w_half, :] = METRIC3D_MEAN
    padded[:, -(pad_w - pad_w_half):, :] = METRIC3D_MEAN

    return padded, (pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half), (orig_h, orig_w)


def metric3d_postprocess(
    pred: np.ndarray,
    pad_info: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
    focal_length: float = 1000.0,
    max_depth: float = 80.0,
) -> np.ndarray:
    """
    Unpad, resize to original size, and convert canonical depth to metric depth.
    pred: (1, 1, H, W) or (H, W) array.
    """
    if pred.ndim == 4:
        pred = pred.squeeze()
    elif pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)

    pad_h_half, pad_h_bottom, pad_w_half, pad_w_right = pad_info
    orig_h, orig_w = original_size

    # Unpad
    h, w = pred.shape[:2]
    pred = pred[pad_h_half : h - pad_h_bottom, pad_w_half : w - pad_w_right]

    # pred is HxW float32; use cv2 for reliable resize. Size must be (width, height).
    pred_resized = cv2.resize(pred.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Canonical to metric scale (canonical focal length = 1000.0)
    canonical_to_real_scale = focal_length / 1000.0
    metric_depth = pred_resized * canonical_to_real_scale
    metric_depth = np.clip(metric_depth, 0, max_depth)
    return metric_depth


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------


def _grayscale_and_stats(
    metric_depth: np.ndarray,
    clip_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Build uint8 grayscale visualization and metric stats from a metric depth map.

    If *clip_range* is provided, values outside [min, max] are clamped before
    computing stats.  This filters outliers from the model's known pretraining
    range, yielding tighter, more accurate metric_stats.
    """
    if clip_range is not None:
        metric_depth = np.clip(metric_depth, clip_range[0], clip_range[1])

    depth_min = float(metric_depth.min())
    depth_max = float(metric_depth.max())
    if depth_max - depth_min < 1e-6:
        vis_depth = np.zeros_like(metric_depth, dtype=np.uint8)
    else:
        vis_depth = ((metric_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    stats = {
        "min_m": depth_min,
        "max_m": depth_max,
        "mean_m": float(metric_depth.mean()),
    }
    return vis_depth, metric_depth, stats


def sample_depth_at_frac(depth: np.ndarray, x_frac: float, y_frac: float) -> float:
    """Bilinear sample of a float depth map at normalized coordinates."""
    h, w = depth.shape[:2]
    x = float(x_frac) * (w - 1)
    y = float(y_frac) * (h - 1)
    sample = cv2.getRectSubPix(depth.astype(np.float32), (1, 1), (x, y))
    return float(sample[0, 0])


def infer_depth_anything_v2(
    pil_img: Image.Image, model_name: str, return_raw: bool = False,
    target_xy: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    pipe = load_depth_anything_v2(model_name)
    result = pipe(pil_img)
    depth_pil = result["depth"]  # PIL Image, L mode, 0-255

    depth = np.array(depth_pil, dtype=np.uint8)
    colored = colorize_depth(depth.astype(np.float32))

    colored_pil = Image.fromarray(colored)
    grayscale_pil = Image.fromarray(depth, mode="L")

    output: Dict[str, Any] = {
        "colored": encode_image_to_base64(colored_pil),
        "grayscale": encode_image_to_base64(grayscale_pil),
        "model_used": MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title()),
        "engine": "python-transformers",
        "is_metric": False,
    }

    if target_xy is not None:
        x_frac, y_frac = target_xy
        x = int(np.clip(x_frac * (depth.shape[1] - 1), 0, depth.shape[1] - 1))
        y = int(np.clip(y_frac * (depth.shape[0] - 1), 0, depth.shape[0] - 1))
        output["target_depth_m"] = None
        output["target_raw_val"] = int(depth[y, x])

    if return_raw:
        predicted_depth = result.get("predicted_depth")
        if predicted_depth is not None:
            raw = predicted_depth.squeeze().cpu().numpy().astype(np.float32)
            output.update(encode_raw_depth(raw))
        else:
            # Fallback: use the uint8 depth normalized back to 0-1 relative depth.
            logger.warning("predicted_depth not available from pipeline; using uint8 depth as raw fallback")
            output.update(encode_raw_depth(depth.astype(np.float32) / 255.0))

    return output


def infer_metric3d(
    pil_img: Image.Image,
    model_name: str,
    focal_length: float = 1000.0,
    return_raw: bool = False,
    target_xy: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    session = load_metric3d_onnx(model_name)

    preprocessed, pad_info, original_size = metric3d_preprocess(pil_img, model_name)

    # The ONNX model includes normalization, so pass raw RGB (padded with mean) directly.
    # HWC -> CHW, add batch dim
    input_tensor = preprocessed.transpose(2, 0, 1)[None, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    pred_depth = outputs[0]  # expected (1, 1, H, W)

    clip_range = MODEL_DEPTH_RANGES.get(model_name)
    max_depth = clip_range[1] if clip_range else 80.0
    metric_depth = metric3d_postprocess(pred_depth, pad_info, original_size, focal_length, max_depth)

    vis_depth, _, stats = _grayscale_and_stats(metric_depth, clip_range)
    colored = colorize_depth(vis_depth.astype(np.float32))

    colored_pil = Image.fromarray(colored)
    grayscale_pil = Image.fromarray(vis_depth, mode="L")

    output: Dict[str, Any] = {
        "colored": encode_image_to_base64(colored_pil),
        "grayscale": encode_image_to_base64(grayscale_pil),
        "model_used": MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title()),
        "engine": "python-onnx",
        "is_metric": True,
        "metric_stats": stats,
    }

    if target_xy is not None:
        output["target_depth_m"] = sample_depth_at_frac(metric_depth, target_xy[0], target_xy[1])
        output["target_raw_val"] = None

    if return_raw:
        output.update(encode_raw_depth(metric_depth))

    return output


def infer_zoedepth(pil_img: Image.Image, model_name: str, return_raw: bool = False, target_xy: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    model = load_zoedepth(model_name)
    depth = model.infer_pil(pil_img)  # numpy float32, meters

    clip_range = MODEL_DEPTH_RANGES.get(model_name)
    vis_depth, _, stats = _grayscale_and_stats(depth, clip_range)
    colored = colorize_depth(vis_depth.astype(np.float32))

    colored_pil = Image.fromarray(colored)
    grayscale_pil = Image.fromarray(vis_depth, mode="L")

    output: Dict[str, Any] = {
        "colored": encode_image_to_base64(colored_pil),
        "grayscale": encode_image_to_base64(grayscale_pil),
        "model_used": MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title()),
        "engine": "python-zoedepth",
        "is_metric": True,
        "metric_stats": stats,
    }

    if target_xy is not None:
        output["target_depth_m"] = sample_depth_at_frac(depth, target_xy[0], target_xy[1])
        output["target_raw_val"] = None

    if return_raw:
        output.update(encode_raw_depth(depth))

    return output


def infer_da2_metric(pil_img: Image.Image, model_name: str, return_raw: bool = False, target_xy: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    model = load_da2_metric(model_name)

    # The DepthAnythingV2 model expects a BGR image by default in the original repo.
    # Passing RGB also works because the network is trained on ImageNet-normalized data;
    # we keep the same channel order the pretrained weights expect (BGR).
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    depth = model.infer_image(bgr)  # numpy float32, meters

    clip_range = MODEL_DEPTH_RANGES.get(model_name)
    vis_depth, _, stats = _grayscale_and_stats(depth, clip_range)
    colored = colorize_depth(vis_depth.astype(np.float32))

    colored_pil = Image.fromarray(colored)
    grayscale_pil = Image.fromarray(vis_depth, mode="L")

    output: Dict[str, Any] = {
        "colored": encode_image_to_base64(colored_pil),
        "grayscale": encode_image_to_base64(grayscale_pil),
        "model_used": MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title()),
        "engine": "python-da2-metric",
        "is_metric": True,
        "metric_stats": stats,
    }

    if target_xy is not None:
        output["target_depth_m"] = sample_depth_at_frac(depth, target_xy[0], target_xy[1])
        output["target_raw_val"] = None

    if return_raw:
        output.update(encode_raw_depth(depth))

    return output


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------


def get_focal_length_from_exif(pil_img: Image.Image) -> Optional[float]:
    """Extract focal length in pixels from EXIF if available."""
    try:
        exif = pil_img._getexif()
        if not exif:
            return None

        # Map tag names to IDs
        tags = {tag: idx for idx, tag in ExifTags.TAGS.items()}
        fl_id = tags.get("FocalLength")
        fl_35_id = tags.get("FocalLengthIn35mmFilm")

        focal_len_35mm = None
        if fl_35_id and fl_35_id in exif:
            val = exif[fl_35_id]
            focal_len_35mm = float(val) if isinstance(val, (int, float)) else float(val[0]) / float(val[1])

        focal_len = None
        if fl_id and fl_id in exif:
            val = exif[fl_id]
            focal_len = float(val) if isinstance(val, (int, float)) else float(val[0]) / float(val[1])

        # Prefer 35mm equivalent for pixel focal length estimation
        if focal_len_35mm:
            # sensor width approx 36mm, height 24mm; use image diagonal for a simple focal length in px
            w, h = pil_img.size
            diag_px = (w ** 2 + h ** 2) ** 0.5
            diag_mm = (36 ** 2 + 24 ** 2) ** 0.5
            return focal_len_35mm * diag_px / diag_mm

        return focal_len
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": get_device(),
        "available_models": list(ALL_MODELS.keys()),
    }


@app.get("/models")
def list_models() -> Dict[str, Any]:
    def model_type(model_id: str) -> str:
        if model_id in DA2_MODELS:
            return "depth_anything_v2"
        if model_id in METRIC3D_ONNX_MODELS:
            return "metric3d"
        if model_id in ZOEDEPTH_MODELS:
            return "zoedepth"
        if model_id in DA2_METRIC_MODELS:
            return "da2_metric"
        return "unknown"

    return {
        "models": [
            {
                "id": k,
                "hf_id": v if isinstance(v, str) else None,
                "display_name": MODEL_DISPLAY_NAMES.get(k, k.replace("_", " ").title()),
                "type": model_type(k),
                "is_metric": k not in DA2_MODELS,
                "depth_range": MODEL_DEPTH_RANGES.get(k),
            }
            for k, v in ALL_MODELS.items()
        ]
    }


@app.post("/depth")
async def depth(
    image: UploadFile = File(...),
    model: str = Form("depth_anything_v2_small"),
    focal_length: Optional[float] = Form(None),
    target_x_frac: Optional[float] = Form(None),
    target_y_frac: Optional[float] = Form(None),
) -> JSONResponse:
    if model not in ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(ALL_MODELS.keys())}",
        )

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as exc:
        logger.exception("Failed to read uploaded image")
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    target_xy: Optional[Tuple[float, float]] = None
    if target_x_frac is not None and target_y_frac is not None:
        target_xy = (float(target_x_frac), float(target_y_frac))

    try:
        if model in DA2_MODELS:
            result = infer_depth_anything_v2(pil_img, model, target_xy=target_xy)
        elif model in METRIC3D_ONNX_MODELS:
            fl = focal_length if focal_length is not None else 1000.0
            result = infer_metric3d(pil_img, model, fl, target_xy=target_xy)
        elif model in ZOEDEPTH_MODELS:
            result = infer_zoedepth(pil_img, model, target_xy=target_xy)
        elif model in DA2_METRIC_MODELS:
            result = infer_da2_metric(pil_img, model, target_xy=target_xy)
        else:
            raise HTTPException(status_code=400, detail=f"Model dispatcher missing for '{model}'")
    except Exception as exc:
        logger.exception(f"Inference failed for model {model}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return JSONResponse(content=result)


@app.post("/depth-raw")
async def depth_raw(
    image: UploadFile = File(...),
    relative_model: str = Form("depth_anything_v2_base"),
    metric_model: str = Form("metric3d_vit_small"),
) -> JSONResponse:
    """
    Return raw float32 depth arrays for both a relative model and a metric model.
    Used by the independent point-cloud service (port 3002) for Method A fusion.
    """
    if relative_model not in DA2_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"relative_model must be one of {list(DA2_MODELS.keys())}",
        )
    if metric_model not in ALL_MODELS or metric_model in DA2_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"metric_model must be a metric model. Available: {[m for m in ALL_MODELS if m not in DA2_MODELS]}",
        )

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as exc:
        logger.exception("Failed to read uploaded image")
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        rel_result = infer_depth_anything_v2(pil_img, relative_model, return_raw=True)
    except Exception as exc:
        logger.exception(f"Relative depth inference failed for {relative_model}")
        raise HTTPException(status_code=500, detail=f"Relative depth inference failed: {exc}")

    try:
        if metric_model in METRIC3D_ONNX_MODELS:
            met_result = infer_metric3d(pil_img, metric_model, return_raw=True)
        elif metric_model in ZOEDEPTH_MODELS:
            met_result = infer_zoedepth(pil_img, metric_model, return_raw=True)
        elif metric_model in DA2_METRIC_MODELS:
            met_result = infer_da2_metric(pil_img, metric_model, return_raw=True)
        else:
            raise HTTPException(status_code=400, detail=f"Metric model dispatcher missing for '{metric_model}'")
    except Exception as exc:
        logger.exception(f"Metric depth inference failed for {metric_model}")
        raise HTTPException(status_code=500, detail=f"Metric depth inference failed: {exc}")

    focal_length = get_focal_length_from_exif(pil_img)

    return JSONResponse(
        content={
            "image_size": pil_img.size,
            "focal_length_px": focal_length,
            "relative": {
                "model": relative_model,
                "display_name": MODEL_DISPLAY_NAMES.get(relative_model, relative_model),
                **{k: v for k, v in rel_result.items() if k in ("raw_depth", "raw_shape", "raw_dtype")},
            },
            "metric": {
                "model": metric_model,
                "display_name": MODEL_DISPLAY_NAMES.get(metric_model, metric_model),
                **{k: v for k, v in met_result.items() if k in ("raw_depth", "raw_shape", "raw_dtype", "metric_stats")},
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PYTHON_BACKEND_PORT", "3001"))
    host = os.environ.get("PYTHON_BACKEND_HOST", "127.0.0.1")
    logger.info(f"Starting local inference backend on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
