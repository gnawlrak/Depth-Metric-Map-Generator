"""
Independent point-cloud service for DepthViz.

- Runs on port 3002.
- Receives an image from the frontend (port 3000), reads EXIF for camera intrinsics,
  calls the inference backend (port 3001) for raw relative + metric depth,
  fuses them with global-scale Method A, and generates an interactive Three.js
  point cloud viewer.
"""

import io
import os
import base64
import logging
import time
import uuid
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ExifTags
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pointcloud-backend")

app = FastAPI(title="DepthViz Point Cloud Backend")

INFERENCE_URL = os.environ.get("INFERENCE_BACKEND_URL", "http://127.0.0.1:3001")
_POINT_CLOUD_STORE: Dict[str, Dict[str, Any]] = {}
_STORE_TTL_SECONDS = 3600
_MAX_STORE_SIZE = 20
_MAX_PC_WIDTH = 640  # downsample depth/RGB before point-cloud to limit point count

# Cache for model depth ranges fetched from the inference backend (single source
# of truth: inference_server.py MODEL_DEPTH_RANGES).  Populated lazily on first use.
_model_ranges_cache: Optional[Dict[str, Tuple[float, float]]] = None


def _get_model_ranges() -> Dict[str, Tuple[float, float]]:
    """Return cached model depth ranges, fetching from inference backend if needed."""
    global _model_ranges_cache
    if _model_ranges_cache is not None:
        return _model_ranges_cache
    try:
        r = requests.get(f"{INFERENCE_URL}/models", timeout=3.0)
        if r.ok:
            cache: Dict[str, Tuple[float, float]] = {}
            for m in r.json().get("models", []):
                dr = m.get("depth_range")
                if dr and len(dr) == 2:
                    cache[m["id"]] = (float(dr[0]), float(dr[1]))
            if cache:
                _model_ranges_cache = cache
                logger.info(f"Fetched {len(cache)} model depth ranges from inference backend")
                return cache
    except Exception:
        pass
    # Fallback when inference backend is unreachable
    _model_ranges_cache = {}
    return _model_ranges_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_old_entries() -> None:
    now = time.time()
    expired = [k for k, v in _POINT_CLOUD_STORE.items() if now - v.get("created_at", 0) > _STORE_TTL_SECONDS]
    for k in expired:
        del _POINT_CLOUD_STORE[k]
    if len(_POINT_CLOUD_STORE) > _MAX_STORE_SIZE:
        oldest = sorted(_POINT_CLOUD_STORE.items(), key=lambda item: item[1].get("created_at", 0))[: len(_POINT_CLOUD_STORE) - _MAX_STORE_SIZE]
        for k, _ in oldest:
            del _POINT_CLOUD_STORE[k]


def _depth_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    """Decode a base64-encoded float32 depth array from the inference backend."""
    raw_b64 = payload["raw_depth"]
    shape = tuple(payload["raw_shape"])
    dtype = np.dtype(payload["raw_dtype"])
    raw_bytes = base64.b64decode(raw_b64)
    arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
    return arr.astype(np.float32)


def get_focal_length_from_exif(pil_img: Image.Image) -> Optional[float]:
    """Extract focal length in pixels from EXIF if available."""
    try:
        exif = pil_img._getexif()
        if not exif:
            return None

        tags = {tag: idx for idx, tag in ExifTags.TAGS.items()}
        fl_id = tags.get("FocalLength")
        fl_35_id = tags.get("FocalLengthIn35mmFilm")

        focal_len_35mm = None
        if fl_35_id and fl_35_id in exif:
            val = exif[fl_35_id]
            focal_len_35mm = float(val) if isinstance(val, (int, float)) else float(val[0]) / float(val[1])

        if focal_len_35mm:
            w, h = pil_img.size
            diag_px = (w ** 2 + h ** 2) ** 0.5
            diag_mm = (36 ** 2 + 24 ** 2) ** 0.5
            return focal_len_35mm * diag_px / diag_mm

        if fl_id and fl_id in exif:
            val = exif[fl_id]
            return float(val) if isinstance(val, (int, float)) else float(val[0]) / float(val[1])

        return None
    except Exception:
        return None


def method_a_fusion(
    depth_relative: np.ndarray,
    depth_metric: np.ndarray,
    depth_range: Tuple[float, float] = (0.1, 80.0),
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Global scale alignment (Method A): align relative depth to metric depth.

    DA2 relative output is disparity (larger = nearer).  Metric depth is depth
    in metres (larger = farther).  We must flip the relative polarity before
    computing the global scale, otherwise the scale is meaningless.

    *depth_range* (min_m, max_m) is the metric model's empirical pretraining
    range, used to filter valid pixels for robust scale estimation.
    """
    d_min, d_max = depth_range

    # Resize metric to match relative if shapes differ
    if depth_metric.shape != depth_relative.shape:
        depth_metric = cv2.resize(
            depth_metric.astype(np.float32),
            (depth_relative.shape[1], depth_relative.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # --- Polarity detection ---
    # Sample valid pixels and compute Pearson correlation.  If negative, the
    # relative map is disparity (larger=nearer) and must be flipped.
    valid_base = (depth_metric > d_min) & (depth_metric < d_max) & np.isfinite(depth_metric) & np.isfinite(depth_relative)

    polarity_flipped = False
    if np.count_nonzero(valid_base) > 50:
        sample_size = min(5000, np.count_nonzero(valid_base))
        sample_idx = np.random.choice(np.nonzero(valid_base)[0], sample_size, replace=False)
        corr = float(np.corrcoef(depth_relative.ravel()[sample_idx], depth_metric.ravel()[sample_idx])[0, 1])
        if corr < 0:
            depth_relative = float(depth_relative.max()) - depth_relative
            polarity_flipped = True
            logger.info(f"Fusion: relative polarity flipped (corr={corr:.3f})")

    # Valid metric depth mask using model-specific range
    valid = (depth_metric > d_min) & (depth_metric < d_max) & np.isfinite(depth_metric) & np.isfinite(depth_relative)

    # Edge / high-gradient filter to avoid using object boundaries for scale
    if np.any(valid):
        grad_x = cv2.Sobel(depth_metric, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_metric, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_threshold = np.percentile(grad_mag[valid], 90) if np.count_nonzero(valid) > 10 else 1.0
        valid = valid & (grad_mag < grad_threshold)

    if not np.any(valid):
        logger.warning("No valid pixels for Method A fusion; using metric depth directly")
        return depth_metric.copy(), 1.0, {"valid_pixels": 0, "scale": 1.0, "polarity_flipped": polarity_flipped}

    rel_valid = depth_relative[valid]
    met_valid = depth_metric[valid]

    # Robust scale via median to reduce outlier influence
    ratios = met_valid / np.clip(rel_valid, a_min=1e-6, a_max=None)
    scale = float(np.median(ratios))

    if not np.isfinite(scale) or scale <= 0:
        scale = float(met_valid.mean() / rel_valid.mean())

    depth_fused = depth_relative * scale

    stats = {
        "valid_pixels": int(np.count_nonzero(valid)),
        "scale": scale,
        "relative_mean": float(rel_valid.mean()),
        "metric_mean": float(met_valid.mean()),
        "fused_mean": float(depth_fused[valid].mean()),
        "polarity_flipped": polarity_flipped,
    }
    return depth_fused, scale, stats


def generate_point_cloud(
    pil_img: Image.Image,
    depth_fused: np.ndarray,
    focal_length_px: Optional[float],
    depth_trunc: float = 80.0,
) -> Dict[str, Any]:
    """Build an Open3D point cloud and return a Three.js-friendly binary payload.

    Uses base64-encoded Float32 / Uint8 arrays instead of JSON lists to keep
    the transfer size manageable for large clouds.
    """
    import open3d as o3d

    rgb = np.array(pil_img.convert("RGB"))

    # --- Downsample to limit point count (avoids browser OOM) ---
    h, w = depth_fused.shape
    downscale_factor = 1.0
    if w > _MAX_PC_WIDTH:
        downscale_factor = _MAX_PC_WIDTH / w
        new_w = _MAX_PC_WIDTH
        new_h = max(1, int(h * downscale_factor))
        depth_fused = cv2.resize(depth_fused.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if focal_length_px is not None:
            focal_length_px = focal_length_px * downscale_factor
        logger.info(f"Downsampled point cloud from {w}x{h} to {new_w}x{new_h}")

    h, w = depth_fused.shape
    fx = fy = focal_length_px if focal_length_px is not None else max(w, h)
    cx = w / 2.0
    cy = h / 2.0

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth_fused.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)

    return {
        "width": w,
        "height": h,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "point_count": len(points),
        "positions_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "colors_b64": base64.b64encode((colors * 255).astype(np.uint8).tobytes()).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    inference_ok = False
    try:
        r = requests.get(f"{INFERENCE_URL}/health", timeout=2.0)
        inference_ok = r.ok
    except Exception:
        pass
    return {"status": "ok", "inference_backend": INFERENCE_URL, "inference_reachable": inference_ok}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    metric_model: str = Form("metric3d_vit_small"),
    relative_model: str = Form("depth_anything_v2_base"),
    focal_length: Optional[float] = Form(None),
) -> JSONResponse:
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as exc:
        logger.exception("Failed to read uploaded image")
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # 1. Call inference backend for raw depths
    files = {"image": ("image.jpg", contents, image.content_type or "image/jpeg")}
    try:
        r = requests.post(
            f"{INFERENCE_URL}/depth-raw",
            files=files,
            data={"relative_model": relative_model, "metric_model": metric_model},
            timeout=600,
        )
        if not r.ok:
            raise RuntimeError(f"Inference backend error ({r.status_code}): {r.text}")
        payload = r.json()
    except Exception as exc:
        logger.exception("Failed to fetch raw depth from inference backend")
        raise HTTPException(status_code=502, detail=f"Inference backend call failed: {exc}")

    # 2. Decode raw depth arrays
    try:
        depth_relative = _depth_from_payload(payload["relative"])
        depth_metric = _depth_from_payload(payload["metric"])
    except Exception as exc:
        logger.exception("Failed to decode raw depth arrays")
        raise HTTPException(status_code=500, detail=f"Raw depth decode failed: {exc}")

    # 3. Method A fusion with model-specific depth range
    depth_range = _get_model_ranges().get(metric_model, (0.1, 80.0))
    depth_fused, scale, fusion_stats = method_a_fusion(depth_relative, depth_metric, depth_range)

    # 4. Camera intrinsics: request override > inference response > local EXIF
    focal_length_px = focal_length
    if focal_length_px is None:
        focal_length_px = payload.get("focal_length_px")
    if focal_length_px is None:
        focal_length_px = get_focal_length_from_exif(pil_img)

    # 5. Generate point cloud
    try:
        pcd_json = generate_point_cloud(pil_img, depth_fused, focal_length_px, depth_range[1])
    except Exception as exc:
        logger.exception("Failed to generate point cloud")
        raise HTTPException(status_code=500, detail=f"Point cloud generation failed: {exc}")

    # 6. Store and return viewer URL
    cloud_id = str(uuid.uuid4())
    _clean_old_entries()
    _POINT_CLOUD_STORE[cloud_id] = {
        "created_at": time.time(),
        "metric_model": metric_model,
        "relative_model": relative_model,
        "scale": scale,
        "fusion_stats": fusion_stats,
        **pcd_json,
    }

    return JSONResponse(
        content={
            "cloud_id": cloud_id,
            "viewer_url": f"/view/{cloud_id}",
            "point_count": pcd_json["point_count"],
            "scale": scale,
            "fusion_stats": fusion_stats,
            "focal_length_px": focal_length_px,
        }
    )


@app.get("/pointcloud/{cloud_id}")
def get_pointcloud(cloud_id: str) -> JSONResponse:
    if cloud_id not in _POINT_CLOUD_STORE:
        raise HTTPException(status_code=404, detail="Point cloud not found")
    return JSONResponse(content=_POINT_CLOUD_STORE[cloud_id])


@app.get("/view/{cloud_id}")
def view_pointcloud(cloud_id: str) -> HTMLResponse:
    if cloud_id not in _POINT_CLOUD_STORE:
        raise HTTPException(status_code=404, detail="Point cloud not found")
    return HTMLResponse(content=VIEWER_HTML.replace("{{CLOUD_ID}}", cloud_id))


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(content="""
    <h1>DepthViz Point Cloud Service</h1>
    <p>POST an image to <code>/generate</code> to create an interactive point cloud.</p>
    <p>This service is independent from the depth/ranging UI on port 3000.</p>
    """)


# ---------------------------------------------------------------------------
# Three.js viewer HTML (served from /view/{cloud_id})
# ---------------------------------------------------------------------------

VIEWER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DepthViz Point Cloud</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #050505; color: #e2e8f0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; overflow: hidden; }
    #canvas { position: fixed; inset: 0; width: 100%; height: 100%; }
    #ui {
      position: fixed; top: 16px; left: 16px; z-index: 10;
      background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px);
      border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px;
      padding: 16px; min-width: 240px; pointer-events: none;
    }
    #ui h1 { font-size: 14px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 10px; color: #60a5fa; }
    #ui .row { display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0; }
    #ui .row span:first-child { color: #94a3b8; }
    #ui .row span:last-child { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 600; }
    #loading {
      position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
      background: #050505; z-index: 20; transition: opacity 0.3s;
    }
    #loading.hidden { opacity: 0; pointer-events: none; }
    .spinner { width: 40px; height: 40px; border: 3px solid rgba(96, 165, 250, 0.2); border-top-color: #60a5fa; border-radius: 50%; animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    #error {
      position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      background: rgba(220, 38, 38, 0.15); border: 1px solid rgba(220, 38, 38, 0.4);
      color: #fca5a5; padding: 20px 28px; border-radius: 12px; max-width: 480px;
      display: none; z-index: 30;
    }
  </style>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }
  }
  </script>
</head>
<body>
  <div id="loading"><div class="spinner"></div></div>
  <div id="error"></div>
  <div id="ui">
    <h1>DEPTHVIZ POINT CLOUD</h1>
    <div class="row"><span>Points</span><span id="point-count">--</span></div>
    <div class="row"><span>Scale</span><span id="scale">--</span></div>
    <div class="row"><span>Polarity Flipped</span><span id="polarity">--</span></div>
    <div class="row"><span>Metric Model</span><span id="metric-model">--</span></div>
    <div class="row"><span>Relative Model</span><span id="relative-model">--</span></div>
    <div class="row"><span>Controls</span><span>Left drag / scroll</span></div>
  </div>
  <canvas id="canvas"></canvas>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const cloudId = '{{CLOUD_ID}}';
    const loading = document.getElementById('loading');
    const errorEl = document.getElementById('error');

    function showError(msg) {
      loading.classList.add('hidden');
      errorEl.style.display = 'block';
      errorEl.textContent = msg;
    }

    async function loadCloud() {
      const res = await fetch(`/pointcloud/${cloudId}`);
      if (!res.ok) throw new Error(`Failed to load point cloud: ${res.status} ${await res.text()}`);
      return await res.json();
    }

    function decodeBase64ToArray(b64) {
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }

    function initScene(data) {
      const canvas = document.getElementById('canvas');
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050505);

      const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1000);
      camera.position.set(0, 0, 2);

      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.target.set(0, 0, 0);

      // Decode binary point cloud data
      const positions = new Float32Array(decodeBase64ToArray(data.positions_b64));
      const colorsRaw = new Uint8Array(decodeBase64ToArray(data.colors_b64));
      const count = positions.length / 3;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const colorsF32 = new Float32Array(colorsRaw.length);
      for (let i = 0; i < colorsRaw.length; i++) colorsF32[i] = colorsRaw[i] / 255;
      geometry.setAttribute('color', new THREE.BufferAttribute(colorsF32, 3));

      const material = new THREE.PointsMaterial({ size: 0.008, vertexColors: true, sizeAttenuation: true });
      const points = new THREE.Points(geometry, material);
      scene.add(points);

      // Center camera on cloud
      geometry.computeBoundingSphere();
      const center = geometry.boundingSphere.center;
      const radius = Math.max(geometry.boundingSphere.radius, 0.1);
      controls.target.copy(center);
      camera.position.copy(center.clone().add(new THREE.Vector3(0, 0, radius * 2.5)));

      // Optional floor grid
      const grid = new THREE.GridHelper(Math.max(radius * 4, 5), 50, 0x1e293b, 0x0f172a);
      grid.position.copy(center);
      grid.position.y -= radius * 0.5;
      scene.add(grid);

      // Stats
      document.getElementById('point-count').textContent = count.toLocaleString();
      document.getElementById('scale').textContent = data.scale?.toFixed(4) ?? '--';
      document.getElementById('polarity').textContent = data.fusion_stats?.polarity_flipped ? 'Yes' : 'No';
      document.getElementById('metric-model').textContent = data.metric_model ?? '--';
      document.getElementById('relative-model').textContent = data.relative_model ?? '--';

      window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }
      animate();
      loading.classList.add('hidden');
    }

    loadCloud()
      .then(initScene)
      .catch((err) => showError(err.message));
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("POINTCLOUD_BACKEND_PORT", "3002"))
    host = os.environ.get("POINTCLOUD_BACKEND_HOST", "0.0.0.0")
    logger.info(f"Starting point cloud backend on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
