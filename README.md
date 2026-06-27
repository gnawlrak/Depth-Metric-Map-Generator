# DepthViz Pro — Local Depth Anything V2 + Metric3D Backend

This app estimates depth maps locally. It supports:

- **Depth Anything V2** (Small / Basic / Large)
- **Metric3D** (ViT Small / Large) — true metric depth in meters

A Python FastAPI backend runs the full PyTorch/ONNX models. The Node.js server automatically falls back to a JS/ONNX pipeline when the Python backend is not running.

## Run Locally

**Prerequisites:** Node.js, Python 3.10+

1. Install Node dependencies:
   ```bash
   npm install
   ```

2. (Optional but recommended) Install Python backend dependencies:
   ```bash
   pip install -r python_backend/requirements.txt
   ```

3. Copy `.env.example` to `.env.local` and set any required values.

4. Start the Python backend (in one terminal):
   ```bash
   npm run dev:python
   ```

5. Start the Node.js server (in another terminal):
   ```bash
   npm run dev
   ```

The app will be available at `http://localhost:3000`. The Python backend runs on `http://localhost:3001` by default (override with `PYTHON_BACKEND_URL`).

## Model Selection

Use the **Depth Model** dropdown in the sidebar to switch between:

- `Depth Anything V2 Small`
- `Depth Anything V2 Basic`
- `Depth Anything V2 Large`
- `Metric3D ViT Small`
- `Metric3D ViT Large`

Metric3D models require the Python backend. If it is not running, the server falls back to Depth Anything V2 Small via JS/ONNX and shows a notice.

## Notes

- First inference for a model downloads weights from Hugging Face and caches them locally.
- Metric3D depth maps are returned in metric meters; the UI also shows min/max/mean statistics.
- For GPU acceleration, ensure your Python environment has a CUDA-enabled `torch` and `onnxruntime-gpu`.
