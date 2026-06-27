import 'dotenv/config';
import express from 'express';
import path from 'path';
import fs from 'fs';
import https from 'https';
import multer from 'multer';
import { createServer as createViteServer } from 'vite';
import { pipeline, RawImage } from '@xenova/transformers';
import { PNG } from 'pngjs';

const upload = multer({ storage: multer.memoryStorage() });

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

interface ModelInfo {
  id: string;
  displayName: string;
  family: 'depth_anything_v2' | 'metric3d';
  pythonModel: string;
  jsFallbackModel?: string;
  depthRange?: { min: number; max: number; label: string };
}

const MODELS: ModelInfo[] = [
  {
    id: 'depth_anything_v2_small',
    displayName: 'Depth Anything V2 Small',
    family: 'depth_anything_v2',
    pythonModel: 'depth_anything_v2_small',
    jsFallbackModel: 'Xenova/depth-anything-v2-small',
  },
  {
    id: 'depth_anything_v2_base',
    displayName: 'Depth Anything V2 Basic',
    family: 'depth_anything_v2',
    pythonModel: 'depth_anything_v2_base',
    jsFallbackModel: 'Xenova/depth-anything-v2-base',
  },
  {
    id: 'depth_anything_v2_large',
    displayName: 'Depth Anything V2 Large',
    family: 'depth_anything_v2',
    pythonModel: 'depth_anything_v2_large',
    jsFallbackModel: 'Xenova/depth-anything-v2-large',
  },
  {
    id: 'metric3d_vit_small',
    displayName: 'Metric3D ViT Small',
    family: 'metric3d',
    pythonModel: 'metric3d_vit_small',
    depthRange: { min: 0.3, max: 80, label: 'NYU+KITTI 0.3-80m' },
  },
  {
    id: 'metric3d_vit_large',
    displayName: 'Metric3D ViT Large',
    family: 'metric3d',
    pythonModel: 'metric3d_vit_large',
    depthRange: { min: 0.3, max: 80, label: 'NYU+KITTI 0.3-80m' },
  },
  {
    id: 'metric3d_vit_giant2',
    displayName: 'Metric3D ViT Giant2',
    family: 'metric3d',
    pythonModel: 'metric3d_vit_giant2',
    depthRange: { min: 0.3, max: 80, label: 'NYU+KITTI 0.3-80m' },
  },
  {
    id: 'zoedepth_n',
    displayName: 'ZoeDepth N (Indoor)',
    family: 'metric3d',
    pythonModel: 'zoedepth_n',
    depthRange: { min: 0.3, max: 20, label: 'NYU Indoor 0.3-20m' },
  },
  {
    id: 'zoedepth_k',
    displayName: 'ZoeDepth K (Outdoor)',
    family: 'metric3d',
    pythonModel: 'zoedepth_k',
    depthRange: { min: 1.0, max: 80, label: 'KITTI Outdoor 1-80m' },
  },
  {
    id: 'zoedepth_nk',
    displayName: 'ZoeDepth NK (Mixed)',
    family: 'metric3d',
    pythonModel: 'zoedepth_nk',
    depthRange: { min: 0.3, max: 80, label: 'NYU+KITTI 0.3-80m' },
  },
  {
    id: 'da2_metric_indoor_small',
    displayName: 'DA2 Metric Indoor Small',
    family: 'metric3d',
    pythonModel: 'da2_metric_indoor_small',
    depthRange: { min: 0.1, max: 20, label: 'Hypersim Indoor 0.1-20m' },
  },
  {
    id: 'da2_metric_indoor_base',
    displayName: 'DA2 Metric Indoor Basic',
    family: 'metric3d',
    pythonModel: 'da2_metric_indoor_base',
    depthRange: { min: 0.1, max: 20, label: 'Hypersim Indoor 0.1-20m' },
  },
  {
    id: 'da2_metric_indoor_large',
    displayName: 'DA2 Metric Indoor Large',
    family: 'metric3d',
    pythonModel: 'da2_metric_indoor_large',
    depthRange: { min: 0.1, max: 20, label: 'Hypersim Indoor 0.1-20m' },
  },
  {
    id: 'da2_metric_outdoor_small',
    displayName: 'DA2 Metric Outdoor Small',
    family: 'metric3d',
    pythonModel: 'da2_metric_outdoor_small',
    depthRange: { min: 0.1, max: 80, label: 'VKITTI Outdoor 0.1-80m' },
  },
  {
    id: 'da2_metric_outdoor_base',
    displayName: 'DA2 Metric Outdoor Basic',
    family: 'metric3d',
    pythonModel: 'da2_metric_outdoor_base',
    depthRange: { min: 0.1, max: 80, label: 'VKITTI Outdoor 0.1-80m' },
  },
  {
    id: 'da2_metric_outdoor_large',
    displayName: 'DA2 Metric Outdoor Large',
    family: 'metric3d',
    pythonModel: 'da2_metric_outdoor_large',
    depthRange: { min: 0.1, max: 80, label: 'VKITTI Outdoor 0.1-80m' },
  },
];

const DEFAULT_MODEL = MODELS[0];

function resolveModel(input?: string): ModelInfo {
  if (!input) return DEFAULT_MODEL;
  const found = MODELS.find((m) => m.id === input || m.displayName === input);
  return found || DEFAULT_MODEL;
}

  // ---------------------------------------------------------------------------
  // Python backend client
  // ---------------------------------------------------------------------------

  const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:3001';
  const POINTCLOUD_BACKEND_URL = process.env.POINTCLOUD_BACKEND_URL || 'http://127.0.0.1:3002';
  const PUBLIC_POINTCLOUD_URL = process.env.PUBLIC_POINTCLOUD_URL;
  let pythonBackendAvailable: boolean | null = null;
let pythonBackendLastCheck = 0;
const PYTHON_BACKEND_CACHE_MS = 10000;

// Cache for Python-sourced model metadata (depth ranges). Single source of truth.
let _pythonModelsCache: Array<{ id: string; depth_range?: [number, number] }> | null = null;

async function fetchPythonModelsMeta(): Promise<void> {
  if (_pythonModelsCache !== null) return;
  try {
    const res = await fetch(`${PYTHON_BACKEND_URL}/models`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json() as any;
      _pythonModelsCache = data.models;
      console.log('[server] Model metadata (depth ranges) fetched from Python backend');
    }
  } catch {
    // Python backend not available; MODELS array provides fallback values
  }
}

async function isPythonBackendHealthy(): Promise<boolean> {
  const now = Date.now();
  if (pythonBackendAvailable !== null && now - pythonBackendLastCheck < PYTHON_BACKEND_CACHE_MS) {
    return pythonBackendAvailable;
  }
  try {
    const res = await fetch(`${PYTHON_BACKEND_URL}/health`, { signal: AbortSignal.timeout(2000) });
    pythonBackendAvailable = res.ok;
  } catch {
    pythonBackendAvailable = false;
  }
  pythonBackendLastCheck = now;
  return pythonBackendAvailable;
}

async function callPythonBackend(
  fileBuffer: Buffer,
  mimetype: string,
  model: ModelInfo,
  focalLength?: number,
  targetXFrac?: number,
  targetYFrac?: number
): Promise<{ colored: string; grayscale: string; modelUsed: string; warning?: string; metricStats?: any; targetDepthM?: number | null; targetRawVal?: number | null }> {
  const form = new FormData();
  form.append('image', new Blob([fileBuffer], { type: mimetype }), 'image');
  form.append('model', model.pythonModel);
  if (focalLength !== undefined) {
    form.append('focal_length', String(focalLength));
  }
  if (targetXFrac !== undefined && targetYFrac !== undefined) {
    form.append('target_x_frac', String(targetXFrac));
    form.append('target_y_frac', String(targetYFrac));
  }

  const res = await fetch(`${PYTHON_BACKEND_URL}/depth`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(600000), // 10 min for large model downloads
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Python backend error (${res.status}): ${text}`);
  }

  const data = (await res.json()) as any;
  return {
    colored: `data:image/png;base64,${data.colored}`,
    grayscale: `data:image/png;base64,${data.grayscale}`,
    modelUsed: `${data.model_used} (Python Local)`,
    warning: data.warning,
    metricStats: data.metric_stats,
    targetDepthM: data.target_depth_m,
    targetRawVal: data.target_raw_val,
  };
}

// ---------------------------------------------------------------------------
// JS/ONNX fallback pipeline
// ---------------------------------------------------------------------------

const pipelineCache: Record<string, any> = {};

const FALLBACK_CHAIN: Record<string, string[]> = {
  'Xenova/depth-anything-v2-large': ['Xenova/depth-anything-v2-base', 'Xenova/depth-anything-v2-small', 'Xenova/depth-anything-small-hf'],
  'Xenova/depth-anything-v2-base': ['Xenova/depth-anything-v2-small', 'Xenova/depth-anything-small-hf'],
  'Xenova/depth-anything-v2-small': ['Xenova/depth-anything-small-hf'],
  'Xenova/depth-anything-small-hf': [],
};

async function getJsPipeline(requestedModel: string): Promise<{ estimator: any; actualModel: string }> {
  const chain = FALLBACK_CHAIN[requestedModel] || [requestedModel];
  const modelsToTry = [requestedModel, ...chain];
  let lastErr: any;

  for (const modelName of modelsToTry) {
    try {
      if (!pipelineCache[modelName]) {
        pipelineCache[modelName] = await pipeline('depth-estimation', modelName);
      }
      return { estimator: pipelineCache[modelName], actualModel: modelName };
    } catch (err: any) {
      lastErr = err;
      console.warn(`[server] Failed to load ${modelName}: ${err.message}.`);
    }
  }

  throw new Error(`All JS/ONNX fallbacks failed for ${requestedModel}. Last error: ${lastErr?.message || 'unknown'}`);
}

function colorizeDepthMap(data: Uint8Array): Uint8Array {
  const coloredData = new Uint8Array(data.length * 3);
  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    const x = val / 255.0;
    const r = Math.max(0, Math.min(255, Math.floor(255 * (3.11 * x - 2.11 * x * x))));
    const g = Math.max(0, Math.min(255, Math.floor(255 * (2.0 * x - x * x))));
    const b = Math.max(0, Math.min(255, Math.floor(255 * (1.5 * x))));
    coloredData[i * 3] = r;
    coloredData[i * 3 + 1] = g;
    coloredData[i * 3 + 2] = b;
  }
  return coloredData;
}

function encodePngGrayscale(data: Uint8Array, width: number, height: number): Buffer {
  const rgba = Buffer.alloc(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const v = data[i];
    rgba[i * 4] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }
  const png = new PNG({ width, height });
  png.data = rgba;
  return PNG.sync.write(png);
}

function encodePngRgb(data: Uint8Array, width: number, height: number): Buffer {
  const rgba = Buffer.alloc(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4] = data[i * 3];
    rgba[i * 4 + 1] = data[i * 3 + 1];
    rgba[i * 4 + 2] = data[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  const png = new PNG({ width, height });
  png.data = rgba;
  return PNG.sync.write(png);
}

async function runJsFallback(
  fileBuffer: Buffer,
  mimetype: string,
  model: ModelInfo
): Promise<{ colored: string; grayscale: string; modelUsed: string; warning?: string; metricStats?: any; targetDepthM?: number | null; targetRawVal?: number | null }> {
  const requestedModel = model.jsFallbackModel || 'Xenova/depth-anything-small-hf';
  const { estimator, actualModel } = await getJsPipeline(requestedModel);
  const img = await RawImage.fromBlob(new Blob([fileBuffer], { type: mimetype }));
  const result = await estimator(img);
  const { data, width, height } = result.depth;

  const grayscaleBuffer = encodePngGrayscale(data, width, height);
  const coloredData = colorizeDepthMap(data);
  const coloredBuffer = encodePngRgb(coloredData, width, height);

  const downgraded = actualModel !== requestedModel;
  let warning: string | undefined;
  if (model.family === 'metric3d') {
    warning = `Metric3D requires the Python backend (not running). Used ${actualModel} via JS/ONNX fallback instead.`;
  } else if (downgraded) {
    warning = `Python backend unavailable and ${model.displayName} could not be loaded. Downgraded to ${actualModel} via JS/ONNX fallback.`;
  }

  return {
    colored: `data:image/png;base64,${coloredBuffer.toString('base64')}`,
    grayscale: `data:image/png;base64,${grayscaleBuffer.toString('base64')}`,
    modelUsed: `${actualModel} (JS/ONNX${downgraded ? ' Fallback' : ''})`,
    warning,
    targetDepthM: null,
    targetRawVal: null,
  };
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Pre-check Python backend availability (non-blocking)
  isPythonBackendHealthy().then((healthy) => {
    if (healthy) {
      console.log(`[server] Python backend reachable at ${PYTHON_BACKEND_URL}`);
    } else {
      console.log(`[server] Python backend not reachable at ${PYTHON_BACKEND_URL}; will use JS/ONNX fallback.`);
    }
  });

  app.get('/api/models', async (_req, res) => {
    // Fetch depth ranges from Python backend (single source of truth)
    await fetchPythonModelsMeta();
    const models = MODELS.map((m) => {
      const pyModel = _pythonModelsCache?.find((p: any) => p.id === m.id);
      let depthRange = m.depthRange;
      if (pyModel?.depth_range) {
        depthRange = {
          min: pyModel.depth_range[0],
          max: pyModel.depth_range[1],
          label: m.depthRange?.label || `${pyModel.depth_range[0]}-${pyModel.depth_range[1]}m`,
        };
      }
      return { id: m.id, displayName: m.displayName, family: m.family, depthRange };
    });
    res.json({
      models,
      pythonBackendUrl: PYTHON_BACKEND_URL,
      pointcloudBackendUrl: POINTCLOUD_BACKEND_URL,
    });
  });

  app.post('/api/generate-depth', upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
      }

      const model = resolveModel(req.body.model);
      const fileBuffer = req.file.buffer;
      const mimetype = req.file.mimetype;
      const focalLength = req.body.focal_length ? parseFloat(req.body.focal_length) : undefined;
      const targetXFrac = req.body.target_x_frac ? parseFloat(req.body.target_x_frac) : undefined;
      const targetYFrac = req.body.target_y_frac ? parseFloat(req.body.target_y_frac) : undefined;

      let result;
      if (await isPythonBackendHealthy()) {
        try {
          result = await callPythonBackend(fileBuffer, mimetype, model, focalLength, targetXFrac, targetYFrac);
        } catch (pyErr: any) {
          console.warn('[server] Python backend call failed, falling back to JS/ONNX:', pyErr.message);
          result = await runJsFallback(fileBuffer, mimetype, model);
        }
      } else {
        result = await runJsFallback(fileBuffer, mimetype, model);
      }

      return res.json(result);
    } catch (error: any) {
      console.error('Error generating depth map:', error);
      let errorMsg = 'An error occurred while generating the depth map.';
      if (error && typeof error === 'object') {
        if (error.message) errorMsg = error.message;
        else {
          try {
            errorMsg = JSON.stringify(error);
          } catch (e) {
            errorMsg = error.toString();
          }
        }
      } else if (typeof error === 'string') {
        errorMsg = error;
      }
      res.status(500).json({ error: errorMsg });
    }
  });

  // ---------------------------------------------------------------------------
  // Point cloud backend proxy helpers
  // ---------------------------------------------------------------------------

  function getPublicBaseUrl(req: express.Request): string {
    if (PUBLIC_POINTCLOUD_URL) {
      return PUBLIC_POINTCLOUD_URL.replace(/\/$/, '');
    }
    // Use the same host the browser used to reach Node, so LAN/mobile access works
    const protocol = req.protocol === 'https' || req.get('x-forwarded-proto') === 'https' ? 'https' : 'http';
    const host = req.get('host');
    return `${protocol}://${host}`;
  }

  async function waitForPointCloudBackend(maxAttempts = 12, intervalMs = 1000): Promise<void> {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const res = await fetch(`${POINTCLOUD_BACKEND_URL}/health`, { signal: AbortSignal.timeout(1500) });
        if (res.ok) return;
      } catch {
        // ignore and retry
      }
      if (attempt < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }
    }
    throw new Error(`Point cloud backend at ${POINTCLOUD_BACKEND_URL} did not become ready`);
  }

  app.get('/api/pointcloud-view/:cloud_id', async (req, res) => {
    try {
      const response = await fetch(`${POINTCLOUD_BACKEND_URL}/view/${req.params.cloud_id}`);
      if (!response.ok) throw new Error(await response.text());
      const html = await response.text();
      // Rewrite the point-cloud data fetch so the browser hits the Node proxy too
      const rewritten = html.replace(
        /fetch\(`\/pointcloud\/\$\{cloudId\}`\)/g,
        'fetch(`/api/pointcloud-data/${cloudId}`)'
      );
      res.setHeader('Content-Type', 'text/html');
      res.send(rewritten);
    } catch (error: any) {
      console.error('Error proxying point cloud view:', error);
      res.status(502).json({ error: error.message || 'Point cloud view proxy failed' });
    }
  });

  app.get('/api/pointcloud-data/:cloud_id', async (req, res) => {
    try {
      const response = await fetch(`${POINTCLOUD_BACKEND_URL}/pointcloud/${req.params.cloud_id}`);
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      res.json(data);
    } catch (error: any) {
      console.error('Error proxying point cloud data:', error);
      res.status(502).json({ error: error.message || 'Point cloud data proxy failed' });
    }
  });

  app.post('/api/generate-pointcloud', upload.single('image'), async (req, res) => {
    try {
      await waitForPointCloudBackend();
      if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
      }

      const form = new FormData();
      form.append('image', new Blob([req.file.buffer], { type: req.file.mimetype }), 'image');
      form.append('metric_model', req.body.metric_model || 'metric3d_vit_small');
      form.append('relative_model', req.body.relative_model || 'depth_anything_v2_base');
      if (req.body.focal_length) {
        form.append('focal_length', req.body.focal_length);
      }

      const response = await fetch(`${POINTCLOUD_BACKEND_URL}/generate`, {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(600000),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Point cloud backend error (${response.status}): ${text}`);
      }

      const data = (await response.json()) as any;
      const publicBase = getPublicBaseUrl(req);
      // Rewrite backend-relative /view/{id} to Node-proxied /api/pointcloud-view/{id}
      const viewerPath = data.viewer_url.replace('/view', '/api/pointcloud-view');
      return res.json({ ...data, viewer_url: `${publicBase}${viewerPath}` });
    } catch (error: any) {
      console.error('Error generating point cloud:', error);
      const isDown = error?.cause?.code === 'ECONNREFUSED' || error?.message?.includes('fetch failed');
      if (isDown) {
        return res.status(503).json({
          error: '点云服务 (端口 3002) 未运行。请使用 npm run dev:all 或 npm run dev:all:https 启动全部服务。',
        });
      }
      let errorMsg = error?.message || 'An error occurred while generating the point cloud.';
      res.status(500).json({ error: errorMsg });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  // HTTPS support (for camera access from LAN devices)
  const useHttps = process.env.USE_HTTPS === '1' || process.argv.includes('--https');

  if (useHttps) {
    const certPath = path.join(process.cwd(), '.cert', 'cert.pem');
    const keyPath = path.join(process.cwd(), '.cert', 'key.pem');
    if (!fs.existsSync(certPath) || !fs.existsSync(keyPath)) {
      console.error('[server] HTTPS requested but certificates not found.');
      console.error('[server] Generate them first:  npm run generate-cert');
      process.exit(1);
    }
    const httpsServer = https.createServer(
      { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) },
      app
    );
    httpsServer.listen(PORT, '0.0.0.0', () => {
      console.log(`\n  🔒 HTTPS Server running on https://localhost:${PORT}`);
      console.log('  ⚠  Self-signed cert — browser will show a security warning.');
      console.log('     Click "Advanced" → "Proceed" to continue.\n');
    });
  } else {
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  }
}

startServer();
