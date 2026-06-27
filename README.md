# Depth/Metric Map Generator — 本地深度图、绝对测距与 3D 点云

一个本地优先的 Web 应用，能把普通照片和实时摄像头画面转换为**深度图**、**绝对米制距离测量**和**可交互的 3D 点云**。全部在自己机器上运行，不调用任何云端 API。

## 功能特性

- **深度图生成** —— 彩色与灰度，支持相对深度与绝对米制模型。
- **绝对米制测距** —— 图片模式下点击任意像素，或实时模式下用激光准星瞄准，即可得到以米为单位的距离。
- **实时摄像头模式** —— 实时深度画面 + 激光 HUD，显示 EMA 平滑后的距离与滚动波形图，可选光流目标锁定。
- **3D 点云** —— 融合相对深度与米制深度（Method A 全局尺度对齐），生成可交互的 Three.js 点云查看器。
- **按模型绑定经验深度范围** —— 每个模型携带各自预训练数据集（NYU / KITTI / Hypersim / VKITTI）的经验深度范围，由推理后端作为唯一数据源下发。
- **浮点目标点采样** —— 后端对目标像素做 float 双线性采样，避免 8-bit 量化抖动。
- **HTTPS / 局域网访问** —— 内置自签证书，同一局域网内的手机、平板可直接访问。
- **去水印 / 裁剪预设** —— 推理前裁掉图片边缘。

## 架构

```
浏览器 (React + Vite, :3000)
    │  /api/generate-depth          /api/generate-pointcloud
Node/Express 中转层 (server.ts, :3000)
    │  /depth, /models               │  /generate, /view
Python 推理服务 (:3001)              Python 点云服务 (:3002)
```

- **Node 中转层** —— 提供静态资源、转发请求、维护模型注册表；当 Python 后端不可用时，自动回退到 JS/ONNX 推理管线（`@xenova/transformers`）。
- **推理服务** —— FastAPI，以全精度运行 PyTorch / ONNX 模型。
- **点云服务** —— FastAPI，调用推理服务获取相对与米制原始深度，融合后生成 Three.js 查看器。

## 支持的模型

| 系列 | 模型 | 输出 |
|---|---|---|
| Depth Anything V2 | Small / Base / Large | 相对深度（视差） |
| Metric3D | ViT Small / Large / Giant2 | 绝对米制（需焦距） |
| ZoeDepth | N（室内）/ K（室外）/ NK（混合） | 绝对米制 |
| DA2 Metric | Indoor / Outdoor × Small / Base / Large | 绝对米制 |

每个模型首次推理时会从 Hugging Face 下载权重并缓存到本地。

## 部署与运行

### 环境要求

- **Node.js** 18+
- **Python** 3.10+
- （可选）支持 CUDA 的 GPU，用于加速 PyTorch / onnxruntime-gpu

### 安装

1. 安装 Node 依赖：
   ```bash
   npm install
   ```

2. 安装 Python 后端依赖：
   ```bash
   pip install -r python_backend/requirements.txt
   ```

3. 复制环境变量模板并按需调整：
   ```bash
   cp .env.example .env.local
   ```

### 启动

**一键启动全部服务（Python 推理 + 点云 + HTTPS Node 服务）：**
```bash
npm run dev:all:https
```

启动后访问：
- 本机：`https://localhost:3000`
- 局域网其他设备：`https://<你的局域网IP>:3000`（IP 在启动日志中会打印）

浏览器会提示证书不受信任 —— 点击 **高级 → 继续前往** 即可（自签证书的正常现象）。

> 其他启动方式：
> - `npm run dev:all` —— HTTP 模式启动全部服务
> - `npm run dev:https` —— 仅 Node + HTTPS（不含 Python 后端，自动回退 JS/ONNX）
> - `npm run dev` —— 仅 Node + HTTP（最简本地开发）

### 生产构建

```bash
npm run build
NODE_ENV=production node dist/server.cjs
```

## 使用方式

### 图片模式
上传图片 → 选择深度预览模型 → 可选开启「距离设定」→ **PROCESS IMAGE**。在结果图上点击任意位置即可放置测距 pin，显示该点距离（米）。

### 实时模式
切到 **Real-time** → **START CAMERA** → 开启 **Ranging HUD**。瞄准激光准星（或点击画面重新定位），读取实时距离；下方波形图展示最近测距历史。

### 点云
在图片模式下同时开启「深度预测」与「距离设定」，用支持米制的模型生成结果后，点击 **3D Point Cloud** 在新标签页打开交互式 Three.js 查看器。

## 配置项

关键环境变量（见 `.env.example`）：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `PYTHON_BACKEND_URL` | `http://127.0.0.1:3001` | 推理后端地址（Node 用） |
| `POINTCLOUD_BACKEND_URL` | `http://127.0.0.1:3002` | 点云后端地址（Node 用） |
| `INFERENCE_BACKEND_URL` | `http://127.0.0.1:3001` | 点云服务访问推理后端的地址 |
| `PYTHON_BACKEND_HOST` | `127.0.0.1` | 推理服务监听地址 |
| `POINTCLOUD_BACKEND_HOST` | `127.0.0.1` | 点云服务监听地址 |
| `HF_TOKEN` | — | Hugging Face token，提升下载速率与限额 |

## 注意事项

- Metric3D 等米制模型需要 Python 后端；若未运行，Node 会回退到 Depth Anything V2 Small（JS/ONNX）并给出提示。
- 想用 GPU 加速，请确保 Python 环境装了 CUDA 版的 `torch` 与 `onnxruntime-gpu`。
- 模型权重遵循各自上游协议；NYU / KITTI / Hypersim / VKITTI 等训练数据集通常带有非商业使用限制。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。模型权重保留各自上游协议。
