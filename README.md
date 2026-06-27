# Depth-Metric-Map-Generator
本项目是一个本地优先的 Web 应用，能把普通照片和实时摄像头画面转换为深度图、绝对米制距离测量和可交互的 3D 点云。它由三个服务协同：React 前端、带 JS/ONNX 兜底的 Node/Express 中转层，以及运行 Depth Anything V2、Metric3D、ZoeDepth、DA2 Metric 等模型的 Python FastAPI 后端。每个模型绑定各自预训练数据集的经验深度范围（由推理后端作为唯一数据源下发），前端结合后端返回的浮点目标点深度与 EMA 平滑来抑制实时测距中的量化抖动。支持图片和实时相机两种模式、HTTPS/局域网访问移动设备、去水印裁剪，以及基于 Three.js 的点云查看器。
