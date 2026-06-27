import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';

const SMOOTHING_ALPHA = 0.2;
import { useDropzone } from 'react-dropzone';
import { Client } from '@gradio/client';
import { motion, AnimatePresence } from 'motion/react';
import { UploadCloud, Image as ImageIcon, Loader2, Download, Layers, Target, RefreshCw, Ruler, Info, Box, AlertTriangle, Camera } from 'lucide-react';
import { ImageSlider } from './components/ImageSlider';
import { cn } from './lib/utils';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [croppedPreview, setCroppedPreview] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isGeneratingPointCloud, setIsGeneratingPointCloud] = useState(false);
  const [result, setResult] = useState<{
    colored: string;
    grayscale: string;
    depthGrayscale?: string;
    modelUsed?: string;
    warning?: string;
    metricStats?: { min_m: number; max_m: number; mean_m: number };
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [depthModel, setDepthModel] = useState('depth_anything_v2_base');
  const [metricModel, setMetricModel] = useState('metric3d_vit_small');
  const [availableModels, setAvailableModels] = useState<Array<{ id: string; displayName: string; family: string; depthRange?: { min: number; max: number; label: string } }>>([
    { id: 'depth_anything_v2_small', displayName: 'Depth Anything V2 Small', family: 'depth_anything_v2' },
    { id: 'depth_anything_v2_base', displayName: 'Depth Anything V2 Basic', family: 'depth_anything_v2' },
    { id: 'depth_anything_v2_large', displayName: 'Depth Anything V2 Large', family: 'depth_anything_v2' },
    { id: 'metric3d_vit_small', displayName: 'Metric3D ViT Small', family: 'metric3d' },
    { id: 'metric3d_vit_large', displayName: 'Metric3D ViT Large', family: 'metric3d' },
    { id: 'metric3d_vit_giant2', displayName: 'Metric3D ViT Giant2', family: 'metric3d' },
    { id: 'zoedepth_n', displayName: 'ZoeDepth N (Indoor)', family: 'metric3d' },
    { id: 'zoedepth_k', displayName: 'ZoeDepth K (Outdoor)', family: 'metric3d' },
    { id: 'zoedepth_nk', displayName: 'ZoeDepth NK (Mixed)', family: 'metric3d' },
    { id: 'da2_metric_indoor_small', displayName: 'DA2 Metric Indoor Small', family: 'metric3d' },
    { id: 'da2_metric_indoor_base', displayName: 'DA2 Metric Indoor Basic', family: 'metric3d' },
    { id: 'da2_metric_indoor_large', displayName: 'DA2 Metric Indoor Large', family: 'metric3d' },
    { id: 'da2_metric_outdoor_small', displayName: 'DA2 Metric Outdoor Small', family: 'metric3d' },
    { id: 'da2_metric_outdoor_base', displayName: 'DA2 Metric Outdoor Basic', family: 'metric3d' },
    { id: 'da2_metric_outdoor_large', displayName: 'DA2 Metric Outdoor Large', family: 'metric3d' },
  ]);
  const depthModels = availableModels.filter((m) => m.family === 'depth_anything_v2');
  const metricModels = availableModels.filter((m) => m.family === 'metric3d');
  const [isRealtimeMode, setIsRealtimeMode] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [realtimeDepthUrl, setRealtimeDepthUrl] = useState<string | null>(null);
  
  const [rawDepth, setRawDepth] = useState<number | null>(null);
  const depthCanvasRef = useRef<HTMLCanvasElement>(null);

  const [imageAspect, setImageAspect] = useState<number | null>(null);
  const [videoAspect, setVideoAspect] = useState<number | null>(null);

  // Metric3D Calibrated Ranging States
  const [metricDepthEnabled, setMetricDepthEnabled] = useState(false);
  const [metricDistanceEnabled, setMetricDistanceEnabled] = useState(false);
  const [metricScenario, setMetricScenario] = useState<'indoor' | 'outdoor'>('indoor');
  const [staticPins, setStaticPins] = useState<Array<{ id: number; x: number; y: number; distance: number; raw: number; depthRaw?: number }>>([]);

  // Camera intrinsic parameters for metric depth calibration
  // focal_length_px = physical_focal_mm * image_long_edge_px / cmos_long_edge_mm
  const [cameraPreset, setCameraPreset] = useState<'fullframe26' | 'custom'>('fullframe26');
  const [cmosWidth, setCmosWidth] = useState<number | null>(36.0); // mm
  const [physicalFocalLength, setPhysicalFocalLength] = useState<number | null>(26.0); // mm
  const [imageNaturalWidth, setImageNaturalWidth] = useState(1920); // px, updated on image load

  const computeFocalLengthPx = useCallback((widthPx: number): number => {
    if (cmosWidth == null || physicalFocalLength == null || cmosWidth <= 0 || physicalFocalLength <= 0) return 1000;
    return Math.round((physicalFocalLength * widthPx) / cmosWidth);
  }, [cmosWidth, physicalFocalLength]);

  // When switching preset, set or clear values
  const applyCameraPreset = (preset: 'fullframe26' | 'custom') => {
    setCameraPreset(preset);
    if (preset === 'fullframe26') {
      setCmosWidth(36.0);
      setPhysicalFocalLength(26.0);
    }
    // For 'custom', leave current values as-is (user will edit)
  };

  // Fetch available backend models
  useEffect(() => {
    fetch('/api/models')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.models) setAvailableModels(data.models);
      })
      .catch(() => {
        // keep defaults
      });
  }, []);

  const [rtMetricDepthEnabled, setRtMetricDepthEnabled] = useState(false);
  const [rtMetricDistanceEnabled, setRtMetricDistanceEnabled] = useState(false);
  const [rtTargetPoint, setRtTargetPoint] = useState<{ xPct: number; yPct: number } | null>(null);
  const [liveDistanceHistory, setLiveDistanceHistory] = useState<number[]>([]);
  const [rtMetricStats, setRtMetricStats] = useState<{ min_m: number; max_m: number; mean_m: number } | null>(null);
  const [smoothedDistance, setSmoothedDistance] = useState<number | null>(null);

  // Sync waveform history whenever smoothed distance changes (pure state update)
  useEffect(() => {
    if (smoothedDistance !== null) {
      setLiveDistanceHistory(prev => {
        const next = [...prev, smoothedDistance];
        if (next.length > 30) next.shift();
        return next;
      });
    }
  }, [smoothedDistance]);

  // Grayscale image buffer states for static clicking measurement
  const [grayscaleImageData, setGrayscaleImageData] = useState<ImageData | null>(null);
  const [grayscaleSize, setGrayscaleSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  // DA2 relative model's grayscale (for cross-comparison)
  const [depthGrayImageData, setDepthGrayImageData] = useState<ImageData | null>(null);

  // Depth-to-meters: only two paths for metric models.
  // 1. Per-image metric_stats from backend (already clipped to model range)
  // 2. Model-specific empirical range from pretraining data (served by /api/models)
  const modelRange = availableModels.find(m => m.id === metricModel)?.depthRange;

  // Two-point measurement mode (reference calibration)
  const [measureMode, setMeasureMode] = useState<'single' | 'two'>('single');
  const [twoPointPins, setTwoPointPins] = useState<Array<{ id: number; x: number; y: number }>>([]);
  const [referenceDistanceM, setReferenceDistanceM] = useState<number | null>(null);
  const [twoPointDistanceM, setTwoPointDistanceM] = useState<number | null>(null);

  const computeTwoPointDistance = useCallback(() => {
    if (twoPointPins.length !== 2 || !result?.grayscale || !grayscaleImageData) return;
    const [p1, p2] = twoPointPins;
    const imgW = grayscaleSize.width;
    const imgH = grayscaleSize.height;
    const x1 = p1.x / 100 * imgW;
    const y1 = p1.y / 100 * imgH;
    const x2 = p2.x / 100 * imgW;
    const y2 = p2.y / 100 * imgH;

    const pixelDistancePx = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    if (pixelDistancePx <= 0) {
      setTwoPointDistanceM(null);
      return;
    }

    const fx = computeFocalLengthPx(imgW);
    // D_camera = (focal_length_px * real_reference_length) / pixel_reference_length
    const realRef = referenceDistanceM ?? 0;
    if (realRef <= 0) {
      setTwoPointDistanceM(null);
      return;
    }

    const cameraDistance = (fx * realRef) / pixelDistancePx;
    setTwoPointDistanceM(Math.round(cameraDistance * 100) / 100);
  }, [twoPointPins, result, grayscaleImageData, grayscaleSize, computeFocalLengthPx, referenceDistanceM]);

  useEffect(() => {
    computeTwoPointDistance();
  }, [computeTwoPointDistance]);

  const calculateActualDistance = useCallback((
    rawVal: number,
    stats?: { min_m: number; max_m: number; mean_m: number } | null
  ): number => {
    if (stats && stats.max_m > stats.min_m) {
      const dist = stats.min_m + (rawVal / 255) * (stats.max_m - stats.min_m);
      return Math.round(dist * 100) / 100;
    }
    if (modelRange) {
      const dist = modelRange.min + (rawVal / 255) * (modelRange.max - modelRange.min);
      return Math.round(dist * 100) / 100;
    }
    return 0;
  }, [modelRange]);

  // Auto-sync scenario (safety thresholds) from model range
  useEffect(() => {
    if (modelRange) {
      setMetricScenario(modelRange.max <= 25 ? 'indoor' : 'outdoor');
    }
  }, [metricModel, modelRange]);

  // Recalculate pins when model or stats change
  useEffect(() => {
    setStaticPins(prev => prev.map(pin => ({
      ...pin,
      distance: calculateActualDistance(pin.raw, result?.metricStats)
    })));
  }, [calculateActualDistance, result?.metricStats]);

  // Parse grayscale image for static measurement clicks
  useEffect(() => {
    if (result?.grayscale) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0);
          try {
            const imgData = ctx.getImageData(0, 0, img.width, img.height);
            setGrayscaleImageData(imgData);
            setGrayscaleSize({ width: img.width, height: img.height });
          } catch (e) {
            console.error("Canvas image data extraction error:", e);
          }
        }
      };
      img.src = result.grayscale;
      setStaticPins([]);
    } else {
      setGrayscaleImageData(null);
      setStaticPins([]);
    }

    // Parse DA2 relative model's grayscale for cross-comparison
    if (result?.depthGrayscale) {
      const img2 = new Image();
      img2.crossOrigin = "anonymous";
      img2.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img2.width;
        canvas.height = img2.height;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img2, 0, 0);
          try {
            setDepthGrayImageData(ctx.getImageData(0, 0, img2.width, img2.height));
          } catch (e) {
            setDepthGrayImageData(null);
          }
        }
      };
      img2.src = result.depthGrayscale;
    } else {
      setDepthGrayImageData(null);
    }
  }, [result]);

  const handleStaticContainerClick = (xPct: number, yPct: number) => {
    if (measureMode === 'two') {
      if (twoPointPins.length >= 2) setTwoPointPins([]);
      setTwoPointPins(prev => [...prev, { id: Date.now(), x: xPct * 100, y: yPct * 100 }]);
      return;
    }

    if (!metricDistanceEnabled || !grayscaleImageData || !result) return;

    const originalX = Math.min(Math.max(0, Math.floor(xPct * grayscaleSize.width)), grayscaleSize.width - 1);
    const originalY = Math.min(Math.max(0, Math.floor(yPct * grayscaleSize.height)), grayscaleSize.height - 1);

    const pixelIndex = (originalY * grayscaleSize.width + originalX) * 4;
    const rawVal = grayscaleImageData.data[pixelIndex];

    // Also read DA2 relative model's value at the same pixel for cross-comparison
    let depthRaw: number | undefined;
    if (depthGrayImageData) {
      const dx = Math.min(Math.max(0, Math.floor(xPct * depthGrayImageData.width)), depthGrayImageData.width - 1);
      const dy = Math.min(Math.max(0, Math.floor(yPct * depthGrayImageData.height)), depthGrayImageData.height - 1);
      depthRaw = depthGrayImageData.data[(dy * depthGrayImageData.width + dx) * 4];
    }

    const distance = calculateActualDistance(rawVal, result.metricStats);

    const newPin = {
      id: Date.now(),
      x: xPct * 100,
      y: yPct * 100,
      raw: rawVal,
      depthRaw,
      distance: distance
    };

    setStaticPins(prev => [...prev, newPin]);
  };

  const handleDeleteStaticPin = (id: number) => {
    if (measureMode === 'two') {
      setTwoPointPins(prev => prev.filter(pin => pin.id !== id));
      return;
    }
    setStaticPins(prev => prev.filter(pin => pin.id !== id));
  };

  const handleRtFeedClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isCameraActive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setRtTargetPoint({ xPct: x, yPct: y });
  };

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = e.currentTarget;
    if (naturalWidth && naturalHeight) {
      setImageAspect(naturalWidth / naturalHeight);
      setImageNaturalWidth(Math.max(naturalWidth, naturalHeight));
    }
  };

  const handleVideoLoad = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const { videoWidth, videoHeight } = e.currentTarget;
    if (videoWidth && videoHeight) {
      setVideoAspect(videoWidth / videoHeight);
    }
  };

  // Watermark removal / Cropping states
  const [cropPreset, setCropPreset] = useState<string>('none'); // 'none', 'bottom12', 'bottom18', 'topbottom10', 'custom'
  const [cropTop, setCropTop] = useState<number>(0);
  const [cropBottom, setCropBottom] = useState<number>(0);
  const [cropLeft, setCropLeft] = useState<number>(0);
  const [cropRight, setCropRight] = useState<number>(0);

  // Synchronize cropping presets
  useEffect(() => {
    if (cropPreset === 'none') {
      setCropTop(0);
      setCropBottom(0);
      setCropLeft(0);
      setCropRight(0);
    } else if (cropPreset === 'bottom12') {
      setCropTop(0);
      setCropBottom(12);
      setCropLeft(0);
      setCropRight(0);
    } else if (cropPreset === 'bottom18') {
      setCropTop(0);
      setCropBottom(18);
      setCropLeft(0);
      setCropRight(0);
    } else if (cropPreset === 'topbottom10') {
      setCropTop(10);
      setCropBottom(10);
      setCropLeft(0);
      setCropRight(0);
    }
  }, [cropPreset]);

  // Utility to crop static images
  const getCroppedImageBlob = async (originalFile: File): Promise<Blob> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const sx = img.width * (cropLeft / 100);
        const sy = img.height * (cropTop / 100);
        const sWidth = img.width * (1 - (cropLeft + cropRight) / 100);
        const sHeight = img.height * (1 - (cropTop + cropBottom) / 100);

        const canvas = document.createElement('canvas');
        canvas.width = sWidth;
        canvas.height = sHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve(originalFile);
          return;
        }
        ctx.drawImage(img, sx, sy, sWidth, sHeight, 0, 0, sWidth, sHeight);
        canvas.toBlob((blob) => {
          if (blob) {
            resolve(blob);
          } else {
            resolve(originalFile);
          }
        }, originalFile.type || 'image/jpeg', 0.9);
      };
      img.onerror = () => {
        resolve(originalFile);
      };
      img.src = URL.createObjectURL(originalFile);
    });
  };

  const centerDepth = useMemo(() => {
    if (rawDepth === null) return null;
    return Math.round((rawDepth / 255) * 100); // percentage
  }, [rawDepth]);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Camera API requires a secure context (HTTPS or localhost).
  // On http://<IP> browsers set navigator.mediaDevices to undefined, which
  // would crash the component with a synchronous TypeError.
  const cameraAvailable = typeof navigator !== 'undefined' &&
    typeof navigator.mediaDevices !== 'undefined' &&
    typeof navigator.mediaDevices.getUserMedia === 'function';

  const startCamera = async (deviceIdToUse?: string) => {
    if (!cameraAvailable) {
      setError('摄像头不可用：通过 IP 地址访问需要 HTTPS 连接。请使用 localhost 访问，或为服务器配置 HTTPS 证书。');
      return;
    }
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }

      const activeDeviceId = deviceIdToUse || selectedDeviceId;
      const constraints: MediaStreamConstraints = {
        video: activeDeviceId ? { deviceId: { exact: activeDeviceId } } : { facingMode: 'environment' }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);

      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);

      if (!activeDeviceId && videoDevices.length > 0) {
        const activeTrack = stream.getVideoTracks()[0];
        const settings = activeTrack ? activeTrack.getSettings() : null;
        if (settings && settings.deviceId) {
          setSelectedDeviceId(settings.deviceId);
        } else {
          setSelectedDeviceId(videoDevices[0].deviceId);
        }
      }
    } catch (err: any) {
      const errName = err?.name || '';
      if (errName === 'NotAllowedError' || errName === 'SecurityError') {
        setError('摄像头访问被拒绝。浏览器在非 HTTPS 的 IP 访问时会阻止摄像头权限。请使用 localhost 或配置 HTTPS。');
      } else if (errName === 'NotFoundError' || errName === 'OverconstrainedError') {
        setError('未找到合适的摄像头设备。');
      } else {
        setError('无法访问摄像头，请检查权限设置后重试。');
      }
    }
  };

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setRealtimeDepthUrl(null);
    setRtMetricStats(null);
    setRtTargetPoint(null);
  }, []);

  // Real-time processing loop
  useEffect(() => {
    let isActive = true;
    let isProcessing = false;

    const processFrame = async () => {
      if (!isRealtimeMode || !isCameraActive || !videoRef.current || !canvasRef.current) return;
      if (isProcessing) {
        if (isActive) requestAnimationFrame(processFrame);
        return;
      }

      isProcessing = true;
      const video = videoRef.current;
      const canvas = canvasRef.current;

      let targetWidth = video.videoWidth;
      let targetHeight = video.videoHeight;
      const MAX_WIDTH = 640;
      if (targetWidth > MAX_WIDTH) {
        const ratio = MAX_WIDTH / targetWidth;
        targetWidth = MAX_WIDTH;
        targetHeight = Math.round(targetHeight * ratio);
      }

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, targetWidth, targetHeight);
          
          try {
            const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.6));
            if (blob) {
              const formData = new FormData();
              formData.append('image', blob, 'frame.jpg');
              formData.append('model', rtMetricDistanceEnabled ? metricModel : depthModel);
              formData.append('focal_length', String(computeFocalLengthPx(targetWidth)));
              if (rtTargetPoint) {
                formData.append('target_x_frac', String(rtTargetPoint.xPct));
                formData.append('target_y_frac', String(rtTargetPoint.yPct));
              }

              const response = await fetch('/api/generate-depth', {
                method: 'POST',
                body: formData,
              });

              if (response.ok) {
                const data = await response.json();
                if (data.colored && isActive) {
                  setRealtimeDepthUrl(data.colored);
                  setRtMetricStats(data.metricStats || null);
                  
                  // Prefer backend-exact target depth, fall back to grayscale pixel mapping
                  let currentDist: number | null = null;
                  let depthValue: number | null = null;
                  if (rtMetricDistanceEnabled && typeof data.targetDepthM === 'number') {
                    currentDist = data.targetDepthM;
                    depthValue = data.targetRawVal ?? rawDepth;
                  }
                  
                  if (data.grayscale) {
                    const img = new Image();
                    img.onload = () => {
                      const canvas = document.createElement('canvas');
                      canvas.width = img.width;
                      canvas.height = img.height;
                      const ctx = canvas.getContext('2d', { willReadFrequently: true });
                      if (ctx) {
                        ctx.drawImage(img, 0, 0);
                        
                        // Use customized click-to-range target if active, else center (0.5, 0.5)
                        const targetXFrac = rtTargetPoint ? rtTargetPoint.xPct : 0.5;
                        const targetYFrac = rtTargetPoint ? rtTargetPoint.yPct : 0.5;

                        const targetX = Math.min(Math.max(0, Math.floor(canvas.width * targetXFrac)), canvas.width - 1);
                        const targetY = Math.min(Math.max(0, Math.floor(canvas.height * targetYFrac)), canvas.height - 1);

                        const pixel = ctx.getImageData(targetX, targetY, 1, 1).data;
                        const pixelDepth = pixel[0]; // grayscale R=G=B
                        if (isActive) {
                          if (depthValue === null) {
                            depthValue = pixelDepth;
                          }
                          if (currentDist === null) {
                            currentDist = calculateActualDistance(pixelDepth, data.metricStats);
                          }
                          setRawDepth(depthValue);
                          setSmoothedDistance(prev => prev === null ? currentDist! : SMOOTHING_ALPHA * currentDist! + (1 - SMOOTHING_ALPHA) * prev);
                        }
                      }
                    };
                    img.src = data.grayscale;
                  } else if (currentDist !== null && isActive) {
                    setRawDepth(data.targetRawVal ?? null);
                    setSmoothedDistance(prev => prev === null ? currentDist! : SMOOTHING_ALPHA * currentDist! + (1 - SMOOTHING_ALPHA) * prev);
                  }
                }
              }
            }
          } catch (e) {
            console.error('Frame processing error:', e);
          }
        }
      }
      isProcessing = false;
      
      // Delay slightly to not overwhelm backend
      if (isActive) {
        setTimeout(() => requestAnimationFrame(processFrame), 200);
      }
    };

    if (isRealtimeMode && isCameraActive) {
      processFrame();
    }

    return () => {
      isActive = false;
    };
  }, [isRealtimeMode, isCameraActive, cropLeft, cropRight, cropTop, cropBottom, rtTargetPoint, metricScenario, calculateActualDistance, depthModel, metricModel, rtMetricDistanceEnabled, computeFocalLengthPx]);

  // Load devices on mode switch
  useEffect(() => {
    if (isRealtimeMode && cameraAvailable) {
      navigator.mediaDevices!.enumerateDevices()
        .then(allDevices => {
          const videoDevices = allDevices.filter(device => device.kind === 'videoinput');
          setDevices(videoDevices);
        })
        .catch(err => console.error("Enumerate devices error:", err));
    }
  }, [isRealtimeMode, cameraAvailable]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setCroppedPreview(null);
      setImageAspect(null);
      setResult(null);
      setError(null);
    }
  }, []);

  // @ts-ignore
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    maxFiles: 1,
  });

  const generateDepthMap = async () => {
    if (!file) return;
    setIsGenerating(true);
    setError(null);

    try {
      // Crop image if watermark removal / crop settings are active
      let imageToUpload: Blob = file;
      let croppedUrl = preview;
      if (cropTop > 0 || cropBottom > 0 || cropLeft > 0 || cropRight > 0) {
        const croppedBlob = await getCroppedImageBlob(file);
        imageToUpload = croppedBlob;
        croppedUrl = URL.createObjectURL(croppedBlob);
      }

      // Reference calibration mode does not need depth inference.
      if (measureMode === 'two') {
        setCroppedPreview(croppedUrl);
        setResult({
          colored: croppedUrl!,
          grayscale: croppedUrl!,
          modelUsed: 'Reference Calibration',
          warning: '参考标定模式：未运行深度模型，使用原图进行标定。',
          metricStats: undefined,
        });
        return;
      }

      // Determine which models to run:
      // - Depth preview: Depth Anything V2 Basic
      // - Ranging: Metric3D ViT Small
      const runDepth = metricDepthEnabled;
      const runMetric = metricDistanceEnabled;
      const requests: Array<Promise<{ model: string; data: any }>> = [];

      if (runDepth || (!runDepth && !runMetric)) {
        const fd = new FormData();
        fd.append('image', imageToUpload, file.name);
        fd.append('model', depthModel);
        requests.push(
          fetch('/api/generate-depth', { method: 'POST', body: fd })
            .then(async (res) => {
              if (!res.ok) throw new Error(`Depth model failed: ${await res.text()}`);
              return { model: 'depth', data: await res.json() };
            })
        );
      }

      if (runMetric) {
        const fd = new FormData();
        fd.append('image', imageToUpload, file.name);
        fd.append('model', metricModel);
        fd.append('focal_length', String(computeFocalLengthPx(imageNaturalWidth)));
        requests.push(
          fetch('/api/generate-depth', { method: 'POST', body: fd })
            .then(async (res) => {
              if (!res.ok) throw new Error(`Metric3D failed: ${await res.text()}`);
              return { model: 'metric', data: await res.json() };
            })
        );
      }

      const results = await Promise.all(requests);
      const depthResult = results.find((r) => r.model === 'depth')?.data;
      const metricResult = results.find((r) => r.model === 'metric')?.data;

      // Prefer metric depth for ranging; otherwise use depth preview result
      const primaryResult = metricResult || depthResult;
      const previewResult = depthResult || metricResult;

      if (primaryResult?.colored && primaryResult?.grayscale) {
        setCroppedPreview(croppedUrl);
        setResult({
          colored: previewResult.colored,
          grayscale: primaryResult.grayscale,
          depthGrayscale: depthResult?.grayscale,
          modelUsed: previewResult.modelUsed,
          warning: metricResult && depthResult
            ? `Depth preview: ${depthResult.modelUsed} | Ranging: ${metricResult.modelUsed}`
            : primaryResult.warning,
          metricStats: metricResult?.metricStats,
        });
      } else {
        setError("Failed to parse depth map from the response.");
      }
    } catch (err: any) {
      setError(err.message || "An error occurred while generating the depth map.");
    } finally {
      setIsGenerating(false);
    }
  };

  const generatePointCloud = async () => {
    if (!file) return;
    setIsGeneratingPointCloud(true);
    setError(null);

    try {
      let imageToUpload: Blob = file;
      if (cropTop > 0 || cropBottom > 0 || cropLeft > 0 || cropRight > 0) {
        imageToUpload = await getCroppedImageBlob(file);
      }

      const fd = new FormData();
      fd.append('image', imageToUpload, file.name);
      fd.append('metric_model', metricModel);
      fd.append('relative_model', depthModel);
      fd.append('focal_length', String(computeFocalLengthPx(imageNaturalWidth)));

      const res = await fetch('/api/generate-pointcloud', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.viewer_url) {
        window.open(data.viewer_url, '_blank');
      } else {
        throw new Error('Point cloud viewer URL missing');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to generate point cloud.');
    } finally {
      setIsGeneratingPointCloud(false);
    }
  };

  return (
    <div className="w-full h-screen bg-slate-950 text-slate-100 flex flex-col font-sans overflow-hidden">
      {/* Header */}
      <header className="h-16 border-b border-slate-800 flex items-center justify-between px-4 md:px-8 bg-slate-900/50 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Layers size={18} className="text-white" />
          </div>
          <span className="text-lg font-bold tracking-tight uppercase hidden md:inline-block">Depth / Metric Map Generator</span>
        </div>
          <div className="flex items-center gap-4 md:gap-6">
            {result?.modelUsed && (
              <div className={cn("flex items-center gap-2 text-xs font-medium text-slate-400", measureMode === 'two' && "opacity-50")}>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>{result.modelUsed.toUpperCase()}</span>
              </div>
            )}
          </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-6 md:p-8 flex flex-col md:flex-row gap-8">
        
        {/* Sidebar Controls */}
        <aside className="w-full md:w-80 border border-slate-800 bg-slate-900/30 p-6 flex flex-col gap-8 rounded-2xl h-fit shrink-0">
          <section>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Input Mode</h3>
            <div className="flex bg-slate-950 p-1 rounded-lg">
              <button 
                onClick={() => { setIsRealtimeMode(false); stopCamera(); }}
                className={cn("flex-1 py-2 text-sm font-medium rounded-md transition-all", !isRealtimeMode ? "bg-slate-800 text-white shadow" : "text-slate-400 hover:text-slate-200")}
              >
                Image
              </button>
              <button 
                onClick={() => setIsRealtimeMode(true)}
                className={cn("flex-1 py-2 text-sm font-medium rounded-md transition-all", isRealtimeMode ? "bg-slate-800 text-white shadow" : "text-slate-400 hover:text-slate-200")}
              >
                Real-time
              </button>
            </div>
          </section>

          {!isRealtimeMode && (
            <section>
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Depth Model</h3>
              <div className={cn("space-y-3", measureMode === 'two' && "opacity-50 pointer-events-none")}>
                {measureMode === 'two' && (
                  <div className="p-2.5 bg-amber-500/10 border border-amber-500/20 rounded-lg text-[10px] text-amber-400 leading-relaxed">
                    参考标定模式不依赖深度模型。请确保已正确填写相机参数。
                  </div>
                )}
                {(metricDepthEnabled && measureMode !== 'two') && (
                  <div>
                    <label className="text-[11px] text-slate-400 mb-2 block">DEPTH PREVIEW MODEL (深度预测模型)</label>
                    <select
                      className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-sm text-slate-200 outline-none focus:border-indigo-500 cursor-pointer font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                      value={depthModel}
                      onChange={(e) => setDepthModel(e.target.value)}
                      disabled={measureMode === 'two'}
                    >
                      {depthModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.displayName}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {metricDistanceEnabled && measureMode !== 'two' && (
                  <>
                  <div>
                    <label className="text-[11px] text-slate-400 mb-2 block">RANGING MODEL (测距模型)</label>
                    <select
                      className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-sm text-slate-200 outline-none focus:border-indigo-500 cursor-pointer font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                      value={metricModel}
                      onChange={(e) => setMetricModel(e.target.value)}
                      disabled={measureMode === 'two'}
                    >
                      {metricModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.displayName}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Camera intrinsic parameters */}
                  <div className="space-y-2.5 p-3 bg-slate-950/40 rounded-lg border border-slate-800/50">
                    <div className="flex items-center gap-1.5">
                      <Camera size={11} className="text-slate-500" />
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">相机参数 (Camera Intrinsics)</span>
                    </div>

                    <div className="flex gap-1">
                      <button
                        onClick={() => applyCameraPreset('fullframe26')}
                        className={cn(
                          "flex-1 px-2 py-1.5 rounded text-[9px] font-bold transition-all",
                          cameraPreset === 'fullframe26'
                            ? "bg-indigo-600 text-white"
                            : "bg-slate-900 text-slate-400 hover:bg-slate-800 border border-slate-800"
                        )}
                      >
                        全画幅 26mm
                      </button>
                      <button
                        onClick={() => applyCameraPreset('custom')}
                        className={cn(
                          "flex-1 px-2 py-1.5 rounded text-[9px] font-bold transition-all",
                          cameraPreset === 'custom'
                            ? "bg-indigo-600 text-white"
                            : "bg-slate-900 text-slate-400 hover:bg-slate-800 border border-slate-800"
                        )}
                      >
                        自定义
                      </button>
                    </div>

                    {cameraPreset === 'custom' && (
                      <div className="space-y-2">
                        <div>
                          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                            <span>CMOS长边 (Sensor Width)</span>
                            <span className="font-mono text-indigo-400">{cmosWidth != null ? `${cmosWidth}mm` : '—'}</span>
                          </div>
                          <input
                            type="number"
                            step="0.1"
                            value={cmosWidth ?? ''}
                            onChange={(e) => {
                              const v = e.target.value;
                              setCmosWidth(v === '' ? null : parseFloat(v));
                            }}
                            placeholder="留空则使用默认"
                            className="w-full bg-slate-950 border border-slate-800 rounded-md px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500 font-medium placeholder:text-slate-600"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                            <span>物理焦距 (Focal Length)</span>
                            <span className="font-mono text-indigo-400">{physicalFocalLength != null ? `${physicalFocalLength}mm` : '—'}</span>
                          </div>
                          <input
                            type="number"
                            step="0.5"
                            value={physicalFocalLength ?? ''}
                            onChange={(e) => {
                              const v = e.target.value;
                              setPhysicalFocalLength(v === '' ? null : parseFloat(v));
                            }}
                            placeholder="留空则使用默认"
                            className="w-full bg-slate-950 border border-slate-800 rounded-md px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500 font-medium placeholder:text-slate-600"
                          />
                        </div>
                      </div>
                    )}

                    <div className="flex justify-between items-center pt-1.5 border-t border-slate-800/50">
                      <span className="text-[10px] text-slate-500">计算像素焦距 fx</span>
                      <span className="font-mono font-bold text-emerald-400 text-[11px]">{computeFocalLengthPx(imageNaturalWidth)}px</span>
                    </div>
                  </div>
                  </>
                )}

                {(metricDepthEnabled || metricDistanceEnabled) && (
                  <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-lg text-[10px] text-slate-400 leading-relaxed">
                    {metricDepthEnabled && metricDistanceEnabled ? (
                      <span>
                        <span className="font-semibold text-blue-400">{availableModels.find(m => m.id === depthModel)?.displayName || 'Depth Anything V2'}</span> for depth preview,{' '}
                        <span className="font-semibold text-indigo-400">{availableModels.find(m => m.id === metricModel)?.displayName || 'Metric3D'}</span> for metric ranging.
                      </span>
                    ) : metricDepthEnabled ? (
                      <span>
                        <span className="font-semibold text-blue-400">{availableModels.find(m => m.id === depthModel)?.displayName || 'Depth Anything V2'}</span> runs locally. The Python backend provides the full PyTorch model; JS/ONNX is used as a fallback.
                      </span>
                    ) : (
                      <span>
                        <span className="font-semibold text-indigo-400">{availableModels.find(m => m.id === metricModel)?.displayName || 'Metric3D'}</span> for metric ranging. Make sure camera intrinsics are set.
                      </span>
                    )}
                  </div>
                )}
              </div>
            </section>
          )}

          {!isRealtimeMode && (
            <section className="border-t border-slate-800/80 pt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Watermark / Crop</h3>
                <span className="text-[10px] bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded-full font-bold uppercase font-mono font-bold">Algorithm</span>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="text-[11px] text-slate-400 mb-2 block">CROP PRESET (去水印/裁剪预设)</label>
                  <select 
                    className="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-sm text-slate-200 outline-none focus:border-indigo-500 cursor-pointer font-medium"
                    value={cropPreset}
                    onChange={(e) => setCropPreset(e.target.value)}
                  >
                    <option value="none">No Crop (不裁剪 / 无水印)</option>
                    <option value="bottom12">Bottom 12% (去除常规底部水印)</option>
                    <option value="bottom18">Bottom 18% (去除底部大型水印)</option>
                    <option value="topbottom10">Top & Bottom 10% (电影画幅上下裁剪)</option>
                    <option value="custom">Custom Crop (自定义裁剪范围)</option>
                  </select>
                </div>

                {cropPreset === 'custom' && (
                  <div className="space-y-3 bg-slate-950/40 p-3 rounded-lg border border-slate-800/50">
                    <div>
                      <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>CROP TOP (顶部)</span>
                        <span className="font-mono text-indigo-400 font-bold">{cropTop}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="40" 
                        className="w-full h-1 bg-slate-805 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        value={cropTop} 
                        onChange={(e) => setCropTop(Number(e.target.value))} 
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>CROP BOTTOM (底部)</span>
                        <span className="font-mono text-indigo-400 font-bold">{cropBottom}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="40" 
                        className="w-full h-1 bg-slate-805 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        value={cropBottom} 
                        onChange={(e) => setCropBottom(Number(e.target.value))} 
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>CROP LEFT (左侧)</span>
                        <span className="font-mono text-indigo-400 font-bold">{cropLeft}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="40" 
                        className="w-full h-1 bg-slate-805 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        value={cropLeft} 
                        onChange={(e) => setCropLeft(Number(e.target.value))} 
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>CROP RIGHT (右侧)</span>
                        <span className="font-mono text-indigo-400 font-bold">{cropRight}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="40" 
                        className="w-full h-1 bg-slate-805 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        value={cropRight} 
                        onChange={(e) => setCropRight(Number(e.target.value))} 
                      />
                    </div>
                  </div>
                )}

                {(cropTop > 0 || cropBottom > 0 || cropLeft > 0 || cropRight > 0) && (
                  <div className="p-2.5 bg-indigo-500/5 rounded-lg border border-indigo-500/10 text-[10px] text-slate-400 leading-relaxed flex items-start gap-1.5">
                    <span className="text-indigo-400 font-bold">ℹ</span>
                    <span>Active crops will remove specified edges to exclude watermarks/labels before depth calculation.</span>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Static Metric3D Controls */}
          {!isRealtimeMode && (
            <section className="border-t border-slate-800/80 pt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                  <Ruler size={13} className="text-indigo-400" />
                  <span>Metric3D 静态测距</span>
                </h3>
                <span className="text-[10px] bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded-full font-bold uppercase font-mono">Calibrated</span>
              </div>
              
              <div className="space-y-4">
                <div className="flex bg-slate-950 p-1 rounded-lg">
                  <button
                    onClick={() => setMeasureMode('single')}
                    className={cn("flex-1 py-2 text-sm font-medium rounded-md transition-all", measureMode === 'single' ? "bg-slate-800 text-white shadow" : "text-slate-400 hover:text-slate-200")}
                  >
                    深度复原
                  </button>
                  <button
                    onClick={() => setMeasureMode('two')}
                    className={cn("flex-1 py-2 text-sm font-medium rounded-md transition-all", measureMode === 'two' ? "bg-slate-800 text-white shadow" : "text-slate-400 hover:text-slate-200")}
                  >
                    参考标定
                  </button>
                </div>

                {measureMode === 'single' ? (
                <>
                {/* Model depth range — read-only, auto-derived from selected model */}
                {(metricDepthEnabled || metricDistanceEnabled) && (
                <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800/50">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">模型深度范围 (Model Range)</span>
                    <span className="text-[10px] font-mono text-indigo-400">{modelRange?.label || '--'}</span>
                  </div>
                  <div className="flex items-center justify-between mt-1.5">
                    <span className="text-[10px] text-slate-500">安全阈值 (自动)</span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {metricScenario === 'indoor' ? '🏠 告警<1m / 警告<2.5m' : '🌳 告警<3m / 警告<10m'}
                    </span>
                  </div>
                </div>
                )}

                {/* Checkbox 1: Depth Map (深度预测) */}
                <label className="flex items-start gap-3 p-3 bg-slate-950/40 hover:bg-slate-950/80 rounded-xl border border-slate-800/80 cursor-pointer transition-all group">
                  <input 
                    type="checkbox" 
                    className="mt-0.5 h-4 w-4 rounded border-slate-700 text-indigo-600 focus:ring-indigo-500 accent-indigo-500 cursor-pointer"
                    checked={metricDepthEnabled}
                    onChange={(e) => setMetricDepthEnabled(e.target.checked)}
                  />
                  <div className="flex-1">
                    <span className="text-xs font-bold text-slate-200 block group-hover:text-indigo-400 transition-colors">深度预测 (Depth Map)</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5 leading-normal">勾选后启用相对深度预测</span>
                  </div>
                </label>

                {/* Checkbox 2: Distance Measurement (距离设定) */}
                <label className="flex items-start gap-3 p-3 bg-slate-950/40 hover:bg-slate-950/80 rounded-xl border border-slate-800/80 cursor-pointer transition-all group">
                  <input 
                    type="checkbox" 
                    className="mt-0.5 h-4 w-4 rounded border-slate-700 text-indigo-600 focus:ring-indigo-500 accent-indigo-500 cursor-pointer"
                    checked={metricDistanceEnabled}
                    onChange={(e) => setMetricDistanceEnabled(e.target.checked)}
                  />
                  <div className="flex-1">
                    <span className="text-xs font-bold text-slate-200 block group-hover:text-indigo-400 transition-colors">距离设定 (Ranging Pins)</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5 leading-normal">勾选后启用米级测距（需要相机参数）</span>
                  </div>
                </label>
                </>
              ) : (
                <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800/50 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">相机到物体距离</span>
                    <span className="text-lg font-mono font-black text-indigo-400">{twoPointDistanceM !== null ? `${twoPointDistanceM.toFixed(2)}m` : '--'}</span>
                  </div>
                  <p className="text-[10px] text-slate-500 leading-relaxed">
                    在画面中标定一段已知实际长度的参考线段（如身高、门宽），输入其真实长度，即可用相似三角形估算相机到该物体的距离。
                  </p>
                  <div>
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>参考线段真实长度 (m)</span>
                      <span className="font-mono text-indigo-400">{referenceDistanceM != null ? `${referenceDistanceM}m` : '—'}</span>
                    </div>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      value={referenceDistanceM ?? ''}
                      onChange={(e) => {
                        const v = e.target.value;
                        setReferenceDistanceM(v === '' ? null : parseFloat(v));
                      }}
                      placeholder="例如：1.75"
                      className="w-full bg-slate-950 border border-slate-800 rounded-md px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500 font-medium placeholder:text-slate-600"
                    />
                  </div>
                  {twoPointPins.length > 0 && (
                    <button
                      onClick={() => setTwoPointPins([])}
                      className="w-full py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded text-[10px] text-slate-400 font-bold transition-colors"
                    >
                      清除参考点 (Clear Points)
                    </button>
                  )}
                </div>
              )}
              </div>
            </section>
          )}

          {/* Real-time Metric3D Controls */}
          {isRealtimeMode && (
            <section className="border-t border-slate-800/80 pt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                  <Target size={13} className="text-red-400" />
                  <span>Metric3D 实时测距</span>
                </h3>
                <span className="text-[10px] bg-red-500/10 text-red-400 border border-red-500/20 px-2 py-0.5 rounded-full font-bold uppercase font-mono">LIVE</span>
              </div>
              
              <div className="space-y-4">
                {/* Model depth range — read-only, auto-derived from selected model */}
                <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800/50">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">模型深度范围 (Model Range)</span>
                    <span className="text-[10px] font-mono text-red-400">{modelRange?.label || '--'}</span>
                  </div>
                  <div className="flex items-center justify-between mt-1.5">
                    <span className="text-[10px] text-slate-500">安全阈值 (自动)</span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {metricScenario === 'indoor' ? '🏠 告警<1m / 警告<2.5m' : '🌳 告警<3m / 警告<10m'}
                    </span>
                  </div>
                </div>

                {/* Real-time Depth Model Selector */}
                <div>
                  <label className="text-[10px] font-bold text-slate-400 mb-1.5 block uppercase tracking-wider">实时深度模型 (Live Depth Model)</label>
                  <select
                    className="w-full bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 outline-none focus:border-red-500 cursor-pointer font-medium transition-colors"
                    value={depthModel}
                    onChange={(e) => setDepthModel(e.target.value)}
                  >
                    {depthModels.map((m) => (
                      <option key={m.id} value={m.id}>{m.displayName}</option>
                    ))}
                  </select>
                </div>

                {/* Real-time Ranging Model Selector (only when ranging enabled) */}
                {rtMetricDistanceEnabled && (
                  <>
                  <div>
                    <label className="text-[10px] font-bold text-slate-400 mb-1.5 block uppercase tracking-wider">实时测距模型 (Live Ranging Model)</label>
                    <select
                      className="w-full bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 outline-none focus:border-red-500 cursor-pointer font-medium transition-colors"
                      value={metricModel}
                      onChange={(e) => setMetricModel(e.target.value)}
                    >
                      {metricModels.map((m) => (
                        <option key={m.id} value={m.id}>{m.displayName}</option>
                      ))}
                    </select>
                  </div>

                  {/* Camera intrinsic parameters (shared with static mode) */}
                  <div className="space-y-2.5 p-3 bg-slate-950/40 rounded-lg border border-slate-800/50">
                    <div className="flex items-center gap-1.5">
                      <Camera size={11} className="text-slate-500" />
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">相机参数 (Camera Intrinsics)</span>
                    </div>

                    <div className="flex gap-1">
                      <button
                        onClick={() => applyCameraPreset('fullframe26')}
                        className={cn(
                          "flex-1 px-2 py-1.5 rounded text-[9px] font-bold transition-all",
                          cameraPreset === 'fullframe26'
                            ? "bg-red-600 text-white"
                            : "bg-slate-900 text-slate-400 hover:bg-slate-800 border border-slate-800"
                        )}
                      >
                        全画幅 26mm
                      </button>
                      <button
                        onClick={() => applyCameraPreset('custom')}
                        className={cn(
                          "flex-1 px-2 py-1.5 rounded text-[9px] font-bold transition-all",
                          cameraPreset === 'custom'
                            ? "bg-red-600 text-white"
                            : "bg-slate-900 text-slate-400 hover:bg-slate-800 border border-slate-800"
                        )}
                      >
                        自定义
                      </button>
                    </div>

                    {cameraPreset === 'custom' && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-[10px] text-slate-400">
                          <span>CMOS长边 / 物理焦距</span>
                          <span className="font-mono text-red-400">{cmosWidth != null ? `${cmosWidth}mm` : '—'} / {physicalFocalLength != null ? `${physicalFocalLength}mm` : '—'}</span>
                        </div>
                        <div className="flex gap-2">
                          <input
                            type="number"
                            step="0.1"
                            value={cmosWidth ?? ''}
                            onChange={(e) => {
                              const v = e.target.value;
                              setCmosWidth(v === '' ? null : parseFloat(v));
                            }}
                            placeholder="CMOS mm"
                            className="flex-1 bg-slate-950 border border-slate-800 rounded-md px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-red-500 font-medium placeholder:text-slate-600"
                          />
                          <input
                            type="number"
                            step="0.5"
                            value={physicalFocalLength ?? ''}
                            onChange={(e) => {
                              const v = e.target.value;
                              setPhysicalFocalLength(v === '' ? null : parseFloat(v));
                            }}
                            placeholder="焦距 mm"
                            className="flex-1 bg-slate-950 border border-slate-800 rounded-md px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-red-500 font-medium placeholder:text-slate-600"
                          />
                        </div>
                      </div>
                    )}

                    <div className="flex justify-between items-center pt-1.5 border-t border-slate-800/50">
                      <span className="text-[10px] text-slate-500">计算像素焦距 fx (基于640px)</span>
                      <span className="font-mono font-bold text-emerald-400 text-[11px]">{computeFocalLengthPx(640)}px</span>
                    </div>
                  </div>
                  </>
                )}

                {/* Checkbox 1: Depth Map (深度预测) */}
                <label className="flex items-start gap-3 p-3 bg-slate-950/40 hover:bg-slate-950/80 rounded-xl border border-slate-800/80 cursor-pointer transition-all group">
                  <input 
                    type="checkbox" 
                    className="mt-0.5 h-4 w-4 rounded border-slate-700 text-red-600 focus:ring-red-500 accent-red-500 cursor-pointer"
                    checked={rtMetricDepthEnabled}
                    onChange={(e) => setRtMetricDepthEnabled(e.target.checked)}
                  />
                  <div className="flex-1">
                    <span className="text-xs font-bold text-slate-200 block group-hover:text-red-400 transition-colors">深度预测 (Live Scale)</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5 leading-normal">在实时深度流图像右下角动态显示绝对米级深度色标</span>
                  </div>
                </label>

                {/* Checkbox 2: Distance Measurement (距离设定) */}
                <label className="flex items-start gap-3 p-3 bg-slate-950/40 hover:bg-slate-950/80 rounded-xl border border-slate-800/80 cursor-pointer transition-all group">
                  <input 
                    type="checkbox" 
                    className="mt-0.5 h-4 w-4 rounded border-slate-700 text-red-600 focus:ring-red-500 accent-red-500 cursor-pointer"
                    checked={rtMetricDistanceEnabled}
                    onChange={(e) => setRtMetricDistanceEnabled(e.target.checked)}
                  />
                  <div className="flex-1">
                    <span className="text-xs font-bold text-slate-200 block group-hover:text-red-400 transition-colors">距离设定 (Ranging HUD)</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5 leading-normal">激活画面中央高精激光准星雷达波动动态诊断波形图</span>
                  </div>
                </label>
              </div>
            </section>
          )}

          {!isRealtimeMode && measureMode !== 'two' && (metricDepthEnabled || metricDistanceEnabled) && (
            <section>
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Model Information</h3>
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-400 leading-relaxed">
                {metricDepthEnabled && metricDistanceEnabled ? (
                  <span><span className="font-semibold text-blue-400">{availableModels.find(m => m.id === depthModel)?.displayName || 'Depth Anything V2'}</span> for depth preview, <span className="font-semibold text-indigo-400">{availableModels.find(m => m.id === metricModel)?.displayName || 'Metric3D'}</span> for metric ranging.</span>
                ) : metricDepthEnabled ? (
                  <span><span className="font-semibold text-blue-400">{availableModels.find(m => m.id === depthModel)?.displayName || 'Depth Anything V2'}</span> runs locally on the server for fast, private depth extraction.</span>
                ) : (
                  <span><span className="font-semibold text-indigo-400">{availableModels.find(m => m.id === metricModel)?.displayName || 'Metric3D'}</span> runs on the server for metric ranging.</span>
                )}
              </div>
            </section>
          )}

          {!isRealtimeMode && (
            <div className="mt-auto pt-8">
              <button 
                onClick={generateDepthMap}
                disabled={isGenerating || !file}
                className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 rounded-xl font-bold text-white shadow-xl shadow-blue-900/20 active:scale-95 transition-transform disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 tracking-widest text-sm"
              >
                {isGenerating ? (
                  <>
                    <Loader2 size={20} className="animate-spin" />
                    PROCESSING...
                  </>
                ) : 'PROCESS IMAGE'}
              </button>
            </div>
          )}
        </aside>

        {/* Editor Viewport */}
        <div className="flex-1 max-w-4xl w-full flex flex-col gap-8 mx-auto">
        {isRealtimeMode ? (
          !cameraAvailable ? (
            <div className="flex flex-col items-center justify-center min-h-[400px] bg-slate-900 border border-yellow-500/20 rounded-2xl p-8 text-center gap-4">
              <AlertTriangle size={48} className="text-yellow-500" />
              <h2 className="text-xl font-bold text-yellow-400">摄像头不可用 (Camera Unavailable)</h2>
              <p className="text-sm text-slate-400 max-w-md">
                通过 IP 地址以 HTTP 方式访问时，浏览器出于安全限制会禁用摄像头 API。
                实时模式仅可在以下环境中使用：
              </p>
              <div className="text-xs text-slate-500 space-y-2 text-left max-w-md bg-slate-950/50 border border-slate-800 rounded-xl p-4 mt-2">
                <div className="flex items-start gap-2">
                  <span className="text-green-400 font-bold mt-0.5">1.</span>
                  <span>在本机使用 <code className="text-green-400 bg-slate-800 px-1.5 py-0.5 rounded text-[11px]">http://localhost:3000</code> 访问</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400 font-bold mt-0.5">2.</span>
                  <span>为服务器配置 HTTPS 证书后通过 <code className="text-green-400 bg-slate-800 px-1.5 py-0.5 rounded text-[11px]">https://你的IP:3000</code> 访问</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400 font-bold mt-0.5">3.</span>
                  <span>使用 <code className="text-green-400 bg-slate-800 px-1.5 py-0.5 rounded text-[11px]">ngrok</code> / <code className="text-green-400 bg-slate-800 px-1.5 py-0.5 rounded text-[11px]">cloudflared</code> 等内网穿透工具（自动提供 HTTPS）</span>
                </div>
              </div>
              <p className="text-[11px] text-slate-600 mt-1">图片处理模式（Image 模式）不受此限制，可正常使用。</p>
            </div>
          ) : (
          <div className="flex flex-col gap-6 w-full h-full">
            <div className="flex flex-col sm:flex-row gap-4 items-stretch sm:items-center bg-slate-900/30 border border-slate-800/80 p-4 rounded-2xl">
              <button 
                onClick={isCameraActive ? stopCamera : () => startCamera()}
                className={cn("px-6 py-3 rounded-xl font-bold text-white shadow-lg transition-transform active:scale-95 shrink-0", isCameraActive ? "bg-red-600 hover:bg-red-500 shadow-red-900/20" : "bg-blue-600 hover:bg-blue-500 shadow-blue-900/20")}
              >
                {isCameraActive ? 'STOP CAMERA' : 'START CAMERA'}
              </button>

              {devices.length > 0 && (
                <div className="flex-1 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider shrink-0 pl-1">
                    Select Camera:
                  </label>
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => {
                      const newId = e.target.value;
                      setSelectedDeviceId(newId);
                      if (isCameraActive) {
                        startCamera(newId);
                      }
                    }}
                    className="flex-1 bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 outline-none focus:border-blue-500 font-medium cursor-pointer transition-colors"
                  >
                    {devices.map((device, index) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${index + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
            
            <div className={cn("grid gap-6 h-full min-h-[400px]", rtMetricDepthEnabled ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1")}>
              {/* Camera Feed Container */}
              <div 
                className="flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden relative min-h-[300px] cursor-crosshair group"
                onClick={handleRtFeedClick}
                title={isCameraActive ? "点击画面设置对焦测距点" : ""}
              >
                <div className="absolute top-4 left-4 bg-black/65 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold tracking-tighter z-20 text-white border border-slate-800 flex items-center gap-1">
                  <div className={`w-1.5 h-1.5 rounded-full ${isCameraActive ? 'bg-red-500 animate-pulse' : 'bg-slate-500'}`} />
                  <span>CAMERA FEED</span>
                </div>
                <div className="flex-1 flex items-center justify-center bg-black h-full relative">
                  <div 
                    className="relative max-w-full max-h-full w-full h-full flex items-center justify-center transition-all duration-300"
                    style={{ 
                      aspectRatio: videoAspect ? `${videoAspect}` : 'auto',
                      height: videoAspect ? 'auto' : '100%'
                    }}
                  >
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline 
                      muted 
                      onLoadedMetadata={handleVideoLoad}
                      className="w-full h-full object-contain pointer-events-none" 
                    />
                    <canvas ref={canvasRef} className="hidden" />
                    <canvas ref={depthCanvasRef} className="hidden" />
                    
                    {/* Render live Laser HUD and custom tracking lock-on overlay */}
                    {isCameraActive && (
                      <div className="absolute inset-0 pointer-events-none">
                        <div
                          className="absolute z-10 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center transition-all duration-100"
                          style={{
                            left: `${(rtTargetPoint ? rtTargetPoint.xPct : 0.5) * 100}%`,
                            top: `${(rtTargetPoint ? rtTargetPoint.yPct : 0.5) * 100}%`
                          }}
                        >
                          {/* Crosshair */}
                          <div className="relative w-10 h-10 flex items-center justify-center">
                            <div className="absolute top-0 left-0 w-2.5 h-2.5 border-t-2 border-l-2 border-red-500" />
                            <div className="absolute top-0 right-0 w-2.5 h-2.5 border-t-2 border-r-2 border-red-500" />
                            <div className="absolute bottom-0 left-0 w-2.5 h-2.5 border-b-2 border-l-2 border-red-500" />
                            <div className="absolute bottom-0 right-0 w-2.5 h-2.5 border-b-2 border-r-2 border-red-500" />
                            <div className="w-5 h-5 border border-red-500/40 rounded-full flex items-center justify-center">
                              <div className="w-1.5 h-1.5 bg-red-500 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
                            </div>
                          </div>

                          {/* Laser Floating Tag */}
                            {rawDepth !== null && (
                              <div className="mt-2 bg-slate-950/90 border border-red-500/50 px-2 py-0.5 rounded shadow-xl backdrop-blur text-[10px] font-mono text-red-400 font-bold tracking-tight">
                                {rtMetricDistanceEnabled && smoothedDistance !== null ? `${smoothedDistance.toFixed(2)}m` : `Depth: ${Math.round((rawDepth / 255) * 100)}%`}
                              </div>
                            )}
                        </div>

                        {/* Helper tip overlay on hover */}
                        <div className="absolute top-4 right-4 bg-black/60 backdrop-blur text-slate-400 text-[8px] px-2 py-1 rounded border border-slate-800 opacity-0 group-hover:opacity-100 transition-opacity">
                          点击画面设置测距点
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Real-time Depth Feed Container */}
              {rtMetricDepthEnabled && (
                <div 
                  className="flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden relative min-h-[300px] cursor-crosshair group"
                  onClick={handleRtFeedClick}
                >
                  <div className="absolute top-4 left-4 bg-blue-600/80 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold tracking-tighter z-20 text-white">REAL-TIME DEPTH</div>
                  <div className="flex-1 flex items-center justify-center bg-black h-full relative">
                    {realtimeDepthUrl ? (
                      <>
                        <img src={realtimeDepthUrl} alt="Realtime Depth" className="w-full h-full object-cover pointer-events-none" />
                        
                        {isCameraActive && (
                          <div className="absolute inset-0 pointer-events-none">
                            {/* Match lock-on target bubble */}
                            <div
                              className="absolute -translate-x-1/2 -translate-y-1/2 z-10 transition-all duration-100"
                              style={{
                                left: `${(rtTargetPoint ? rtTargetPoint.xPct : 0.5) * 100}%`,
                                top: `${(rtTargetPoint ? rtTargetPoint.yPct : 0.5) * 100}%`
                              }}
                            >
                              <div className="w-8 h-8 border-2 border-red-500/70 rounded-full flex items-center justify-center">
                                <div className="w-1.5 h-1.5 rounded-full bg-red-500" />
                              </div>
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="flex flex-col items-center text-slate-500 gap-3">
                        {isCameraActive ? (
                          <>
                            <Loader2 size={24} className="animate-spin text-blue-500" />
                            <span className="text-sm font-medium">Processing frames...</span>
                          </>
                        ) : (
                          <span className="text-sm font-medium">Waiting for camera...</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Quick reset button for laser target */}
            {isCameraActive && rtTargetPoint && (
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setRtTargetPoint(null)}
                  className="bg-slate-900 border border-slate-700 hover:bg-slate-800 text-[10px] text-slate-400 px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-bold transition-all shadow-md active:scale-95"
                >
                  <RefreshCw size={11} className="animate-spin-slow" />
                  重置瞄准中心 (Reset Laser to Center)
                </button>
              </div>
            )}

            {/* Real-time Telemetry Dashboard & Metric3D Waveform Chart */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-900/40 border border-slate-800 p-5 rounded-2xl shadow-xl">
              {/* Distance Readout Box */}
              <div className="flex flex-col justify-center items-center p-5 bg-slate-950/50 rounded-xl border border-slate-800/60 text-center min-h-[140px] relative overflow-hidden">
                <div className="absolute top-2 left-3 bg-red-500/10 border border-red-500/20 text-red-400 text-[8px] font-mono font-black rounded px-1.5 py-0.5 tracking-wider uppercase">
                  Telemetry HUD
                </div>
                
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-2 mb-1.5">
                  {rtMetricDistanceEnabled ? "激光靶心绝对测距 (Laser Range)" : "对焦深度值 (Center Depth)"}
                </span>

                {rawDepth !== null ? (
                  <div className="flex flex-col items-center">
                    {rtMetricDistanceEnabled ? (
                      (() => {
                        const distance = smoothedDistance ?? calculateActualDistance(rawDepth, rtMetricStats);
                        let safetyLabel = "SAFE";
                        let safetyColor = "text-green-400 border-green-500/20 bg-green-500/5";
                        
                        if (metricScenario === 'indoor') {
                          if (distance < 1.0) {
                            safetyLabel = "TOO CLOSE";
                            safetyColor = "text-red-500 border-red-500/25 bg-red-500/5 animate-pulse";
                          } else if (distance < 2.5) {
                            safetyLabel = "WARNING";
                            safetyColor = "text-yellow-500 border-yellow-500/25 bg-yellow-500/5";
                          }
                        } else {
                          if (distance < 3.0) {
                            safetyLabel = "TOO CLOSE";
                            safetyColor = "text-red-500 border-red-500/25 bg-red-500/5 animate-pulse";
                          } else if (distance < 10.0) {
                            safetyLabel = "WARNING";
                            safetyColor = "text-yellow-500 border-yellow-500/25 bg-yellow-500/5";
                          }
                        }

                        return (
                          <>
                            <div className="flex items-baseline gap-1">
                              <span className="text-4xl font-extrabold text-white font-mono tracking-tighter">
                                {distance.toFixed(2)}
                              </span>
                              <span className="text-sm font-bold text-red-400">米 (m)</span>
                            </div>
                            
                            <span className={cn("text-[8px] font-black border px-2 py-0.5 rounded mt-2 tracking-widest uppercase", safetyColor)}>
                              {safetyLabel}
                            </span>
                          </>
                        );
                      })()
                    ) : (
                      <>
                        <span className="text-4xl font-extrabold text-indigo-400 font-mono tracking-tight">
                          {Math.round((rawDepth / 255) * 100)}%
                        </span>
                        <span className="text-[9px] text-slate-500 mt-1.5 leading-relaxed">
                          相对邻近度: <strong className="text-slate-300">{(rawDepth / 255) > 0.7 ? '极近 (Near)' : (rawDepth / 255) > 0.4 ? '中程 (Mid)' : '极远 (Far)'}</strong>
                        </span>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-slate-500">
                    <Loader2 size={16} className="animate-spin text-slate-600" />
                    <span className="text-[11px] font-medium">正在拉取并解析相机帧信号...</span>
                  </div>
                )}
              </div>

              {/* Ranging History Graphic / Bar Chart */}
              <div className="flex flex-col justify-between p-5 bg-slate-950/50 rounded-xl border border-slate-800/60 min-h-[140px] relative overflow-hidden">
                {rtMetricDistanceEnabled ? (
                  <div className="flex-1 flex flex-col justify-between h-full">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">激光测距动态波动 (Rolling Waveform)</span>
                      <span className="text-[8px] font-mono text-red-500 bg-red-500/5 px-1 rounded">M meters</span>
                    </div>

                    {liveDistanceHistory.length > 0 ? (
                      <div className="flex-1 flex items-end gap-[1.5px] h-16 pt-3 border-b border-slate-800/50">
                        {liveDistanceHistory.map((dist, idx) => {
                          const maxDist = rtMetricStats?.max_m || modelRange?.max || (metricScenario === 'indoor' ? 6.0 : 50.0);
                          const heightPct = Math.max(4, Math.min(100, (dist / maxDist) * 100));
                          return (
                            <div key={idx} className="flex-1 flex flex-col justify-end h-full group relative">
                              <div
                                className="bg-gradient-to-t from-red-600/60 to-red-500 group-hover:from-red-400 group-hover:to-red-300 rounded-t-[1px] transition-all duration-300"
                                style={{ height: `${heightPct}%` }}
                              />
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-slate-600 text-[10px] italic">
                        未捕获动态测距信号...
                      </div>
                    )}

                    <div className="flex justify-between text-[8px] text-slate-600 font-mono mt-1">
                      <span>0m (Closest)</span>
                      <span>动态雷达波形历史 (Dynamic chart)</span>
                      <span>{rtMetricStats ? `${rtMetricStats.max_m.toFixed(1)}m` : (metricScenario === 'indoor' ? '6m' : '50m')}</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col justify-between h-full">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">对焦位置灰阶值 (Disparity)</span>
                      <span className="text-[11px] font-bold font-mono text-indigo-400">{rawDepth !== null ? `${rawDepth} / 255` : '--'}</span>
                    </div>
                    <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden border border-slate-800 mt-2">
                      <div 
                        className="bg-gradient-to-r from-blue-600 via-cyan-400 to-indigo-500 h-full transition-all duration-200" 
                        style={{ width: `${rawDepth !== null ? (rawDepth / 255) * 100 : 0}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-[8px] text-slate-500 mt-2 font-mono">
                      <span>0 (无限远 / 吸收)</span>
                      <span>128 (中深度范围)</span>
                      <span>255 (最近点对焦)</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          )
        ) : !result ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col gap-6"
          >
            <div
              {...getRootProps()}
              className={cn(
                "relative group cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center p-12 text-center bg-slate-900",
                isDragActive ? "border-blue-500 bg-blue-500/5" : "border-slate-800 hover:border-slate-700 hover:bg-slate-800/50",
                preview ? "aspect-auto min-h-[400px] border-none bg-slate-900" : "aspect-video"
              )}
            >
              <input {...getInputProps()} />
              
              {preview ? (
                <div className="relative w-full h-full min-h-[400px] flex items-center justify-center p-4">
                  <div 
                    className="relative max-w-full max-h-[500px] md:max-h-[600px] w-full h-full flex items-center justify-center overflow-hidden rounded-xl transition-all duration-300"
                    style={{ 
                      aspectRatio: imageAspect ? `${imageAspect}` : 'auto',
                      height: imageAspect ? 'auto' : '100%'
                    }}
                  >
                    <img 
                      src={preview} 
                      alt="Preview" 
                      onLoad={handleImageLoad}
                      className="w-full h-full object-contain rounded-xl" 
                    />
                    
                    {/* Crop visual overlays for photo uploads */}
                    <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-xl">
                      <div className="w-full h-full relative">
                        {cropTop > 0 && (
                          <div 
                            className="absolute top-0 left-0 right-0 bg-black/60 border-b border-red-500/30 flex items-center justify-center text-[9px] text-red-400 font-mono tracking-wider z-20"
                            style={{ height: `${cropTop}%` }}
                          >
                            <span className="scale-90 font-bold opacity-80">CROPPED (去水印)</span>
                          </div>
                        )}
                        {cropBottom > 0 && (
                          <div 
                            className="absolute bottom-0 left-0 right-0 bg-black/60 border-t border-red-500/30 flex items-center justify-center text-[9px] text-red-400 font-mono tracking-wider z-20"
                            style={{ height: `${cropBottom}%` }}
                          >
                            <span className="scale-90 font-bold opacity-80">CROPPED (去水印)</span>
                          </div>
                        )}
                        {cropLeft > 0 && (
                          <div 
                            className="absolute top-0 bottom-0 left-0 bg-black/60 border-r border-red-500/30 z-20"
                            style={{ width: `${cropLeft}%`, top: `${cropTop}%`, bottom: `${cropBottom}%` }}
                          />
                        )}
                        {cropRight > 0 && (
                          <div 
                            className="absolute top-0 bottom-0 right-0 bg-black/60 border-l border-red-500/30 z-20"
                            style={{ width: `${cropRight}%`, top: `${cropTop}%`, bottom: `${cropBottom}%` }}
                          />
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm z-30 rounded-2xl">
                    <p className="text-white font-medium flex items-center gap-2">
                      <UploadCloud size={20} />
                      Choose another photo
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4 text-gray-400 group-hover:text-gray-300 transition-colors">
                  <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                    <ImageIcon size={32} />
                  </div>
                  <div>
                    <p className="text-lg font-bold text-slate-200 mb-1 tracking-tight">Upload a photo</p>
                    <p className="text-sm font-medium">Drag and drop, or click to browse</p>
                  </div>
                </div>
              )}
            </div>
            
            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm whitespace-pre-wrap">
                {error.includes('ZeroGPU quota') ? (
                  <>
                    <p className="font-bold mb-1">Rate Limit Exceeded</p>
                    <p>The public Hugging Face model is currently busy or you have reached the IP rate limit. Please try again later.</p>
                    <p className="mt-2 text-xs opacity-80">Tip: To increase your limit, configure a Hugging Face token in the AI Studio Secrets panel as HF_TOKEN.</p>
                  </>
                ) : (
                  error
                )}
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col gap-8"
          >
            {result.warning && (
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-xl text-yellow-400 text-sm">
                <strong>Notice:</strong> {result.warning}
              </div>
            )}
            {result.metricStats && (
              <div className="p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-xl text-indigo-300 text-sm">
                <strong>Depth Stats:</strong> min {result.metricStats.min_m.toFixed(2)}m / max {result.metricStats.max_m.toFixed(2)}m / mean {result.metricStats.mean_m.toFixed(2)}m
                {modelRange && (
                  <span className="text-indigo-400/70 ml-2">| 模型经验范围: {modelRange.label}</span>
                )}
              </div>
            )}
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-medium">Comparison</h2>
                <div className="flex gap-3 items-center">
                  <button
                    onClick={() => {
                      setResult(null);
                      setPreview(null);
                      setFile(null);
                      setStaticPins([]);
                    }}
                    className="text-sm font-bold text-slate-500 hover:text-slate-300 transition-colors px-4 py-2 uppercase tracking-wider"
                  >
                    Start Over
                  </button>
                  {staticPins.length > 0 && (
                    <button
                      onClick={() => setStaticPins([])}
                      className="text-xs font-bold text-red-400 hover:text-red-300 border border-red-500/20 bg-red-500/5 px-3 py-1.5 rounded-md transition-all active:scale-95 uppercase tracking-wider"
                    >
                      Clear Pins ({staticPins.length})
                    </button>
                  )}
                  {metricDepthEnabled && metricDistanceEnabled && (
                    <button
                      onClick={generatePointCloud}
                      disabled={isGeneratingPointCloud}
                      className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-2 rounded-md text-sm font-semibold transition-all shadow-lg shadow-emerald-600/20 flex items-center gap-2"
                    >
                      {isGeneratingPointCloud ? <Loader2 size={16} className="animate-spin" /> : <Box size={16} />}
                      {isGeneratingPointCloud ? 'Generating Cloud...' : '3D Point Cloud'}
                    </button>
                  )}
                  <a
                    href={result.colored}
                    download="depth-map.png"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-md text-sm font-semibold transition-all shadow-lg shadow-indigo-600/20 flex items-center gap-2"
                  >
                    <Download size={16} />
                    Download Map
                  </a>
                </div>
              </div>
              
              <ImageSlider 
                originalUrl={croppedPreview || preview!} 
                depthUrl={result.colored}
                pins={measureMode === 'two' ? twoPointPins.map(p => ({ ...p, distance: 0, raw: 0 })) : staticPins}
                isRangingActive={measureMode === 'two' || metricDistanceEnabled}
                onContainerClick={handleStaticContainerClick}
                onDeletePin={handleDeleteStaticPin}
                metricDepthEnabled={metricDepthEnabled || metricDistanceEnabled}
                showDepthOverlay={metricDepthEnabled}
                hidePinTooltips={measureMode === 'two'}
                connectionLine={measureMode === 'two' && twoPointPins.length === 2 ? {
                  x1: twoPointPins[0].x,
                  y1: twoPointPins[0].y,
                  x2: twoPointPins[1].x,
                  y2: twoPointPins[1].y,
                  label: referenceDistanceM != null ? `${referenceDistanceM}m ref` : 'ref'
                } : null}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden relative">
                <div className="absolute top-4 left-4 bg-black/50 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold tracking-tighter z-10">COLORED DEPTH</div>
                <div className="flex-1 flex items-center justify-center bg-black">
                  <img src={result.colored} alt="Colored Depth Map" className="w-full h-auto object-cover" />
                </div>
                <div className="h-10 bg-slate-900 px-4 flex items-center border-t border-slate-800">
                  <span className="text-[10px] text-slate-500 font-mono font-bold tracking-widest uppercase">DepthViz Render — {result.modelUsed || 'Depth Anything V2 Basic'}</span>
                </div>
              </div>
              <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden relative">
                <div className="absolute top-4 left-4 bg-blue-600/80 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold tracking-tighter z-10 text-white">GRAYSCALE (RAW)</div>
                <div className="flex-1 flex items-center justify-center bg-black">
                  <img src={result.grayscale} alt="Grayscale Depth Map" className="w-full h-auto object-cover" />
                </div>
                <div className="h-10 bg-slate-900 px-4 flex items-center border-t border-slate-800">
                  <span className="text-[10px] text-slate-500 font-mono font-bold tracking-widest uppercase">Raw Disparity Map</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
        </div>
      </main>

      {/* Footer Status Bar */}
      <footer className="h-8 bg-blue-600 text-[10px] flex items-center px-6 font-mono font-medium justify-between shrink-0">
        <div className="flex gap-6 uppercase tracking-wider text-blue-50">
          <span className="hidden sm:inline">GPU: {result?.modelUsed?.includes('Local') ? 'LOCAL INFERENCE' : 'CLOUD INFERENCE'}</span>
          <span>LATENCY: {isGenerating ? 'MEASURING...' : 'OPTIMIZED'}</span>
          <span className="hidden sm:inline">TEMP: NORMAL</span>
        </div>
        <div className="flex gap-4 uppercase text-white font-bold tracking-widest">
          <span className="hidden sm:inline">Session ID: DA-9921-XPR</span>
          <span>{isGenerating ? 'PROCESSING...' : 'READY'}</span>
        </div>
      </footer>
    </div>
  );
}
