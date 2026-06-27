import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { GripVertical, X, Info } from 'lucide-react';

interface ImageSliderProps {
  originalUrl: string;
  depthUrl: string;
  pins?: Array<{ id: number; x: number; y: number; distance: number; raw: number; depthRaw?: number; label?: string }>;
  isRangingActive?: boolean;
  onContainerClick?: (xPct: number, yPct: number) => void;
  onDeletePin?: (id: number) => void;
  metricDepthEnabled?: boolean;
  showDepthOverlay?: boolean;
  connectionLine?: { x1: number; y1: number; x2: number; y2: number; label: string } | null;
  hidePinTooltips?: boolean;
}

export function ImageSlider({
  originalUrl,
  depthUrl,
  pins = [],
  isRangingActive = false,
  onContainerClick,
  onDeletePin,
  metricDepthEnabled = false,
  showDepthOverlay = true,
  connectionLine = null,
  hidePinTooltips = false
}: ImageSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.max(0, Math.min((x / rect.width) * 100, 100));
    setSliderPosition(percent);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  };

  const handleTouchMove = (e: TouchEvent) => {
    if (!isDragging) return;
    handleMove(e.touches[0].clientX);
  };

  const handleMouseUp = () => setIsDragging(false);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging]);

  const handleContainerClickOrTouch = (clientX: number, clientY: number) => {
    if (!containerRef.current || !onContainerClick) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min((clientX - rect.left) / rect.width, 1));
    const y = Math.max(0, Math.min((clientY - rect.top) / rect.height, 1));
    onContainerClick(x, y);
  };

  return (
    <div
      ref={containerRef}
      className={`relative w-full aspect-video rounded-2xl overflow-hidden select-none bg-black border border-slate-800 shadow-2xl ${
        isRangingActive ? 'cursor-crosshair' : ''
      }`}
      onMouseDown={(e) => {
        if (isRangingActive) {
          handleContainerClickOrTouch(e.clientX, e.clientY);
          return;
        }
        setIsDragging(true);
        handleMove(e.clientX);
      }}
      onTouchStart={(e) => {
        if (isRangingActive) {
          handleContainerClickOrTouch(e.touches[0].clientX, e.touches[0].clientY);
          return;
        }
        setIsDragging(true);
        handleMove(e.touches[0].clientX);
      }}
    >
      <img
        src={originalUrl}
        alt="Original"
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
      />
      {metricDepthEnabled && showDepthOverlay && (
        <div
          className="absolute inset-0 w-full h-full object-contain pointer-events-none overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        >
          <img
            src={depthUrl}
            alt="Depth Map"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />
        </div>
      )}
      
      {/* Slider Handle */}
      {metricDepthEnabled && showDepthOverlay && (
        <div
          className="absolute top-0 bottom-0 w-1 bg-indigo-500 cursor-ew-resize hover:bg-indigo-400 transition-colors shadow-[0_0_10px_rgba(99,102,241,0.5)] z-20"
          style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
          onMouseDown={(e) => {
            e.stopPropagation();
            setIsDragging(true);
          }}
          onTouchStart={(e) => {
            e.stopPropagation();
            setIsDragging(true);
          }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-indigo-500 text-white rounded-full flex items-center justify-center shadow-lg cursor-ew-resize">
            <GripVertical size={16} />
          </div>
        </div>
      )}

      {/* Connection line between two measurement points */}
      {connectionLine && (
        <div
          className="absolute pointer-events-none z-25"
          style={{
            left: `${Math.min(connectionLine.x1, connectionLine.x2)}%`,
            top: `${Math.min(connectionLine.y1, connectionLine.y2)}%`,
            width: `${Math.abs(connectionLine.x2 - connectionLine.x1)}%`,
            height: `${Math.abs(connectionLine.y2 - connectionLine.y1)}%`,
          }}
        >
          <svg className="w-full h-full overflow-visible">
            <line
              x1={connectionLine.x1 <= connectionLine.x2 ? '0%' : '100%'}
              y1={connectionLine.y1 <= connectionLine.y2 ? '0%' : '100%'}
              x2={connectionLine.x1 <= connectionLine.x2 ? '100%' : '0%'}
              y2={connectionLine.y1 <= connectionLine.y2 ? '100%' : '0%'}
              stroke="rgba(239,68,68,0.8)"
              strokeWidth="2"
              strokeDasharray="4 2"
            />
          </svg>
          <div
            className="absolute bg-red-600/90 text-white text-[10px] font-bold px-2 py-0.5 rounded-full border border-red-400/50 whitespace-nowrap"
            style={{
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          >
            {connectionLine.label}
          </div>
        </div>
      )}

      {/* Render measurement pins */}
      {isRangingActive && pins.map((pin) => (
        <div
          key={pin.id}
          className="absolute z-30 -translate-x-1/2 -translate-y-1/2 group/pin"
          style={{ left: `${pin.x}%`, top: `${pin.y}%` }}
          onMouseDown={(e) => e.stopPropagation()} // Stop propagation so clicking the dot doesn't place a new dot
          onTouchStart={(e) => e.stopPropagation()}
        >
          {/* Glowing pulse ring */}
          <div className="absolute inset-0 -m-2.5 border-2 border-indigo-400 rounded-full animate-ping opacity-75 pointer-events-none" />
          
          {/* Marker trigger */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (onDeletePin) onDeletePin(pin.id);
            }}
            className="w-4.5 h-4.5 rounded-full bg-indigo-600 border border-white flex items-center justify-center text-white shadow-lg hover:bg-red-600 hover:border-red-300 transition-colors cursor-pointer"
            title="点击删除此测距点 (Click to delete)"
          >
            <span className="text-[8px] font-black font-sans">✕</span>
          </button>
          
          {!hidePinTooltips && (
            <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 bg-slate-950/95 border border-indigo-500/80 px-2.5 py-1.5 rounded-lg shadow-2xl flex flex-col items-center gap-0.5 whitespace-nowrap pointer-events-none">
              <span className="text-[11px] font-black text-indigo-400 font-mono tracking-tight">{pin.label || `${pin.distance.toFixed(2)}m`}</span>
              {pin.depthRaw !== undefined ? (
                <span className="text-[8px] text-amber-400 font-mono">DA2: {Math.round((pin.depthRaw / 255) * 100)}% | 测距: {Math.round((pin.raw / 255) * 100)}%</span>
              ) : (
                <span className="text-[8px] text-slate-400 font-mono">深度: {Math.round((pin.raw / 255) * 100)}%</span>
              )}
            </div>
          )}
        </div>
      ))}

      {/* Display helper overlay if ranging is active but no pins are placed */}
      {isRangingActive && pins.length === 0 && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-indigo-600/90 backdrop-blur text-white font-semibold text-[10px] px-3 py-1.5 rounded-full border border-indigo-400/30 tracking-wider shadow-lg flex items-center gap-1.5 pointer-events-none z-10 animate-pulse">
          <span className="w-1.5 h-1.5 bg-white rounded-full" />
          <span>点击画面任意位置添加测距点 (Click anywhere to measure distance)</span>
        </div>
      )}
    </div>
  );
}
