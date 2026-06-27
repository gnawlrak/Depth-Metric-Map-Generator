import { pipeline } from '@xenova/transformers';
import sharp from 'sharp';

async function main() {
  const estimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf');
  const result = await estimator('https://picsum.photos/200/300');
  const { data, width, height } = result.depth;
  
  // Create grayscale
  const grayscaleBuffer = await sharp(data, {
    raw: { width, height, channels: 1 }
  }).png().toBuffer();

  // Create colored
  const coloredData = new Uint8Array(data.length * 3);
  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    // Simple turbo-like colormap
    const x = val / 255.0;
    coloredData[i * 3] = Math.max(0, Math.min(255, Math.floor(255 * (3.11 * x - 2.11 * x * x))));
    coloredData[i * 3 + 1] = Math.max(0, Math.min(255, Math.floor(255 * (2.0 * x - x * x))));
    coloredData[i * 3 + 2] = Math.max(0, Math.min(255, Math.floor(255 * (1.5 * x)))); // random approximation
  }

  const coloredBuffer = await sharp(coloredData, {
    raw: { width, height, channels: 3 }
  }).png().toBuffer();
  
  console.log('Grayscale:', grayscaleBuffer.length, 'Colored:', coloredBuffer.length);
}
main();
