const E2E_INPUT_SIZE = 224;
const E2E_RESIZE_SIZE = 256;
const E2E_CHANNELS = 3;
const E2E_FLOAT_COUNT = E2E_CHANNELS * E2E_INPUT_SIZE * E2E_INPUT_SIZE;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const SAFE_NORMALIZED_MIN = -2.0;
const SAFE_NORMALIZED_MAX = 2.0;
const MAX_DIMENSION = 8192;
const MAX_PIXELS = 40_000_000;

function parsePngDimensions(view) {
  if (view.byteLength < 24) throw new Error('PNG header truncated');
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let index = 0; index < signature.length; index += 1) {
    if (view.getUint8(index) !== signature[index]) throw new Error('Invalid PNG signature');
  }
  const chunkType = String.fromCharCode(
    view.getUint8(12),
    view.getUint8(13),
    view.getUint8(14),
    view.getUint8(15),
  );
  if (chunkType !== 'IHDR') throw new Error('PNG missing IHDR');
  return { width: view.getUint32(16, false), height: view.getUint32(20, false), format: 'png' };
}

function parseBmpDimensions(view) {
  if (view.byteLength < 26) throw new Error('BMP header truncated');
  if (String.fromCharCode(view.getUint8(0), view.getUint8(1)) !== 'BM') throw new Error('Invalid BMP signature');
  const dibSize = view.getUint32(14, true);
  if (dibSize < 12) throw new Error('Unsupported BMP DIB header');
  if (dibSize === 12) {
    return { width: view.getUint16(18, true), height: view.getUint16(20, true), format: 'bmp' };
  }
  if (view.byteLength < 26) throw new Error('BMP DIB header truncated');
  return { width: Math.abs(view.getInt32(18, true)), height: Math.abs(view.getInt32(22, true)), format: 'bmp' };
}

function parseWebpDimensions(view) {
  if (view.byteLength < 30) throw new Error('WebP header truncated');
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const webp = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (riff !== 'RIFF' || webp !== 'WEBP') throw new Error('Invalid WebP signature');
  let offset = 12;
  let chunks = 0;
  while (offset + 8 <= view.byteLength && chunks < 256) {
    const chunkType = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);
    const dataOffset = offset + 8;
    if (chunkSize < 0 || dataOffset + chunkSize > view.byteLength + 1) {
      throw new Error('Invalid WebP chunk size');
    }
    if (chunkType === 'VP8X') {
      if (chunkSize < 10 || dataOffset + 10 > view.byteLength) throw new Error('VP8X chunk truncated');
      const width = 1 + view.getUint8(dataOffset + 4) + (view.getUint8(dataOffset + 5) << 8) + (view.getUint8(dataOffset + 6) << 16);
      const height = 1 + view.getUint8(dataOffset + 7) + (view.getUint8(dataOffset + 8) << 8) + (view.getUint8(dataOffset + 9) << 16);
      return { width, height, format: 'webp' };
    }
    if (chunkType === 'VP8 ') {
      if (chunkSize < 10 || dataOffset + 10 > view.byteLength) throw new Error('VP8 chunk truncated');
      if (view.getUint8(dataOffset + 3) !== 0x9d || view.getUint8(dataOffset + 4) !== 0x01 || view.getUint8(dataOffset + 5) !== 0x2a) {
        throw new Error('VP8 frame header invalid');
      }
      return { width: view.getUint16(dataOffset + 6, true) & 0x3fff, height: view.getUint16(dataOffset + 8, true) & 0x3fff, format: 'webp' };
    }
    if (chunkType === 'VP8L') {
      if (chunkSize < 5 || dataOffset + 5 > view.byteLength) throw new Error('VP8L chunk truncated');
      if (view.getUint8(dataOffset) !== 0x2f) throw new Error('VP8L signature invalid');
      const bits =
        view.getUint8(dataOffset + 1)
        | (view.getUint8(dataOffset + 2) << 8)
        | (view.getUint8(dataOffset + 3) << 16)
        | (view.getUint8(dataOffset + 4) << 24);
      const width = 1 + (bits & 0x3fff);
      const height = 1 + ((bits >> 14) & 0x3fff);
      return { width, height, format: 'webp' };
    }
    offset = dataOffset + chunkSize + (chunkSize % 2);
    chunks += 1;
  }
  throw new Error('WebP dimensions not found');
}

function parseJpegDimensions(view) {
  if (view.byteLength < 4 || view.getUint8(0) !== 0xff || view.getUint8(1) !== 0xd8) {
    throw new Error('Invalid JPEG signature');
  }
  let offset = 2;
  let segments = 0;
  const limit = Math.min(view.byteLength, 256 * 1024);
  while (offset + 4 <= limit && segments < 1024) {
    while (offset < limit && view.getUint8(offset) !== 0xff) {
      offset += 1;
    }
    while (offset < limit && view.getUint8(offset) === 0xff) {
      offset += 1;
    }
    if (offset >= limit) break;
    const marker = view.getUint8(offset);
    offset += 1;
    if (marker === 0xd9 || marker === 0xda) break;
    if (marker >= 0xd0 && marker <= 0xd7) {
      segments += 1;
      continue;
    }
    if (offset + 2 > limit) throw new Error('JPEG segment truncated');
    const segmentLength = view.getUint16(offset, false);
    if (segmentLength < 2) throw new Error('JPEG segment length invalid');
    const segmentStart = offset + 2;
    const segmentEnd = segmentStart + segmentLength - 2;
    if (segmentEnd > view.byteLength) throw new Error('JPEG segment exceeds file length');
    if ((marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) || (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)) {
      if (segmentLength < 7) throw new Error('JPEG SOF segment truncated');
      return {
        width: view.getUint16(segmentStart + 3, false),
        height: view.getUint16(segmentStart + 1, false),
        format: 'jpeg',
      };
    }
    offset = segmentEnd;
    segments += 1;
  }
  throw new Error('JPEG dimensions not found');
}

function sniffImageDimensions(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 12) throw new Error('Image header truncated');
  if (view.getUint8(0) === 0x89 && view.getUint8(1) === 0x50) return parsePngDimensions(view);
  if (view.getUint8(0) === 0xff && view.getUint8(1) === 0xd8) return parseJpegDimensions(view);
  if (String.fromCharCode(view.getUint8(0), view.getUint8(1)) === 'BM') return parseBmpDimensions(view);
  if (String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)) === 'RIFF') return parseWebpDimensions(view);
  throw new Error('Unsupported image header');
}

function packFloat32LE(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      throw new Error(`Non-finite float detected at index ${index}`);
    }
    view.setFloat32(index * 4, Math.fround(value), true);
  }
  return bytes;
}

function clampFiniteNormalized(value) {
  if (!Number.isFinite(value)) {
    throw new Error('Normalized pixel became non-finite before serialization.');
  }
  return Math.fround(Math.max(SAFE_NORMALIZED_MIN, Math.min(SAFE_NORMALIZED_MAX, value)));
}

function ensureFiniteFloat32(value, context) {
  if (!Number.isFinite(value)) {
    throw new Error(`${context} became non-finite before serialization.`);
  }
  return Math.fround(value);
}

function fillRandomUniformMinus2To2(target) {
  const chunkSize = 16384;
  const random = new Uint32Array(Math.min(chunkSize, target.length));
  let offset = 0;
  while (offset < target.length) {
    const count = Math.min(chunkSize, target.length - offset);
    const view = count === random.length ? random : new Uint32Array(count);
    crypto.getRandomValues(view);
    for (let index = 0; index < count; index += 1) {
      target[offset + index] = Math.fround((view[index] / 4294967296) * 4 - 2);
    }
    offset += count;
  }
}

async function digestSha256Hex(bufferLike) {
  const digest = await crypto.subtle.digest('SHA-256', bufferLike);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join('');
}

function buildAuditNonce() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

function computeQualitySummary(rgbTensor) {
  const pixelCount = E2E_INPUT_SIZE * E2E_INPUT_SIZE;
  const planeSize = pixelCount;
  const luma = new Float32Array(pixelCount);
  let mean = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const value = Math.fround(
      0.299 * rgbTensor[index]
      + 0.587 * rgbTensor[planeSize + index]
      + 0.114 * rgbTensor[planeSize * 2 + index],
    );
    luma[index] = value;
    mean += value;
  }
  mean /= pixelCount;
  let variance = 0;
  let over = 0;
  let under = 0;
  let effective = 0;
  const sorted = Array.from(luma).sort((left, right) => left - right);
  for (let index = 0; index < pixelCount; index += 1) {
    const value = luma[index];
    variance += (value - mean) * (value - mean);
    if (value >= 0.95) over += 1;
    if (value <= 0.05) under += 1;
    if (value >= 0.02 && value <= 0.98) effective += 1;
  }
  const p05 = sorted[Math.floor((pixelCount - 1) * 0.05)];
  const p95 = sorted[Math.floor((pixelCount - 1) * 0.95)];
  let lapMean = 0;
  let lapVar = 0;
  const lapCount = (E2E_INPUT_SIZE - 2) * (E2E_INPUT_SIZE - 2);
  const lapSamples = new Float32Array(lapCount);
  let lapIndex = 0;
  for (let y = 1; y < E2E_INPUT_SIZE - 1; y += 1) {
    for (let x = 1; x < E2E_INPUT_SIZE - 1; x += 1) {
      const center = y * E2E_INPUT_SIZE + x;
      const lap = -4 * luma[center] + luma[center - 1] + luma[center + 1] + luma[center - E2E_INPUT_SIZE] + luma[center + E2E_INPUT_SIZE];
      lapSamples[lapIndex] = lap;
      lapMean += lap;
      lapIndex += 1;
    }
  }
  lapMean /= lapCount;
  for (let index = 0; index < lapSamples.length; index += 1) {
    lapVar += (lapSamples[index] - lapMean) * (lapSamples[index] - lapMean);
  }
  return {
    mean_luma: Number(mean.toFixed(8)),
    std_luma: Number(Math.sqrt(variance / pixelCount).toFixed(8)),
    overexposed_ratio: Number((over / pixelCount).toFixed(8)),
    underexposed_ratio: Number((under / pixelCount).toFixed(8)),
    effective_luma_ratio: Number((effective / pixelCount).toFixed(8)),
    dynamic_range_p95_p05: Number((p95 - p05).toFixed(8)),
    laplacian_variance: Number((lapVar / lapCount).toFixed(8)),
  };
}

async function handleMedicalRun(message) {
  const timings = {};
  let bitmap = null;
  let resizeCanvas = null;
  let cropCanvas = null;
  const totalStart = performance.now();
  try {
    const headerStart = performance.now();
    const dimensions = sniffImageDimensions(message.fileBuffer);
    timings.header_sniff_ms = performance.now() - headerStart;
    if (Math.max(dimensions.width, dimensions.height) > MAX_DIMENSION || dimensions.width * dimensions.height > MAX_PIXELS) {
      self.postMessage({ ok: false, error_code: 'image_dimensions_exceeded', message: `Image dimensions ${dimensions.width}x${dimensions.height} exceed guard rails.` });
      return;
    }

    const decodeStart = performance.now();
    const blob = new Blob([message.fileBuffer], { type: message.fileType || 'application/octet-stream' });
    try {
      bitmap = await createImageBitmap(blob);
    } catch (error) {
      self.postMessage({ ok: false, error_code: 'image_decode_failed', message: error?.message || 'createImageBitmap failed' });
      return;
    }
    timings.decode_ms = performance.now() - decodeStart;

    const preprocessStart = performance.now();
    const shortSide = Math.min(bitmap.width, bitmap.height);
    const scale = E2E_RESIZE_SIZE / shortSide;
    const resizedWidth = Math.round(bitmap.width * scale);
    const resizedHeight = Math.round(bitmap.height * scale);

    resizeCanvas = new OffscreenCanvas(resizedWidth, resizedHeight);
    const resizeCtx = resizeCanvas.getContext('2d', { alpha: false });
    resizeCtx.imageSmoothingEnabled = true;
    resizeCtx.imageSmoothingQuality = 'high';
    resizeCtx.drawImage(bitmap, 0, 0, resizedWidth, resizedHeight);

    cropCanvas = new OffscreenCanvas(E2E_INPUT_SIZE, E2E_INPUT_SIZE);
    const cropCtx = cropCanvas.getContext('2d', { alpha: false });
    const cropX = Math.floor((resizedWidth - E2E_INPUT_SIZE) / 2);
    const cropY = Math.floor((resizedHeight - E2E_INPUT_SIZE) / 2);
    cropCtx.drawImage(
      resizeCanvas,
      cropX,
      cropY,
      E2E_INPUT_SIZE,
      E2E_INPUT_SIZE,
      0,
      0,
      E2E_INPUT_SIZE,
      E2E_INPUT_SIZE,
    );

    const imageData = cropCtx.getImageData(0, 0, E2E_INPUT_SIZE, E2E_INPUT_SIZE);
    const rgba = imageData.data;
    const tensor = new Float32Array(E2E_FLOAT_COUNT);
    const rgbTensor = new Float32Array(E2E_FLOAT_COUNT);
    const planeSize = E2E_INPUT_SIZE * E2E_INPUT_SIZE;
    for (let y = 0; y < E2E_INPUT_SIZE; y += 1) {
      for (let x = 0; x < E2E_INPUT_SIZE; x += 1) {
        const hwIndex = y * E2E_INPUT_SIZE + x;
        const rgbaIndex = hwIndex * 4;
        const red = Math.fround(rgba[rgbaIndex] / 255);
        const green = Math.fround(rgba[rgbaIndex + 1] / 255);
        const blue = Math.fround(rgba[rgbaIndex + 2] / 255);
        rgbTensor[hwIndex] = red;
        rgbTensor[planeSize + hwIndex] = green;
        rgbTensor[planeSize * 2 + hwIndex] = blue;
        tensor[hwIndex] = clampFiniteNormalized((red - IMAGENET_MEAN[0]) / IMAGENET_STD[0]);
        tensor[planeSize + hwIndex] = clampFiniteNormalized((green - IMAGENET_MEAN[1]) / IMAGENET_STD[1]);
        tensor[planeSize * 2 + hwIndex] = clampFiniteNormalized((blue - IMAGENET_MEAN[2]) / IMAGENET_STD[2]);
      }
    }
    timings.preprocess_ms = performance.now() - preprocessStart;

    const dqaStart = performance.now();
    const clientQualitySummary = computeQualitySummary(rgbTensor);
    timings.dqa_ms = performance.now() - dqaStart;

    const hashStart = performance.now();
    const sourceImageSha256 = await digestSha256Hex(message.fileBuffer);
    const tensorBytes = packFloat32LE(tensor);
    const normalizedTensorSha256 = await digestSha256Hex(tensorBytes.buffer);
    timings.hash_pre_share_ms = performance.now() - hashStart;

    const shareStart = performance.now();
    const share0 = new Float32Array(tensor.length);
    const share1 = new Float32Array(tensor.length);
    fillRandomUniformMinus2To2(share0);
    for (let index = 0; index < tensor.length; index += 1) {
      share1[index] = ensureFiniteFloat32(tensor[index] - share0[index], 'share1');
    }
    const share0Bytes = packFloat32LE(share0);
    const share1Bytes = packFloat32LE(share1);
    timings.share_build_ms = performance.now() - shareStart;

    const finalHashStart = performance.now();
    const share0Sha256 = await digestSha256Hex(share0Bytes.buffer);
    const share1Sha256 = await digestSha256Hex(share1Bytes.buffer);
    const auditNonce = buildAuditNonce();
    const auditChainPayload = new TextEncoder().encode(
      `v7|${auditNonce}|${sourceImageSha256}|${normalizedTensorSha256}|${share0Sha256}|${share1Sha256}`,
    );
    const auditChainSha256 = await digestSha256Hex(auditChainPayload.buffer);
    timings.hash_ms = timings.hash_pre_share_ms + (performance.now() - finalHashStart);

    const clientAuditManifest = {
      audit_nonce: auditNonce,
      source_image_sha256: sourceImageSha256,
      normalized_tensor_sha256: normalizedTensorSha256,
      share0_sha256: share0Sha256,
      share1_sha256: share1Sha256,
      audit_chain_sha256: auditChainSha256,
    };
    const clientControlPlaneMetrics = {
      decode_ms: Number((timings.decode_ms || 0).toFixed(3)),
      preprocess_ms: Number((timings.preprocess_ms || 0).toFixed(3)),
      dqa_ms: Number((timings.dqa_ms || 0).toFixed(3)),
      hash_ms: Number((timings.hash_ms || 0).toFixed(3)),
      share_build_ms: Number((timings.share_build_ms || 0).toFixed(3)),
      total_ms: Number((performance.now() - totalStart).toFixed(3)),
    };

    self.postMessage({
      ok: true,
      share0Buffer: share0Bytes.buffer,
      share1Buffer: share1Bytes.buffer,
      clientQualitySummary,
      clientAuditManifest,
      clientControlPlaneMetrics,
    }, [share0Bytes.buffer, share1Bytes.buffer]);
  } catch (error) {
    self.postMessage({
      ok: false,
      error_code: 'image_decode_failed',
      message: error?.message || 'Worker failed while preparing medical control plane payload.',
    });
  } finally {
    if (bitmap && typeof bitmap.close === 'function') bitmap.close();
    if (resizeCanvas) {
      resizeCanvas.width = 0;
      resizeCanvas.height = 0;
    }
    if (cropCanvas) {
      cropCanvas.width = 0;
      cropCanvas.height = 0;
    }
  }
}

self.onmessage = async (event) => {
  const message = event.data || {};
  if (message.type !== 'medical-run') {
    self.postMessage({ ok: false, error_code: 'unsupported_worker_message', message: 'Unsupported worker message type.' });
    return;
  }
  await handleMedicalRun(message);
};
