type WorkerConfig = {
  threshold: number;
  bundle: { bundle_dir: string };
  input_size: number;
  shape: number[];
  dtype: string;
  mean: number[];
  std: number[];
  clip_abs: number;
  crop_pct: number;
  resize_shorter_side: number;
  allowed_mime_types: string[];
  max_file_size_bytes: number;
  max_image_dimension: number;
};

type WorkerInput = {
  file: File;
  config: WorkerConfig;
};

type QualitySummary = {
  mean_luma: number;
  std_luma: number;
  overexposed_ratio: number;
  underexposed_ratio: number;
  effective_luma_ratio: number;
  dynamic_range_p95_p05: number;
  laplacian_variance: number;
};

function postProgress(status: "worker_preprocessing") {
  self.postMessage({ type: "progress", status });
}

function toHex(buffer: ArrayBuffer) {
  return Array.from(new Uint8Array(buffer))
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function sha256Hex(data: ArrayBuffer | ArrayBufferView) {
  const source =
    data instanceof ArrayBuffer
      ? new Uint8Array(data)
      : new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
  const digest = await crypto.subtle.digest("SHA-256", source);
  return toHex(digest);
}

function sniffPngDimensions(bytes: Uint8Array) {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (!signature.every((value, index) => bytes[index] === value)) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return {
    mime: "image/png",
    width: view.getUint32(16, false),
    height: view.getUint32(20, false)
  };
}

function sniffJpegDimensions(bytes: Uint8Array) {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  let offset = 2;
  while (offset + 9 < bytes.length) {
    if (bytes[offset] !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = bytes[offset + 1];
    if (marker === 0xda || marker === 0xd9) break;
    const length = (bytes[offset + 2] << 8) | bytes[offset + 3];
    if (length < 2 || offset + 1 + length >= bytes.length) break;
    const sofMarkers = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
    if (sofMarkers.has(marker)) {
      return {
        mime: "image/jpeg",
        height: (bytes[offset + 5] << 8) | bytes[offset + 6],
        width: (bytes[offset + 7] << 8) | bytes[offset + 8]
      };
    }
    offset += 2 + length;
  }
  return { mime: "image/jpeg", width: 0, height: 0 };
}

function sniffFile(bytes: Uint8Array) {
  return sniffPngDimensions(bytes) ?? sniffJpegDimensions(bytes);
}

function percentile(values: Float32Array, q: number) {
  const copy = Array.from(values).sort((left, right) => left - right);
  const position = (copy.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return copy[lower];
  const weight = position - lower;
  return copy[lower] * (1 - weight) + copy[upper] * weight;
}

function computeQualitySummary(rgb: Float32Array, inputSize: number): QualitySummary {
  const pixelCount = inputSize * inputSize;
  const luma = new Float32Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    const red = rgb[index];
    const green = rgb[pixelCount + index];
    const blue = rgb[pixelCount * 2 + index];
    luma[index] = 0.299 * red + 0.587 * green + 0.114 * blue;
  }
  let mean = 0;
  for (const value of luma) mean += value;
  mean /= luma.length;
  let variance = 0;
  let over = 0;
  let under = 0;
  let effective = 0;
  for (const value of luma) {
    variance += (value - mean) ** 2;
    if (value >= 0.95) over += 1;
    if (value <= 0.05) under += 1;
    if (value >= 0.02 && value <= 0.98) effective += 1;
  }
  variance /= luma.length;
  let lapVariance = 0;
  let lapCount = 0;
  for (let y = 1; y < inputSize - 1; y += 1) {
    for (let x = 1; x < inputSize - 1; x += 1) {
      const center = y * inputSize + x;
      const lap =
        -4 * luma[center] +
        luma[center - inputSize] +
        luma[center + inputSize] +
        luma[center - 1] +
        luma[center + 1];
      lapVariance += lap * lap;
      lapCount += 1;
    }
  }
  return {
    mean_luma: Number(mean.toFixed(8)),
    std_luma: Number(Math.sqrt(variance).toFixed(8)),
    overexposed_ratio: Number((over / luma.length).toFixed(8)),
    underexposed_ratio: Number((under / luma.length).toFixed(8)),
    effective_luma_ratio: Number((effective / luma.length).toFixed(8)),
    dynamic_range_p95_p05: Number((percentile(luma, 0.95) - percentile(luma, 0.05)).toFixed(8)),
    laplacian_variance: Number((lapVariance / Math.max(lapCount, 1)).toFixed(8))
  };
}

function packFloat32LE(values: Float32Array) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      throw new Error(`Non-finite float at ${index}`);
    }
    view.setFloat32(index * 4, Math.fround(value), true);
  }
  return bytes;
}

function randomShareValue() {
  const buffer = new Uint32Array(1);
  crypto.getRandomValues(buffer);
  return (buffer[0] / 0xffffffff - 0.5) * 1.0;
}

function roundHalfToEven(value: number) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

async function runWorker(input: WorkerInput) {
  postProgress("worker_preprocessing");
  const decodeStarted = performance.now();
  const rawBuffer = await input.file.arrayBuffer();
  const rawBytes = new Uint8Array(rawBuffer);
  const sniffed = sniffFile(rawBytes);
  if (!sniffed) {
    throw new Error("非法图片头或不支持的格式");
  }
  if (!input.config.allowed_mime_types.includes(sniffed.mime)) {
    throw new Error(`当前只允许 ${input.config.allowed_mime_types.join(" / ")}`);
  }
  if (input.file.size > input.config.max_file_size_bytes) {
    throw new Error(`文件超过 ${Math.round(input.config.max_file_size_bytes / 1024 / 1024)} MiB 限制`);
  }
  if (
    sniffed.width <= 0 ||
    sniffed.height <= 0 ||
    sniffed.width > input.config.max_image_dimension ||
    sniffed.height > input.config.max_image_dimension
  ) {
    throw new Error("图片尺寸非法或过大");
  }

  const bitmap = await createImageBitmap(input.file);
  const decodeMs = performance.now() - decodeStarted;
  const preprocessStarted = performance.now();
  const shorterSide = input.config.resize_shorter_side;
  const resizedWidth = bitmap.width <= bitmap.height
    ? shorterSide
    : Math.floor((shorterSide * bitmap.width) / bitmap.height);
  const resizedHeight = bitmap.width <= bitmap.height
    ? Math.floor((shorterSide * bitmap.height) / bitmap.width)
    : shorterSide;
  const resizedCanvas = new OffscreenCanvas(resizedWidth, resizedHeight);
  const resizedContext = resizedCanvas.getContext("2d");
  if (!resizedContext) throw new Error("无法创建缩放画布");
  resizedContext.imageSmoothingEnabled = true;
  resizedContext.imageSmoothingQuality = "high";
  resizedContext.drawImage(bitmap, 0, 0, resizedWidth, resizedHeight);

  const canvas = new OffscreenCanvas(input.config.input_size, input.config.input_size);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("无法创建离屏画布");
  // torchvision CenterCrop uses Python round(), including ties-to-even.
  const sourceX = roundHalfToEven((resizedWidth - input.config.input_size) / 2);
  const sourceY = roundHalfToEven((resizedHeight - input.config.input_size) / 2);
  context.drawImage(
    resizedCanvas,
    sourceX,
    sourceY,
    input.config.input_size,
    input.config.input_size,
    0,
    0,
    input.config.input_size,
    input.config.input_size
  );
  const imageData = context.getImageData(0, 0, input.config.input_size, input.config.input_size);
  const pixelCount = input.config.input_size * input.config.input_size;
  const rgb = new Float32Array(pixelCount * 3);
  for (let index = 0; index < pixelCount; index += 1) {
    rgb[index] = imageData.data[index * 4] / 255;
    rgb[pixelCount + index] = imageData.data[index * 4 + 1] / 255;
    rgb[pixelCount * 2 + index] = imageData.data[index * 4 + 2] / 255;
  }
  const preprocessMs = performance.now() - preprocessStarted;

  const dqaStarted = performance.now();
  const qualitySummary = computeQualitySummary(rgb, input.config.input_size);
  const dqaMs = performance.now() - dqaStarted;

  const normalizeStarted = performance.now();
  const normalized = new Float32Array(rgb.length);
  for (let channel = 0; channel < 3; channel += 1) {
    const mean = input.config.mean[channel];
    const std = input.config.std[channel];
    const channelOffset = channel * pixelCount;
    for (let index = 0; index < pixelCount; index += 1) {
      const normalizedValue = (rgb[channelOffset + index] - mean) / std;
      normalized[channelOffset + index] = input.config.clip_abs > 0
        ? Math.max(-input.config.clip_abs, Math.min(input.config.clip_abs, normalizedValue))
        : normalizedValue;
    }
  }

  const hashStarted = performance.now();
  const normalizedBytes = packFloat32LE(normalized);
  const sourceImageSha256 = await sha256Hex(rawBuffer);
  const normalizedTensorSha256 = await sha256Hex(normalizedBytes);
  const hashMsBase = performance.now() - hashStarted;

  const shareStarted = performance.now();
  const share0 = new Float32Array(normalized.length);
  const share1 = new Float32Array(normalized.length);
  for (let index = 0; index < normalized.length; index += 1) {
    const left = randomShareValue();
    share0[index] = left;
    share1[index] = normalized[index] - left;
  }
  const share0Bytes = packFloat32LE(share0);
  const share1Bytes = packFloat32LE(share1);
  const shareBuildMs = performance.now() - shareStarted;

  const hashResumeStarted = performance.now();
  const share0Sha256 = await sha256Hex(share0Bytes);
  const share1Sha256 = await sha256Hex(share1Bytes);
  const auditNonce = crypto.randomUUID();
  const auditChainPayload = new TextEncoder().encode(
    `medical_live_demo_v1|${auditNonce}|${sourceImageSha256}|${normalizedTensorSha256}|${share0Sha256}|${share1Sha256}`
  );
  const auditChainSha256 = await sha256Hex(auditChainPayload);
  const hashMs = hashMsBase + (performance.now() - hashResumeStarted);
  const totalMs = decodeMs + preprocessMs + dqaMs + hashMs + shareBuildMs;
  const previewUrl = canvas.convertToBlob({ type: sniffed.mime }).then((blob) => URL.createObjectURL(blob));

  bitmap.close();
  resizedCanvas.width = 1;
  resizedCanvas.height = 1;
  canvas.width = 1;
  canvas.height = 1;

  return {
    requestManifest: {
      manifest_type: "transshield_showcase_medical_live_request_v1",
      contract_version: "medical_live_demo_v1",
      bundle_dir: input.config.bundle.bundle_dir,
      input_size: input.config.input_size,
      shape: input.config.shape,
      dtype: "float32_le",
      preprocessing: {
        resize_shorter_side: input.config.resize_shorter_side,
        center_crop_size: input.config.input_size,
        crop_pct: input.config.crop_pct,
        interpolation: "browser_canvas_high_quality",
        normalization_clip_abs: input.config.clip_abs
      },
      source_file_name: input.file.name,
      source_mime: sniffed.mime,
      source_size_bytes: input.file.size,
      source_dimensions: { width: sniffed.width, height: sniffed.height },
      audit_nonce: auditNonce,
      source_image_sha256: sourceImageSha256,
      normalized_tensor_sha256: normalizedTensorSha256,
      share0_sha256: share0Sha256,
      share1_sha256: share1Sha256,
      audit_chain_sha256: auditChainSha256
    },
    qualityAssurance: {
      status: "pass",
      client_quality_summary: qualitySummary
    },
    audit: {
      hash_chain_version: "medical_live_demo_v1",
      browser_generated_shares: true,
      server_should_receive_plain_image: false,
      server_should_receive_plain_pixel_values: false,
      centralized_demo_reconstructs_normalized_tensor_for_dqa: true,
      production_target_should_not_co_locate_both_shares: true
    },
    controlPlaneMetrics: {
      decode_ms: Number(decodeMs.toFixed(3)),
      preprocess_ms: Number(preprocessMs.toFixed(3)),
      dqa_ms: Number(dqaMs.toFixed(3)),
      hash_ms: Number(hashMs.toFixed(3)),
      share_build_ms: Number(shareBuildMs.toFixed(3)),
      total_ms: Number(totalMs.toFixed(3))
    },
    share0: share0Bytes,
    share1: share1Bytes,
    previewUrl: await previewUrl
  };
}

self.onmessage = async (event: MessageEvent<WorkerInput>) => {
  try {
    const payload = await runWorker(event.data);
    self.postMessage({ type: "completed", payload });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : "worker 处理失败"
    });
  }
};
