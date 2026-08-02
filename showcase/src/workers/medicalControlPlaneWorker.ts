type WorkerConfig = {
  threshold: number;
  bundle: { bundle_dir: string };
  input_size: number;
  shape: number[];
  dtype: string;
  mean: number[];
  std: number[];
  clip_abs: number;
  allowed_mime_types: string[];
  max_file_size_bytes: number;
  max_image_dimension: number;
  pruning?: {
    patch_size: number;
    stage_layers: number[];
    stage_keep_ratios: number[];
    total_patches: number;
  };
};

type SampleCropPreset = {
  leftRatio: number;
  topRatio: number;
  sizeRatio: number;
};

type WorkerInput = {
  file: File;
  config: WorkerConfig;
  sampleCrop?: SampleCropPreset;
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

type PruningStageSummary = {
  stage_index: number;
  layer: number;
  keep_ratio: number;
  kept_patches: number;
  dropped_patches: number;
  visible_area_ratio: number;
};

type PruningPreview = {
  original_dimensions: { width: number; height: number };
  processed_dimensions: { width: number; height: number };
  patch_size: number;
  grid_size: number;
  total_patches: number;
  estimated_effective_pixels: number;
  stage_summaries: PruningStageSummary[];
  final_kept_patches: number;
  final_visible_area_ratio: number;
  processed_preview_url: string;
  pruned_preview_url: string;
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

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
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

function computePatchScores(
  rgb: Float32Array,
  inputSize: number,
  patchSize: number
) {
  const patchesPerSide = Math.floor(inputSize / patchSize);
  const pixelCount = inputSize * inputSize;
  const luma = new Float32Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    const red = rgb[index];
    const green = rgb[pixelCount + index];
    const blue = rgb[pixelCount * 2 + index];
    luma[index] = 0.299 * red + 0.587 * green + 0.114 * blue;
  }

  const scores = new Float32Array(patchesPerSide * patchesPerSide);
  for (let patchY = 0; patchY < patchesPerSide; patchY += 1) {
    for (let patchX = 0; patchX < patchesPerSide; patchX += 1) {
      let energy = 0;
      let count = 0;
      const yStart = patchY * patchSize;
      const xStart = patchX * patchSize;
      for (let y = yStart; y < yStart + patchSize; y += 1) {
        for (let x = xStart; x < xStart + patchSize; x += 1) {
          const center = y * inputSize + x;
          const value = luma[center];
          energy += value;
          if (x + 1 < inputSize) {
            energy += Math.abs(value - luma[center + 1]) * 1.4;
          }
          if (y + 1 < inputSize) {
            energy += Math.abs(value - luma[center + inputSize]) * 1.4;
          }
          count += 1;
        }
      }
      scores[patchY * patchesPerSide + patchX] = energy / Math.max(count, 1);
    }
  }
  return { scores, patchesPerSide };
}

function buildKeepMask(
  scores: Float32Array,
  keepCount: number
) {
  const total = scores.length;
  const boundedKeepCount = Math.max(1, Math.min(keepCount, total));
  const ranked = Array.from(scores, (score, index) => ({ score, index })).sort(
    (left, right) => right.score - left.score || left.index - right.index
  );
  const mask = new Array<boolean>(total).fill(false);
  for (let index = 0; index < boundedKeepCount; index += 1) {
    mask[ranked[index].index] = true;
  }
  return mask;
}

async function buildPreviewUrlFromImageData(imageData: ImageData, mime: string) {
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("无法创建预览画布");
  context.putImageData(imageData, 0, 0);
  const blob = await canvas.convertToBlob({ type: mime });
  return URL.createObjectURL(blob);
}

function buildPrunedImageData(
  source: ImageData,
  keepMask: boolean[],
  patchSize: number,
  patchesPerSide: number
) {
  const pruned = new ImageData(
    new Uint8ClampedArray(source.data),
    source.width,
    source.height
  );
  for (let patchY = 0; patchY < patchesPerSide; patchY += 1) {
    for (let patchX = 0; patchX < patchesPerSide; patchX += 1) {
      const keep = keepMask[patchY * patchesPerSide + patchX];
      for (let y = patchY * patchSize; y < (patchY + 1) * patchSize; y += 1) {
        for (let x = patchX * patchSize; x < (patchX + 1) * patchSize; x += 1) {
          const offset = (y * source.width + x) * 4;
          if (!keep) {
            pruned.data[offset] = Math.round(pruned.data[offset] * 0.08);
            pruned.data[offset + 1] = Math.round(pruned.data[offset + 1] * 0.08);
            pruned.data[offset + 2] = Math.round(pruned.data[offset + 2] * 0.08);
          }
          if (x % patchSize === 0 || y % patchSize === 0) {
            pruned.data[offset] = keep ? 26 : 148;
            pruned.data[offset + 1] = keep ? 99 : 163;
            pruned.data[offset + 2] = keep ? 235 : 184;
          }
        }
      }
    }
  }
  return pruned;
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
  const canvas = new OffscreenCanvas(input.config.input_size, input.config.input_size);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("无法创建离屏画布");

  const preprocessStarted = performance.now();
  const defaultCropSize = Math.min(bitmap.width, bitmap.height);
  const requestedCropSize = input.sampleCrop
    ? Math.round(Math.min(bitmap.width, bitmap.height) * clamp(input.sampleCrop.sizeRatio, 0.4, 1))
    : defaultCropSize;
  const cropSize = clamp(requestedCropSize, 1, Math.min(bitmap.width, bitmap.height));
  const sourceX = input.sampleCrop
    ? clamp(Math.round(bitmap.width * input.sampleCrop.leftRatio), 0, bitmap.width - cropSize)
    : Math.max(0, Math.floor((bitmap.width - cropSize) / 2));
  const sourceY = input.sampleCrop
    ? clamp(Math.round(bitmap.height * input.sampleCrop.topRatio), 0, bitmap.height - cropSize)
    : Math.max(0, Math.floor((bitmap.height - cropSize) / 2));
  context.drawImage(
    bitmap,
    sourceX,
    sourceY,
    cropSize,
    cropSize,
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
      normalized[channelOffset + index] = Math.max(-input.config.clip_abs, Math.min(input.config.clip_abs, normalizedValue));
    }
  }

  const hashStarted = performance.now();
  const normalizedBytes = packFloat32LE(normalized);
  const sourceImageSha256 = await sha256Hex(rawBuffer);
  const normalizedTensorSha256 = await sha256Hex(normalizedBytes);
  const hashMsBase = performance.now() - hashStarted;

  const pruningStarted = performance.now();
  const patchSize = input.config.pruning?.patch_size ?? 16;
  const stageLayers = input.config.pruning?.stage_layers ?? [3, 6, 9];
  const stageKeepRatios = input.config.pruning?.stage_keep_ratios ?? [0.7, 0.49, 0.343];
  const totalPatches = input.config.pruning?.total_patches ?? (input.config.input_size / patchSize) ** 2;
  const { scores, patchesPerSide } = computePatchScores(rgb, input.config.input_size, patchSize);
  const stageSummaries = stageKeepRatios.map((keepRatio, stageIndex) => {
    const keptPatches = Math.max(1, Math.round(totalPatches * keepRatio));
    return {
      stage_index: stageIndex,
      layer: stageLayers[stageIndex] ?? stageLayers[stageLayers.length - 1] ?? 0,
      keep_ratio: Number(keepRatio.toFixed(6)),
      kept_patches: keptPatches,
      dropped_patches: totalPatches - keptPatches,
      visible_area_ratio: Number((keptPatches / totalPatches).toFixed(6))
    };
  });
  const finalKeptPatches = stageSummaries[stageSummaries.length - 1]?.kept_patches ?? totalPatches;
  const finalKeepMask = buildKeepMask(scores, finalKeptPatches);
  const prunedImageData = buildPrunedImageData(imageData, finalKeepMask, patchSize, patchesPerSide);
  const processedPreviewUrl = await buildPreviewUrlFromImageData(imageData, sniffed.mime);
  const prunedPreviewUrl = await buildPreviewUrlFromImageData(prunedImageData, sniffed.mime);
  const pruningMs = performance.now() - pruningStarted;

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
  const totalMs = decodeMs + preprocessMs + dqaMs + hashMs + pruningMs + shareBuildMs;

  bitmap.close();
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
      server_should_receive_plain_pixel_values: false
    },
    controlPlaneMetrics: {
      decode_ms: Number(decodeMs.toFixed(3)),
      preprocess_ms: Number(preprocessMs.toFixed(3)),
      dqa_ms: Number(dqaMs.toFixed(3)),
      hash_ms: Number(hashMs.toFixed(3)),
      pruning_preview_ms: Number(pruningMs.toFixed(3)),
      share_build_ms: Number(shareBuildMs.toFixed(3)),
      total_ms: Number(totalMs.toFixed(3))
    },
    pruningPreview: {
      original_dimensions: { width: sniffed.width, height: sniffed.height },
      processed_dimensions: { width: input.config.input_size, height: input.config.input_size },
      patch_size: patchSize,
      grid_size: patchesPerSide,
      total_patches: totalPatches,
      estimated_effective_pixels: finalKeptPatches * patchSize * patchSize,
      stage_summaries: stageSummaries,
      final_kept_patches: finalKeptPatches,
      final_visible_area_ratio: Number((finalKeptPatches / totalPatches).toFixed(6)),
      processed_preview_url: processedPreviewUrl,
      pruned_preview_url: prunedPreviewUrl
    },
    share0: share0Bytes,
    share1: share1Bytes,
    processedPreviewUrl
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
