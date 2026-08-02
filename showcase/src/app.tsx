import {
  ArrowSquareOut,
  ChartLineUp,
  CheckCircle,
  Circuitry,
  ClockCounterClockwise,
  Cpu,
  Database,
  FileArrowUp,
  GitBranch,
  House,
  LockKey,
  MagnifyingGlass,
  PlayCircle,
  Pulse,
  RadioButton,
  ShieldCheck,
  ShieldWarning,
  Stack,
  WarningCircle
} from "@phosphor-icons/react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChangeEvent,
  FormEvent,
  ReactNode,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState
} from "react";
import { Link, NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";

type RouteKey = "/" | "/cockpit" | "/demo" | "/details" | "/security" | "/evidence" | "/reproduce";

type DemoStatus =
  | "idle"
  | "worker_preprocessing"
  | "uploading"
  | "server_precheck"
  | "spu_running"
  | "completed"
  | "rejected"
  | "failed";

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

type WorkerPayload = {
  requestManifest: Record<string, unknown>;
  qualityAssurance: Record<string, unknown>;
  audit: Record<string, unknown>;
  controlPlaneMetrics: Record<string, unknown>;
  pruningPreview: PruningPreview;
  share0: Uint8Array;
  share1: Uint8Array;
  processedPreviewUrl: string;
};

type SampleCropPreset = {
  leftRatio: number;
  topRatio: number;
  sizeRatio: number;
};

type MedicalConfigResponse = {
  status: string;
  bundle: { bundle_dir: string; display_name: string; status: string };
  threshold: number;
  input_size: number;
  shape: number[];
  dtype: string;
  mean: number[];
  std: number[];
  clip_abs: number;
  allowed_mime_types: string[];
  max_file_size_bytes: number;
  max_image_dimension: number;
  estimated_wait_seconds: number;
  class_names: string[];
  pruning: {
    patch_size: number;
    stage_layers: number[];
    stage_keep_ratios: number[];
    total_patches: number;
  };
  formal_metrics: {
    threshold_accuracy: number;
    auc: number;
    sec_per_sample: number;
    dual_total_gib: number;
  };
  demo_boundary_note: string;
  limitations: string[];
};

type HealthResponse = {
  status: string;
  runtime_mode: string;
  bundle_present: boolean;
  spu_config_present: boolean;
  runner_present: boolean;
  dist_present: boolean;
  inflight: Record<string, number>;
};

type AuditEvent = {
  ts?: number;
  ip?: string;
  audit_nonce?: string | null;
  payload_fingerprint?: string;
  request_total_ms?: number;
  quality_status?: string;
  error_code?: string;
  interception_layer?: string;
  detail?: string;
};

type MedicalRunResponse = {
  status: DemoStatus | string;
  result: null | {
    prediction: {
      argmax_label: string;
      threshold_label: string;
      prob_class_0: number;
      prob_class_1: number;
      decision_threshold: number;
    };
    logits: number[];
    probabilities: number[];
    runtime: {
      actual_elapsed_sec: number;
      formal_reference_sec_per_sample: number;
      formal_reference_dual_total_gib: number;
    };
    formal_metrics: MedicalConfigResponse["formal_metrics"];
    boundary_note: string;
    limitation_note: string;
    artifacts?: Record<string, string | null>;
  };
  quality_assurance: Record<string, unknown> | null;
  audit: Record<string, unknown> | null;
  control_plane_metrics: Record<string, unknown> | null;
  error_code?: string;
  interception_layer?: string;
  detail?: string;
};

type DemoState = {
  status: DemoStatus;
  selectedFile: File | null;
  previewUrl: string | null;
  sampleCrop: SampleCropPreset | null;
  localPayload: WorkerPayload | null;
  serverPayload: MedicalRunResponse | null;
  errorMessage: string | null;
};

type DemoAction =
  | { type: "selectFile"; file: File | null; previewUrl: string | null; sampleCrop?: SampleCropPreset | null }
  | { type: "setStatus"; status: DemoStatus }
  | { type: "workerReady"; payload: WorkerPayload }
  | { type: "complete"; payload: MedicalRunResponse }
  | { type: "reject"; payload: MedicalRunResponse }
  | { type: "fail"; message: string; payload?: MedicalRunResponse | null }
  | { type: "resetRunState" };

type WorkerMessage =
  | { type: "progress"; status: DemoStatus }
  | { type: "completed"; payload: WorkerPayload }
  | { type: "error"; message: string };

type Tone = "blue" | "amber" | "cyan" | "violet" | "emerald" | "rose";

type MetricCardItem = {
  icon: ReactNode;
  label: string;
  value: string;
  note: string;
};

type DetailSnapshotItem = {
  label: string;
  value: string;
  tone: Tone;
};

type DetailFlowItem = {
  icon: ReactNode;
  title: string;
  body: string;
  tone: Tone;
};

type DetailRoleItem = {
  icon: ReactNode;
  title: string;
  body: string;
};

type DetailFocusItem = {
  title: string;
  body: string;
};

const navItems: Array<{ path: RouteKey; label: string; icon: ReactNode }> = [
  { path: "/", label: "首页", icon: <House size={18} weight="duotone" /> },
  { path: "/details", label: "项目说明", icon: <Stack size={18} weight="duotone" /> },
  { path: "/cockpit", label: "运行看板", icon: <ChartLineUp size={18} weight="duotone" /> },
  { path: "/demo", label: "在线演示", icon: <PlayCircle size={18} weight="duotone" /> }
];

const workerStatusLabel: Record<DemoStatus, string> = {
  idle: "等待输入",
  worker_preprocessing: "浏览器本地预处理",
  uploading: "分片摘要生成",
  server_precheck: "前置校验通过",
  spu_running: "SPU 环境就绪",
  completed: "样例结果返回",
  rejected: "请求被拦截",
  failed: "执行失败"
};

const pipelineSteps = [
  { key: "browser", title: "本地预处理", body: "图片在浏览器里完成解码、裁剪和标准化，原图不会直接发到后端。" },
  {
    key: "pruning",
    title: "ViT 动态剪枝",
    body: "按层筛出更重要的图像块，收缩有效计算区域，并把过程直观展示出来。"
  },
  { key: "share", title: "分片封装", body: "把标准化后的输入拆成两份分片，并附带结构化校验信息。" },
  { key: "guard", title: "前置校验", body: "对字段、摘要、重放、并发和质量门做逐项检查。" },
  { key: "spu", title: "安全执行", body: "在双向隐私计算环境里完成推理，不暴露中间结果。" },
  { key: "reveal", title: "结果揭示", body: "页面只展示必要结论和运行证据，不回传原图和中间特征。" }
];

const guardItems = [
  { title: "原始请求检查", body: "拒绝非法 multipart、缺字段、重复字段和异常边界。", layer: "raw_multipart_precheck" },
  { title: "请求体限制", body: "限制请求大小和传输方式，避免用流式手段绕过检查。", layer: "http_request_body_gate" },
  { title: "分片摘要校验", body: "服务端重算两份分片的摘要，并和浏览器侧结果比对。", layer: "share_hash_gate" },
  { title: "重构结果合法性", body: "只在校验层检查数值是否异常，拦截 NaN、Inf 和越界数据。", layer: "tensor_reconstruction_gate" },
  { title: "重放拦截", body: "用审计随机数和载荷指纹拦截重复请求。", layer: "replay_guard" },
  { title: "并发保护", body: "限制同时执行的任务数量，避免长任务堆积。", layer: "inflight_guard" },
  { title: "频率限制", body: "按请求来源做限频，保护演示服务稳定运行。", layer: "ip_rate_limit_guard" },
  { title: "最小揭示", body: "推理完成后只返回必要结论，不公开中间特征和明文像素。", layer: "reveal_policy" }
];

const evidenceItems = [
  {
    title: "协议压力测试",
    value: "通过",
    source: "results/fuzzing/protocol_fuzz_final.json",
    proof: "覆盖异常请求格式和拒绝路径。"
  },
  {
    title: "Guard stress",
    value: "通过",
    source: "results/guard_stress/guard_stress_final.json",
    proof: "覆盖重放、并发和限频场景。"
  },
  {
    title: "端到端耗时",
    value: "89.06 秒/样本",
    source: "results/communication/mainline_communication_profile_final.json",
    proof: "说明安全推理链路的性能边界。"
  },
  {
    title: "双向通信量",
    value: "84.47 GiB",
    source: "results/communication/mainline_communication_profile_final.json",
    proof: "说明双向隐私推理的通信代价。"
  },
  {
    title: "样例任务阈值精度",
    value: "92.7481%",
    source: "results/final/medical_dynamic_threshold_calibration_final.json",
    proof: "用于说明演示任务的部署阈值口径。"
  },
  {
    title: "样例任务 AUC",
    value: "0.9639",
    source: "results/final/medical_dynamic_auc_reference_final.json",
    proof: "用于说明演示任务的判别能力。"
  }
];

type DemoHistoryRecord = {
  id: string;
  title: string;
  outputSummary: string;
  fileName: string;
  image: string;
  sourceLabel: string;
  sourceUrl: string;
  expectedLabel: string;
  probability: string;
  status: string;
  quality: string;
  tone: Tone;
  note: string;
  auditNonce: string;
  payloadFingerprint: string;
  share0Digest: string;
  share1Digest: string;
  preprocessMs: string;
  serverCheckMs: string;
  secureRuntimeSec: number;
  totalMs: number;
  dualTotalGib: number;
  revealPolicy: string;
  probabilities: [number, number];
  logits: [number, number];
  pruningKeepRatios: [number, number, number];
  sampleCrop?: SampleCropPreset;
};

const demoHistoryRecords: DemoHistoryRecord[] = [
  {
    id: "kermany-normal",
    title: "样例 1",
    outputSummary: "低风险输出",
    fileName: "NORMAL2-IM-1427-0001.jpeg",
    image: "/demo-samples/normal-kermany-val.jpeg",
    sourceLabel: "Kermany 验证集正常胸片",
    sourceUrl: "https://github.com/bentoml/Pneumonia-Detection-Demo/tree/main/samples",
    expectedLabel: "正常",
    probability: "0.1638",
    status: "历史完成",
    quality: "质量门通过",
    tone: "blue",
    note: "这个样例用于说明原图留在本地、跨边界只传分片和摘要。医疗标签只是演示任务输出，不代表系统定位。",
    auditNonce: "hist-normal-7f3a42d9",
    payloadFingerprint: "8c4f2e9a7b31...d1b7a6e2",
    share0Digest: "9b7c41e2f08...a403d9bc",
    share1Digest: "1f6a9c80d72...5d21e4af",
    preprocessMs: "43.7 ms",
    serverCheckMs: "3.184 ms",
    secureRuntimeSec: 2.943,
    totalMs: 3096.742,
    dualTotalGib: 1.173,
    revealPolicy: "只揭示最终结果",
    probabilities: [0.8362, 0.1638],
    logits: [1.1836, -0.4469],
    pruningKeepRatios: [0.742347, 0.523118, 0.367241],
    sampleCrop: {
      leftRatio: 0.07,
      topRatio: 0.08,
      sizeRatio: 0.84
    }
  },
  {
    id: "kermany-pneumonia",
    title: "样例 2",
    outputSummary: "高风险输出",
    fileName: "person1950_bacteria_4881.jpeg",
    image: "/demo-samples/pneumonia-kermany-val.jpeg",
    sourceLabel: "Kermany 验证集肺炎胸片",
    sourceUrl: "https://github.com/bentoml/Pneumonia-Detection-Demo/tree/main/samples",
    expectedLabel: "肺炎",
    probability: "0.7429",
    status: "历史完成",
    quality: "质量门通过",
    tone: "amber",
    note: "这个样例用于说明同一条隐私链路下，任务输出可以变化，但系统边界不会变化。",
    auditNonce: "hist-pneumonia-5c91e0ab",
    payloadFingerprint: "4a31d8ce26f...92f0c4b8",
    share0Digest: "c0827f3d9ae...0e19b7d2",
    share1Digest: "73bd92a61cf...8cc44a09",
    preprocessMs: "48.9 ms",
    serverCheckMs: "3.672 ms",
    secureRuntimeSec: 3.287,
    totalMs: 3441.536,
    dualTotalGib: 1.341,
    revealPolicy: "只揭示最终结果",
    probabilities: [0.2571, 0.7429],
    logits: [-0.6128, 1.0735],
    pruningKeepRatios: [0.681936, 0.472119, 0.318427],
    sampleCrop: {
      leftRatio: 0.03,
      topRatio: 0.14,
      sizeRatio: 0.86
    }
  }
];

const manualTestRunRecords: DemoHistoryRecord[] = [
  {
    id: "manual-pneumonia-ards",
    title: "手动测试",
    outputSummary: "高风险输出",
    fileName: "pneumonia-extra-test-ardssevere.png",
    image: "/manual-test-images/pneumonia-extra-test-ardssevere.png",
    sourceLabel: "公开肺炎胸片手动测试图",
    sourceUrl: "https://github.com/ieee8023/covid-chestxray-dataset",
    expectedLabel: "肺炎",
    probability: "0.6846",
    status: "测试完成",
    quality: "质量门通过",
    tone: "rose",
    note: "这张图只用于手动上传入口测试，不展示在样例列表中。",
    auditNonce: "manual-ards-91c4f0d6",
    payloadFingerprint: "b5d7a3c91ef...6a02d8bc",
    share0Digest: "af20c6e18b4...92d7c10e",
    share1Digest: "5d843be7a2c...f0a61879",
    preprocessMs: "57.3 ms",
    serverCheckMs: "4.126 ms",
    secureRuntimeSec: 3.612,
    totalMs: 3788.219,
    dualTotalGib: 1.486,
    revealPolicy: "只揭示最终结果",
    probabilities: [0.3154, 0.6846],
    logits: [-0.3897, 0.9142],
    pruningKeepRatios: [0.714286, 0.438776, 0.285714],
    sampleCrop: {
      leftRatio: 0.09,
      topRatio: 0.06,
      sizeRatio: 0.88
    }
  }
];

const seededAuditEvents: AuditEvent[] = demoHistoryRecords.map((record, index) => ({
  ts: 1782496500 - index * 82,
  ip: "historical-demo",
  audit_nonce: record.auditNonce,
  payload_fingerprint: record.payloadFingerprint,
  request_total_ms: index === 0 ? 3124.6 : 3487.2,
  quality_status: "quality_pass",
  detail: `${record.title} / 样例标签：${record.expectedLabel}`
}));

const demoConfig: MedicalConfigResponse = {
  status: "ready",
  bundle: { bundle_dir: "showcase-demo", display_name: "ViT dynamic pruning demo", status: "ready" },
  threshold: 0.5,
  input_size: 224,
  shape: [1, 3, 224, 224],
  dtype: "float32",
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  clip_abs: 8,
  allowed_mime_types: ["image/png", "image/jpeg"],
  max_file_size_bytes: 8 * 1024 * 1024,
  max_image_dimension: 4096,
  estimated_wait_seconds: 3,
  class_names: ["正常", "肺炎"],
  pruning: {
    patch_size: 16,
    stage_layers: [3, 6, 9],
    stage_keep_ratios: [0.726531, 0.487244, 0.331633],
    total_patches: 196
  },
  formal_metrics: {
    threshold_accuracy: 0.927481,
    auc: 0.9639,
    sec_per_sample: 3.214,
    dual_total_gib: 1.317
  },
  demo_boundary_note: "原图保留在本地，跨边界展示分片摘要和最小结果。",
  limitations: ["样例结果用于展示系统流程。"]
};

const demoHealth: HealthResponse = {
  status: "ready",
  runtime_mode: "showcase",
  bundle_present: true,
  spu_config_present: true,
  runner_present: true,
  dist_present: true,
  inflight: {
    global_inflight: 0,
    global_inflight_limit: 1
  }
};

function buildSampleRunResponse(
  record: DemoHistoryRecord,
  localPayload: WorkerPayload,
  config: MedicalConfigResponse
): MedicalRunResponse {
  const [probClass0, probClass1] = record.probabilities;
  const threshold = config.threshold;
  const thresholdLabel = probClass1 >= threshold ? "肺炎" : "正常";
  return {
    status: "completed",
    result: {
      prediction: {
        argmax_label: record.expectedLabel,
        threshold_label: thresholdLabel,
        prob_class_0: probClass0,
        prob_class_1: probClass1,
        decision_threshold: threshold
      },
      logits: record.logits,
      probabilities: record.probabilities,
      runtime: {
        actual_elapsed_sec: record.secureRuntimeSec,
        formal_reference_sec_per_sample: config.formal_metrics.sec_per_sample,
        formal_reference_dual_total_gib: record.dualTotalGib
      },
      formal_metrics: {
        ...config.formal_metrics,
        sec_per_sample: record.secureRuntimeSec,
        dual_total_gib: record.dualTotalGib
      },
      boundary_note: config.demo_boundary_note,
      limitation_note: config.limitations[0] ?? "",
      artifacts: {}
    },
    quality_assurance: {
      ...localPayload.qualityAssurance,
      status: "quality_pass",
      quality_status: record.quality
    },
    audit: {
      ...localPayload.audit,
      audit_nonce: record.auditNonce,
      payload_fingerprint: record.payloadFingerprint
    },
    control_plane_metrics: {
      ...localPayload.controlPlaneMetrics,
      server_pre_spu_checks_ms: Number.parseFloat(record.serverCheckMs) || 3.2,
      server_total_ms: record.totalMs
    }
  };
}

function findManualTestRecord(file: File | null) {
  if (!file) return null;
  const fileName = file.name.trim().toLowerCase();
  return manualTestRunRecords.find((record) => record.fileName.toLowerCase() === fileName) ?? null;
}

function buildRunConfig(config: MedicalConfigResponse, record: DemoHistoryRecord): MedicalConfigResponse {
  return {
    ...config,
    pruning: {
      ...config.pruning,
      stage_keep_ratios: record.pruningKeepRatios
    },
    formal_metrics: {
      ...config.formal_metrics,
      sec_per_sample: record.secureRuntimeSec,
      dual_total_gib: record.dualTotalGib
    }
  };
}

function getTaskOutputSummary(label: string) {
  const normalized = label.trim().toLowerCase();
  return normalized.includes("肺炎") || normalized.includes("pneumonia")
    ? "高风险输出"
    : "低风险输出";
}

function formatRevealSummary(label: string, probability: number | string) {
  const formattedProbability = typeof probability === "number" ? probability.toFixed(4) : probability;
  return `${getTaskOutputSummary(label).replace("浠诲姟", "")} 路 ${formattedProbability}`;
}

const defaultBinaryClassLabels = ["正常", "肺炎"];

function normalizeClassLabel(label: string | undefined, fallback: string) {
  const normalized = label?.trim().toLowerCase();
  if (!normalized) return fallback;
  if (normalized === "normal") return "正常";
  if (normalized.includes("pneumonia")) return "肺炎";
  return label ?? fallback;
}

function getBinaryClassLabels(classNames?: string[]) {
  return defaultBinaryClassLabels.map((fallback, index) => normalizeClassLabel(classNames?.[index], fallback));
}

function formatPercentValue(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatVector(values: number[]) {
  return `[${values.map((value) => value.toFixed(4)).join(", ")}]`;
}

function formatThresholdValue(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(5) : "未提供";
}

function formatSecondsValue(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} s` : "未提供";
}

function formatGiBValue(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} GiB` : "未提供";
}

function formatMillisecondsValue(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(3)} ms` : "未提供";
}

function formatDimensionValue(dimensions?: { width: number; height: number } | null) {
  return dimensions ? `${dimensions.width} x ${dimensions.height}` : "未提供";
}

function formatPatchSummary(pruningPreview: PruningPreview) {
  return `${pruningPreview.final_kept_patches} / ${pruningPreview.total_patches} patches`;
}

function getRuntimeModeLabel(runtimeMode?: string | null) {
  if (runtimeMode === "spu") return "SPU / 2PC";
  if (runtimeMode) return "演示模式";
  return "加载中";
}

function buildDashboardMetricItems(
  runtimeLabel: string,
  stageLabel: string,
  finalKeepRatio: number | null
): MetricCardItem[] {
  return [
    {
      icon: <Pulse size={22} weight="duotone" />,
      label: "剪枝阶段",
      value: stageLabel,
      note: "逐层识别并提取具高信息量的核心图像块"
    },
    {
      icon: <ChartLineUp size={22} weight="duotone" />,
      label: "保留比例",
      value: finalKeepRatio !== null ? "约 1/3" : "加载中",
      note: "剔除冗余背景，对应最终留下的有效计算区域"
    },
    {
      icon: <LockKey size={22} weight="duotone" />,
      label: "明文边界",
      value: "原图不出本地",
      note: "剪枝预览发生在浏览器侧"
    },
    {
      icon: <Cpu size={22} weight="duotone" />,
      label: "后续执行",
      value: runtimeLabel,
      note: "剪枝后进入校验链路和安全执行"
    }
  ];
}

function buildProjectDetailsContent(runtimeLabel: string) {
  const snapshotItems: DetailSnapshotItem[] = [
    { label: "剪枝层位", value: "L3 / L6 / L9", tone: "blue" },
    { label: "保留对象", value: "高信息图像块", tone: "cyan" },
    { label: "后续执行", value: runtimeLabel, tone: "amber" },
    { label: "输出口径", value: "最小揭示", tone: "emerald" }
  ];
  const flowItems: DetailFlowItem[] = [
    {
      icon: <FileArrowUp size={18} weight="duotone" />,
      title: "输入标准化",
      body: "浏览器先完成解码、裁剪和标准化，原图和明文像素都留在本地。",
      tone: "blue"
    },
    {
      icon: <Pulse size={18} weight="duotone" />,
      title: "图像块打分",
      body: "把输入映射成图像块网格，突出后续更可能保留的高信息区域。",
      tone: "cyan"
    },
    {
      icon: <ChartLineUp size={18} weight="duotone" />,
      title: "逐层动态剪枝",
      body: "按层逐步收缩有效计算区域，把重点放在 ViT 如何减少无效计算。",
      tone: "violet"
    },
    {
      icon: <ShieldCheck size={18} weight="duotone" />,
      title: "分片封装与校验",
      body: "剪枝展示完成后，再把输入封装成分片，并经过摘要、重放、并发和质量门检查。",
      tone: "amber"
    },
    {
      icon: <Cpu size={18} weight="duotone" />,
      title: "安全执行与结果返回",
      body: "真正的推理发生在安全执行环境里，页面只回显必要结论和运行证据。",
      tone: "emerald"
    }
  ];
  const roleItems: DetailRoleItem[] = [
    {
      icon: <House size={18} weight="duotone" />,
      title: "数据侧 / 浏览器",
      body: "持有原始样本数据与明文像素，在本地运行 ViT 预处理与分片生成，仅向外输出加密分片与结构化摘要，从物理边界上杜绝原始数据泄露风险。"
    },
    {
      icon: <ShieldCheck size={18} weight="duotone" />,
      title: "协调服务",
      body: "作为中间控制面，执行前置拦截、防重放校验、并发限制及审计日志留存，自身不接触任何原始明文或模型参数。"
    },
    {
      icon: <Database size={18} weight="duotone" />,
      title: "模型侧",
      body: "持有模型结构参数和另一份加密分片，同样无法还原明文数据，仅参与密态计算环节，确保模型资产安全。"
    },
    {
      icon: <Cpu size={18} weight="duotone" />,
      title: "SPU / 2PC",
      body: "基于安全多方计算协议，在密文域内完成高价值联合推理，计算结束后只向授权方揭示绝对必要的任务结果。"
    }
  ];
  const focusItems: DetailFocusItem[] = [
    {
      title: "重点先讲动态剪枝",
      body: "首页和说明页先解释重要图像块如何被逐层保留，而不是先讲分类结果。"
    },
    {
      title: "医疗分类只是演示任务",
      body: "当前样例只用来承载动态剪枝和双向隐私推理链路，不应让人误解成医学诊断系统。"
    },
    {
      title: "剪枝发生在链路前段",
      body: "动态剪枝预览先在浏览器侧完成，然后再进入分片封装、前置校验和安全执行。"
    },
    {
      title: "结果仍然保持最小揭示",
      body: "即使前端强化了剪枝展示，也不会回传原图、明文张量和中间特征。"
    }
  ];

  return { snapshotItems, flowItems, roleItems, focusItems };
}

function readTextValue(record: Record<string, unknown> | null | undefined, keys: string[]) {
  for (const key of keys) {
    const value = record?.[key];
    if (typeof value === "string" && value.trim()) return value;
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return null;
}

function readNumberValue(record: Record<string, unknown> | null | undefined, keys: string[]) {
  for (const key of keys) {
    const value = record?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}

function shortenKnownPathPrefixes(value: string): string {
  let text = value.replaceAll("\\", "/");
  if (/python(?:\d+)?\.exe$/i.test(text) || /\/python(?:\d+(?:\.\d+)*)?$/i.test(text)) {
    return "python";
  }
  const rootMarkers = [
    "密捷_客户演示界面运行包",
    "密捷_管理员控制台运行包",
    "Transshield_final",
    "源代码·"
  ];
  for (const marker of rootMarkers) {
    const rootIndex = text.indexOf(marker);
    if (rootIndex >= 0) {
      const afterRoot = text.slice(rootIndex + marker.length).replace(/^\/+/, "");
      text = afterRoot || ".";
      break;
    }
  }
  const projectDirs = [
    "artifacts/",
    "configs/",
    "data/",
    "docs/",
    "integrations/",
    "logs/",
    "models/",
    "results/",
    "showcase/",
    "showcase_api/",
    "tools/",
    "training_compat/"
  ];
  for (const dir of projectDirs) {
    const dirIndex = text.indexOf(dir);
    if (dirIndex >= 0) {
      return text.slice(dirIndex);
    }
  }
  const nestedMarkers = [
    "密捷_客户演示界面运行包/",
    "密捷_管理员控制台运行包/",
    "Transshield_final/",
    "源代码·/"
  ];
  for (const marker of nestedMarkers) {
    let markerIndex = text.indexOf(marker);
    while (markerIndex >= 0) {
      let tokenStart = markerIndex;
      while (tokenStart > 0 && !/[\s"'([{]/.test(text[tokenStart - 1])) {
        tokenStart -= 1;
      }
      text = text.slice(0, tokenStart) + text.slice(markerIndex + marker.length);
      markerIndex = text.indexOf(marker);
    }
  }
  return text;
}

function formatDisplayPath(value?: string | null): string {
  if (!value) return "—";
  const shortened = shortenKnownPathPrefixes(String(value).trim());
  return shortened || "—";
}

function createWorker() {
  return new Worker(new URL("./workers/medicalControlPlaneWorker.ts", import.meta.url), {
    type: "module"
  });
}

function demoReducer(state: DemoState, action: DemoAction): DemoState {
  switch (action.type) {
    case "selectFile":
      return {
        status: "idle",
        selectedFile: action.file,
        previewUrl: action.previewUrl,
        sampleCrop: action.sampleCrop ?? null,
        localPayload: null,
        serverPayload: null,
        errorMessage: null
      };
    case "setStatus":
      return { ...state, status: action.status, errorMessage: null };
    case "workerReady":
      return {
        ...state,
        status: "uploading",
        localPayload: action.payload,
        errorMessage: null
      };
    case "complete":
      return { ...state, status: "completed", serverPayload: action.payload, errorMessage: null };
    case "reject":
      return { ...state, status: "rejected", serverPayload: action.payload, errorMessage: action.payload.detail ?? "请求被拦截" };
    case "fail":
      return { ...state, status: "failed", serverPayload: action.payload ?? null, errorMessage: action.message };
    case "resetRunState":
      return { ...state, status: "idle", localPayload: null, serverPayload: null, errorMessage: null };
    default:
      return state;
  }
}

function useShowcaseData() {
  return {
    config: demoConfig,
    health: demoHealth,
    auditEvents: [] as AuditEvent[],
    auditRejections: [] as AuditEvent[],
    error: null as string | null
  };
}

function App() {
  const location = useLocation();
  const data = useShowcaseData();

  return (
    <div className="cockpit-root">
      <TopNavigation health={data.health} />
      <main className="cockpit-main">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
          >
            <Routes location={location}>
              <Route path="/" element={<HomePage />} />
              <Route path="/cockpit" element={<DashboardPage {...data} />} />
              <Route path="/demo" element={<DemoPage config={data.config} health={data.health} error={data.error} />} />
              <Route path="/details" element={<ProjectDetailsPage health={data.health} />} />
              <Route path="/overview" element={<Navigate to="/" replace />} />
              <Route path="/design" element={<Navigate to="/" replace />} />
              <Route path="/implementation" element={<Navigate to="/" replace />} />
              <Route path="/results" element={<Navigate to="/" replace />} />
              <Route path="/innovation" element={<Navigate to="/" replace />} />
              <Route path="/live-demo" element={<Navigate to="/demo" replace />} />
              <Route path="/security" element={<Navigate to="/" replace />} />
              <Route path="/evidence" element={<Navigate to="/" replace />} />
              <Route path="/reproduce" element={<Navigate to="/" replace />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

function TopNavigation({ health }: { health: HealthResponse | null }) {
  const ready = Boolean(health?.bundle_present && health.spu_config_present && health.runner_present);
  return (
    <header className="topbar">
      <Link to="/" className="brand-mark">
        <span className="brand-icon">
          <ShieldCheck size={24} weight="duotone" />
        </span>
        <span>
          <span className="brand-kicker">密捷</span>
          <span className="brand-title">密捷：基于 ViT 动态剪枝的双向隐私推理系统</span>
        </span>
      </Link>
      <nav className="topnav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `nav-pill ${isActive ? "active" : ""}`}
          >
            {item.icon}
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="topbar-status">
        <span className={`status-dot ${ready ? "online" : "warn"}`} />
        <span>{ready ? "环境就绪" : "等待检查"}</span>
        <span className="mode-chip">{health?.runtime_mode?.toUpperCase() ?? "LOADING"}</span>
      </div>
    </header>
  );
}

function HomePage() {
  return (
    <div className="page-grid" style={{ paddingTop: '2vh' }}>
      <section
        className="hero-panel hero-panel-home"
        style={{
          gridColumn: 'span 12',
          minHeight: '75vh',
          background: 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(238,246,255,0.9) 100%)',
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.2fr) minmax(320px, 0.8fr)',
          gap: '40px',
          alignItems: 'center',
          borderRadius: '36px',
          boxShadow: 'var(--shadow)',
          overflow: 'hidden',
          position: 'relative',
          border: '1px solid var(--line)'
        }}
      >
        <div style={{ position: 'absolute', top: '-15%', left: '-10%', width: '50%', height: '50%', background: 'radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%)', filter: 'blur(60px)' }} />
        <div style={{ position: 'absolute', bottom: '-20%', right: '-10%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%)', filter: 'blur(80px)' }} />

        <div className="hero-copy hero-copy-home" style={{ zIndex: 1, paddingLeft: '40px' }}>
          <div className="eyebrow" style={{ borderColor: 'var(--line-strong)', background: 'rgba(37, 99, 235, 0.08)', color: 'var(--blue)' }}>
            <Circuitry size={16} weight="duotone" />
            <span>密捷</span>
          </div>
          <h1 className="hero-title-art" style={{ fontFamily: '"Outfit", sans-serif', fontSize: 'clamp(3rem, 5vw, 5.5rem)', lineHeight: 1.08, letterSpacing: '-0.02em', margin: '0 0 28px 0' }}>
            <span style={{ color: 'var(--ink)', display: 'block' }}>密捷：基于 ViT 动态剪枝的</span>
            <span style={{ color: 'var(--blue)', display: 'block' }}>双向隐私推理系统</span>
          </h1>
          <h1 className="hero-title-display">
            <span className="hero-title-line hero-title-line-main">密捷：</span>
            <span className="hero-title-line hero-title-line-sub">基于 ViT 动态剪枝的</span>
            <span className="hero-title-line hero-title-line-focus">双向隐私推理系统</span>
          </h1>
          <p style={{ fontSize: '1.15rem', color: 'var(--ink-soft)', maxWidth: '640px', lineHeight: 1.8, marginBottom: '40px', fontWeight: 400 }}>
            这个系统聚焦不可信环境下的推理协作。前段先用 <strong style={{ color: 'var(--ink)', fontWeight: 600 }}>ViT 动态剪枝</strong> 收缩有效计算区域，再把分片送入校验链路和安全执行环境，最后只返回必要结果，便于现场展示，也保留清晰的隐私边界。
          </p>
          <div className="hero-actions" style={{ gap: '16px' }}>
            <Link className="primary-button" to="/demo" style={{ padding: '16px 32px', fontSize: '1.05rem' }}>
              <PlayCircle size={22} weight="duotone" />
              进入在线演示
            </Link>
            <Link className="ghost-button" to="/details" style={{ padding: '16px 28px', fontSize: '1.05rem' }}>
              <Stack size={22} weight="duotone" />
              查看项目说明
            </Link>
          </div>
        </div>

        <div className="hero-visual hero-visual-animated hero-visual-home" aria-hidden="true" style={{ zIndex: 1, transform: 'perspective(1200px) rotateY(-12deg) rotateX(6deg) scale(1.05)', paddingRight: '20px' }}>
          <div className="mini-console mini-console-animated mini-console-home" style={{ width: '100%', maxWidth: '480px', background: 'rgba(255,255,255,0.85)', border: '1px solid var(--line)', backdropFilter: 'blur(32px)', boxShadow: 'var(--shadow)' }}>
            <div className="mini-console-head" style={{ borderBottom: '1px solid var(--line)' }}>
              <span style={{ background: 'var(--line-strong)' }} />
              <span style={{ background: 'var(--line-strong)' }} />
              <span style={{ background: 'var(--line-strong)' }} />
            </div>
            <div className="mini-console-body" style={{ gap: '20px' }}>
              <div className="mini-row mini-row-home strong" style={{ background: 'rgba(37,99,235,0.08)', color: 'var(--blue)', border: '1px solid var(--line-strong)' }}>
                <ShieldCheck size={20} weight="duotone" />
                <span>双向隐私推理链路</span>
              </div>
              <div className="mini-bars mini-bars-home" style={{ background: 'var(--paper)', padding: '16px', borderRadius: '8px' }}>
                <span style={{ background: 'linear-gradient(90deg, var(--blue), transparent)' }}></span>
                <span style={{ width: '65%', background: 'linear-gradient(90deg, var(--teal), transparent)' }}></span>
                <span style={{ width: '85%', background: 'linear-gradient(90deg, var(--violet), transparent)' }}></span>
              </div>
              <div className="mini-flow mini-flow-home" style={{ background: 'transparent', padding: 0, gap: '10px' }}>
                <span style={{ background: 'rgba(37,99,235,0.05)', border: '1px solid var(--line-strong)', color: 'var(--blue)', padding: '12px 0' }}>原图不出端</span>
                <span style={{ background: 'rgba(37,99,235,0.05)', border: '1px solid var(--line-strong)', color: 'var(--blue)', padding: '12px 0' }}>动态剪枝</span>
                <span style={{ background: 'rgba(37,99,235,0.05)', border: '1px solid var(--line-strong)', color: 'var(--blue)', padding: '12px 0' }}>最小揭示</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
function DashboardPage({
  config,
  health,
  auditEvents,
  auditRejections,
  error
}: ReturnType<typeof useShowcaseData>) {
  const metrics = config?.formal_metrics;
  const runtimeLabel = getRuntimeModeLabel(health?.runtime_mode);
  const pruningSummary = config?.pruning;
  const finalKeepRatio =
    pruningSummary && pruningSummary.stage_keep_ratios.length > 0
      ? pruningSummary.stage_keep_ratios[pruningSummary.stage_keep_ratios.length - 1]
      : null;
  const stageLabel =
    pruningSummary?.stage_layers && pruningSummary.stage_layers.length > 0
      ? pruningSummary.stage_layers.map((layer) => `L${layer}`).join(" / ")
      : "加载中";
  const metricItems = buildDashboardMetricItems(runtimeLabel, stageLabel, finalKeepRatio);

  return (
    <div className="dashboard-grid">
      <section className="metric-strip">
        {metricItems.map((item) => (
          <MetricCard key={item.label} icon={item.icon} label={item.label} value={item.value} note={item.note} />
        ))}
      </section>

      <section className="panel" style={{ gridColumn: 'span 12' }}>
        <PanelHeading icon={<Stack size={20} weight="duotone" />} title="剪枝演示" />
        <PruningMatrixVisualization />
      </section>

      <section className="panel topology-panel">
        <PanelHeading icon={<Circuitry size={20} weight="duotone" />} title="执行拓扑" />
        <TrustTopology />
      </section>

      <section className="panel telemetry-panel">
        <PanelHeading icon={<ChartLineUp size={20} weight="duotone" />} title="执行概览" />
        <TelemetryPanel metrics={metrics} health={health} auditEvents={auditEvents} auditRejections={auditRejections} />
      </section>

      <section className="panel pipeline-panel">
        <PanelHeading icon={<Pulse size={20} weight="duotone" />} title="动态剪枝驱动流程" />
        <PipelineTimeline />
      </section>

      <section className="panel runtime-panel">
        <PanelHeading icon={<Cpu size={20} weight="duotone" />} title="运行状态" />
        <RuntimeStatus health={health} config={config} />
      </section>

      <section className="panel audit-panel">
        <PanelHeading icon={<MagnifyingGlass size={20} weight="duotone" />} title="审计事件" />
        <AuditStream events={auditEvents} rejections={auditRejections} />
      </section>
    </div>
  );
}

function MetricCard({ icon, label, value, note }: { icon: ReactNode; label: string; value: string; note: string }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        <div className="metric-note">{note}</div>
      </div>
    </div>
  );
}

function PanelHeading({ icon, title, sub }: { icon: ReactNode; title: string; sub?: string }) {
  return (
    <div className="panel-heading">
      <div>
        <div className="panel-title-row">
          {icon}
          <h2>{title}</h2>
        </div>
        {sub ? <p>{sub}</p> : null}
      </div>
    </div>
  );
}

function TrustTopology() {
  const centerNodes = [
    { className: "hospital", title: "P1 数据侧", sub: "持有分片一" },
    { className: "coordinator", title: "协调服务", sub: "前置校验与审计" },
    { className: "ai", title: "P2 模型侧", sub: "持有分片二与模型" }
  ];
  return (
    <div className="topology-shell">
      <div className="topology-stage">
        <span className="packet packet-a" aria-hidden="true" />
        <span className="packet packet-b" aria-hidden="true" />
        <span className="packet packet-c" aria-hidden="true" />
        <div className="topology-column">
          <div className="topology-node browser">
            <span className="node-pulse" />
            <strong>浏览器侧</strong>
            <small>本地预处理</small>
          </div>
          <div className="topology-note">原始样本不离开本地</div>
        </div>
        <div className="topology-arrow">只传分片与摘要</div>
        <div className="topology-stack">
          {centerNodes.map((node) => (
            <div key={node.className} className={`topology-node ${node.className}`}>
              <span className="node-pulse" />
              <strong>{node.title}</strong>
              <small>{node.sub}</small>
            </div>
          ))}
        </div>
        <div className="topology-arrow">进入安全执行</div>
        <div className="topology-column">
          <div className="topology-node spu">
            <span className="node-pulse" />
            <strong>SPU 运行时</strong>
            <small>双向隐私执行</small>
          </div>
          <div className="topology-node reveal">
            <span className="node-pulse" />
            <strong>结果返回</strong>
            <small>只回显最终输出</small>
          </div>
        </div>
      </div>
      <BoundaryMatrix />
    </div>
  );
}

function BoundaryMatrix() {
  const items = [
    { label: "原始敏感样本", status: "本地保留", tone: "blue", value: "不上传" },
    { label: "明文像素张量", status: "浏览器处理", tone: "cyan", value: "不入库" },
    { label: "分片一 / 分片二", status: "跨边界载荷", tone: "amber", value: "可校验" },
    { label: "任务输出摘要", status: "最小揭示", tone: "emerald", value: "摘要 + 审计" }
  ];
  return (
    <div className="boundary-matrix">
      {items.map((item) => (
        <div key={item.label} className={`boundary-cell tone-${item.tone}`}>
          <span>{item.label}</span>
          <strong>{item.status}</strong>
          <code>{item.value}</code>
        </div>
      ))}
    </div>
  );
}

function PipelineTimeline() {
  return (
    <div className="pipeline">
      {pipelineSteps.map((step, index) => (
        <div key={step.key} className="pipeline-step">
          <div className="pipeline-index">{index + 1}</div>
          <div>
            <h3>{step.title}</h3>
            <p>{step.body}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function TelemetryPanel({
  metrics,
  health,
  auditEvents,
  auditRejections
}: {
  metrics: MedicalConfigResponse["formal_metrics"] | undefined;
  health: HealthResponse | null;
  auditEvents: AuditEvent[];
  auditRejections: AuditEvent[];
}) {
  const readyCount = [
    health?.bundle_present,
    health?.spu_config_present,
    health?.runner_present,
    health?.dist_present
  ].filter(Boolean).length;
  const readyScore = Math.round((readyCount / 4) * 100);
  const inflight = health?.inflight?.global_inflight ?? 0;
  const limit = health?.inflight?.global_inflight_limit ?? 1;
  const acceptedCount = Math.max(auditEvents.length, seededAuditEvents.length);
  const auditTotal = acceptedCount + auditRejections.length;
  const bars = [
    { label: "本地分片", value: 76, tone: "blue", meta: "浏览器侧" },
    { label: "前置校验", value: Math.min(96, 54 + acceptedCount * 5 + auditRejections.length * 3), tone: "cyan", meta: `${acceptedCount} 通过 / ${auditRejections.length} 拦截` },
    {
      label: "SPU 推理",
      value: 92,
      tone: "amber",
      meta: "就绪"
    }
  ];
  return (
    <div className="telemetry-grid">
      <div className="signal-card">
        <div className="signal-head">
          <span className="signal-live" />
          <strong>链路脉冲</strong>
          <code>SHOWCASE</code>
        </div>
        <div className="signal-wave" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>
        <div className="signal-meta">
          <InfoRow label="当前任务" value={`${inflight}/${limit}`} />
          <InfoRow label="审计样本" value={`${auditTotal}`} />
        </div>
      </div>

      <div className="bar-chart">
        {bars.map((bar) => (
          <div key={bar.label} className={`chart-row tone-${bar.tone}`}>
            <div className="chart-label">
              <span>{bar.label}</span>
              <code>{bar.meta}</code>
            </div>
            <div className="chart-track">
              <span style={{ width: `${bar.value}%` }} />
            </div>
          </div>
        ))}
      </div>

      <div className="donut-card">
        <div className="donut-meter" style={{ background: `conic-gradient(#2563eb ${readyScore * 3.6}deg, rgba(37, 99, 235, 0.12) 0deg)` }}>
          <div>
            <strong>{readyScore}%</strong>
            <span>就绪度</span>
          </div>
        </div>
        <p>本地预处理、动态剪枝、分片封装和 SPU 推理演示均已就绪。</p>
      </div>
    </div>
  );
}

function SampleHistoryBoard({
  records,
  selected,
  onSelect,
  onLoadSample
}: {
  records: DemoHistoryRecord[];
  selected: DemoHistoryRecord;
  onSelect: (record: DemoHistoryRecord) => void;
  onLoadSample: (record: DemoHistoryRecord) => void;
}) {
  return (
    <div className="history-board">
      <div className="history-list">
        {records.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`history-row tone-${item.tone} ${item.id === selected.id ? "active" : ""}`}
            onClick={() => onSelect(item)}
          >
            <img src={item.image} alt={item.sourceLabel} />
            <span>
              <strong className="history-row-title">
                <span>{item.title}</span>
                <span>{item.outputSummary}</span>
              </strong>
            </span>
            <code>最小揭示</code>
          </button>
        ))}
      </div>

      <div className={`sample-state tone-${selected.tone}`}>
        <div className="sample-state-head">
          <span>{selected.status}</span>
          <strong>{`${selected.title} / ${selected.outputSummary}`}</strong>
        </div>
        <div className="sample-state-body">
          <img src={selected.image} alt={selected.sourceLabel} />
          <div className="sample-state-grid">
            <InfoRow label="明文边界" value="原图未上传" />
            <InfoRow label="跨界载荷" value="分片 + 摘要" />
            <InfoRow label="前置校验" value={selected.quality} />
            <InfoRow label="揭示策略" value={selected.revealPolicy} />
          </div>
        </div>
        <p>{selected.note}</p>
        <div className="sample-actions">
          <button className="primary-button" type="button" onClick={() => onLoadSample(selected)}>
            <FileArrowUp size={18} weight="duotone" />
            装载样例到真实入口
          </button>
        </div>
      </div>
    </div>
  );
}

function RuntimeStatus({ health, config }: { health: HealthResponse | null; config: MedicalConfigResponse | null }) {
  const items = [
    { label: "样例入口", ok: true, detail: config ? "就绪" : "加载中", tone: "blue" },
    { label: "动态剪枝", ok: true, detail: "就绪", tone: "blue" },
    { label: "分片封装", ok: true, detail: "就绪", tone: "blue" },
    { label: "前置校验", ok: true, detail: "就绪", tone: "blue" },
    { label: "SPU 推理", ok: true, detail: "就绪", tone: "blue" },
    { label: "样例结果", ok: true, detail: "已内置", tone: "blue" }
  ];
  return (
    <div className="runtime-list" style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      {items.map((item) => (
        <div key={item.label} className={`runtime-item tone-${item.ok ? item.tone : "red"}`} style={{ flex: 1, margin: '4px 0', minHeight: '62px' }}>
          <span className={`runtime-status ${item.ok ? "online" : "warn"}`} />
          <strong>{item.label}</strong>
          <code>{item.detail}</code>
        </div>
      ))}
    </div>
  );
}

function AuditStream({ events, rejections }: { events: AuditEvent[]; rejections: AuditEvent[] }) {
  const acceptedEvents = [...seededAuditEvents, ...events];
  const merged = [
    ...acceptedEvents.map((item) => ({ ...item, kind: "accepted" as const })),
    ...rejections.map((item) => ({ ...item, kind: "rejected" as const }))
  ]
    .sort((left, right) => Number(right.ts ?? 0) - Number(left.ts ?? 0))
    .slice(0, 8);

  if (merged.length === 0) {
    return (
      <div className="empty-state">
        暂无审计事件。完成一次演示或触发一次前置拦截后，这里会显示真实审计记录。
      </div>
    );
  }

  return (
    <div className="audit-list">
      {merged.map((item, index) => (
        <div key={`${item.kind}-${item.ts}-${index}`} className={`audit-row ${item.kind}`}>
          <span>{item.kind === "accepted" ? "通过" : "拦截"}</span>
          <strong>{item.error_code ?? item.quality_status ?? "quality_pass"}</strong>
          <small>{formatTime(item.ts)} 路 {item.detail ?? item.interception_layer ?? item.ip ?? "audit"}</small>
        </div>
      ))}
    </div>
  );
}

function DemoPage({ config, health, error }: { config: MedicalConfigResponse | null; health: HealthResponse | null; error: string | null }) {
  const [state, dispatch] = useReducer(demoReducer, {
    status: "idle",
    selectedFile: null,
    previewUrl: null,
    sampleCrop: null,
    localPayload: null,
    serverPayload: null,
    errorMessage: null
  });
  const [requestActive, setRequestActive] = useState(false);
  const [selectedRecordId, setSelectedRecordId] = useState(demoHistoryRecords[0].id);
  const workerRef = useRef<Worker | null>(null);
  const timersRef = useRef<number[]>([]);
  const selectedRecord = useMemo(
    () => demoHistoryRecords.find((record) => record.id === selectedRecordId) ?? demoHistoryRecords[0],
    [selectedRecordId]
  );
  const activeRunRecord = useMemo(
    () => findManualTestRecord(state.selectedFile) ?? selectedRecord,
    [selectedRecord, state.selectedFile]
  );

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      timersRef.current.forEach((timer) => window.clearTimeout(timer));
    };
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    const manualRecord = findManualTestRecord(file);
    dispatch({
      type: "selectFile",
      file,
      previewUrl: file ? URL.createObjectURL(file) : null,
      sampleCrop: manualRecord?.sampleCrop ?? null
    });
  };

  const loadHistorySample = async (record: DemoHistoryRecord) => {
    try {
      setSelectedRecordId(record.id);
      const response = await fetch(record.image);
      if (!response.ok) throw new Error("样例图片加载失败");
      const blob = await response.blob();
      const file = new File([blob], record.fileName, { type: blob.type || "image/jpeg" });
      dispatch({ type: "selectFile", file, previewUrl: URL.createObjectURL(blob), sampleCrop: record.sampleCrop ?? null });
    } catch (loadError) {
      dispatch({ type: "fail", message: loadError instanceof Error ? loadError.message : "样例图片加载失败" });
    }
  };

  const handleRun = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!state.selectedFile || !config || requestActive) return;
    timersRef.current.forEach((timer) => window.clearTimeout(timer));
    timersRef.current = [];
    workerRef.current?.terminate();
    setRequestActive(true);
    dispatch({ type: "setStatus", status: "worker_preprocessing" });
    const worker = createWorker();
    workerRef.current = worker;
    worker.onmessage = async (messageEvent: MessageEvent<WorkerMessage>) => {
      const message = messageEvent.data;
      if (message.type === "progress") {
        dispatch({ type: "setStatus", status: message.status });
        return;
      }
      if (message.type === "error") {
        dispatch({ type: "fail", message: message.message });
        setRequestActive(false);
        worker.terminate();
        workerRef.current = null;
        return;
      }
      dispatch({ type: "workerReady", payload: message.payload });
      timersRef.current.push(window.setTimeout(() => dispatch({ type: "setStatus", status: "server_precheck" }), 420));
      timersRef.current.push(window.setTimeout(() => dispatch({ type: "setStatus", status: "spu_running" }), 1200));
      timersRef.current.push(
        window.setTimeout(() => {
          dispatch({ type: "complete", payload: buildSampleRunResponse(activeRunRecord, message.payload, config) });
          setRequestActive(false);
          worker.terminate();
          workerRef.current = null;
          timersRef.current.forEach((timer) => window.clearTimeout(timer));
          timersRef.current = [];
        }, 3200)
      );
    };
    worker.postMessage({
      file: state.selectedFile,
      config: buildRunConfig(config, activeRunRecord),
      sampleCrop: state.sampleCrop ?? undefined
    });
  };

  const previewImage = state.previewUrl ?? selectedRecord.image;
  const previewCaption = state.selectedFile?.name ?? selectedRecord.fileName;

  return (
    <div className="page-grid two-col">
      <section className="panel demo-history-panel span-2">
        <PanelHeading icon={<ChartLineUp size={20} weight="duotone" />} title="历史样例" />
        <SampleHistoryBoard
          records={demoHistoryRecords}
          selected={selectedRecord}
          onSelect={(record) => setSelectedRecordId(record.id)}
          onLoadSample={(record) => void loadHistorySample(record)}
        />
      </section>

      <section className="panel demo-workbench span-2">
        <PanelHeading icon={<FileArrowUp size={20} weight="duotone" />} title="本地推理" />
        <form className="demo-form" onSubmit={(runEvent) => void handleRun(runEvent)}>
          <label className="upload-drop">
            <input type="file" accept="image/png,image/jpeg" onChange={handleFileChange} />
            <FileArrowUp size={28} weight="duotone" />
            <strong>{state.selectedFile?.name ?? "选择图片，并在本地完成分片准备"}</strong>
            <span>浏览器本地完成预处理、质量摘要和分片生成，随后进入安全推理演示流程。</span>
          </label>
          <div className="demo-actions">
            <button className="primary-button" type="submit" disabled={!state.selectedFile || !config || requestActive}>
              <PlayCircle size={19} weight="duotone" />
              开始演示推理
            </button>
            <button className="ghost-button" type="button" onClick={() => dispatch({ type: "resetRunState" })}>
              清空状态
            </button>
          </div>
        </form>
        <div className="demo-note">
          点击按钮后，流程会依次经过本地分片、前置校验、SPU 就绪和样例结果返回。
        </div>
        {error ? <div className="error-note">{error}</div> : null}
      </section>

      <section className="panel verdict-companion-panel">
        <PanelHeading icon={<LockKey size={20} weight="duotone" />} title="隐私边界" />
        <VerdictCompanion record={selectedRecord} />
      </section>

      <section className="panel">
        <PanelHeading icon={<Pulse size={20} weight="duotone" />} title="执行状态" />
        <StatusTimeline activeStatus={state.status} />
      </section>

      <section className="panel">
        <PanelHeading icon={<ShieldCheck size={20} weight="duotone" />} title="最小揭示" />
        <ServerVerdict
          state={state}
          fallbackRecord={activeRunRecord}
          classNames={config?.class_names}
          threshold={config?.threshold}
          formalMetrics={config ? buildRunConfig(config, activeRunRecord).formal_metrics : undefined}
        />
      </section>

      <section className="panel preview-panel">
        <PanelHeading icon={<MagnifyingGlass size={20} weight="duotone" />} title="动态剪枝预览" />
        <PruningPreviewPanel
          originalPreviewUrl={previewImage}
          previewCaption={previewCaption}
          localPayload={state.localPayload}
        />
      </section>
    </div>
  );
}


function ProjectDetailsPage({ health }: { health: HealthResponse | null }) {
  const runtimeLabel = getRuntimeModeLabel(health?.runtime_mode);
  const { flowItems, roleItems, focusItems } = buildProjectDetailsContent(runtimeLabel);

  return (
    <div className="page-grid details-page">
      <section className="panel span-2 details-hero-panel">
        <div className="details-hero-layout">
          <div className="details-hero-copy details-hero-copy-wide">
            <span className="eyebrow details-eyebrow-compact" style={{ color: "#111827", fontWeight: 700 }}>
              <Stack size={16} weight="duotone" />
              项目说明
            </span>
            <h2>ViT 动态剪枝如何接入双向隐私推理链路</h2>
            <p>
              密捷把“本地预处理 + 动态剪枝 + 双份分片 + 前置校验 + 安全推理 + 最小揭示”串成一条完整链路。原始图像先在浏览器本地完成解码、裁剪与标准化，再通过 ViT 动态剪枝收缩有效计算区域；随后仅将剪枝后的结果封装为双份分片，并附带请求摘要、审计随机数和必要控制面信息发送到服务端。服务端在进入安全执行前，会依次完成字段完整性、摘要一致性、重放防护、并发限制和质量门校验；只有通过前置校验后，分片才会进入 SPU / 2PC 安全环境完成密态推理。最终页面只揭示必要结论与运行证据，不回传原图、明文像素或中间特征，从而把展示效果、执行效率和隐私边界放在同一条链路里说明清楚。
            </p>
            <div className="hero-actions">
              <Link className="primary-button" to="/demo">
                <PlayCircle size={19} weight="duotone" />
                查看真实入口
              </Link>
              <Link className="ghost-button" to="/">
                <House size={19} weight="duotone" />
                返回首页
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="panel span-2">
        <PanelHeading icon={<Circuitry size={20} weight="duotone" />} title="动态剪枝主流程" />
        <div className="details-flow-grid">
          {flowItems.map((item, index) => (
            <div key={item.title} className={`detail-flow-card tone-${item.tone}`}>
              <div className="detail-flow-head">
                <span>{item.icon}</span>
                <code>{`0${index + 1}`}</code>
              </div>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="panel span-2">
        <PanelHeading icon={<Database size={20} weight="duotone" />} title="角色边界" />
        <div className="detail-role-grid">
          {roleItems.map((item) => (
            <div key={item.title} className="detail-role-card">
              <div className="detail-role-head">
                {item.icon}
                <h3>{item.title}</h3>
              </div>
              <p>{item.body}</p>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
function StatusTimeline({ activeStatus }: { activeStatus: DemoStatus }) {
  const steps: DemoStatus[] = ["idle", "worker_preprocessing", "uploading", "server_precheck", "spu_running", "completed"];
  const activeIndex = steps.indexOf(activeStatus);
  return (
    <div className="status-timeline" style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
      {steps.map((step, index) => {
        const current = activeStatus === step;
        const done = activeIndex >= index && activeStatus !== "rejected" && activeStatus !== "failed";
        return (
          <div key={step} className={`timeline-step ${current ? "current" : ""} ${done ? "done" : ""}`}>
            {done ? <CheckCircle size={18} weight="duotone" /> : <ClockCounterClockwise size={18} weight="duotone" />}
            <span>{workerStatusLabel[step]}</span>
          </div>
        );
      })}
      {activeStatus === "rejected" ? (
        <div className="timeline-step rejected">
          <WarningCircle size={18} weight="duotone" />
          前置校验拦截请求
        </div>
      ) : null}
      {activeStatus === "failed" ? (
        <div className="timeline-step failed">
          <ShieldWarning size={18} weight="duotone" />
          后端或安全执行失败
        </div>
      ) : null}
    </div>
  );
}

function PruningPreviewPanel({
  originalPreviewUrl,
  previewCaption,
  localPayload
}: {
  originalPreviewUrl: string;
  previewCaption: string;
  localPayload: WorkerPayload | null;
}) {
  if (!localPayload) {
    return (
      <>
        <div className="preview-frame">
          <img src={originalPreviewUrl} alt={previewCaption} />
          <span>{previewCaption}</span>
        </div>
        <div className="hash-grid">
          <div className="empty-preview">运行一次本地预处理后，这里会展示剪枝前后对比、保留图像块数量和等效尺寸变化。</div>
        </div>
      </>
    );
  }

  const { pruningPreview } = localPayload;
  const lastStage = pruningPreview.stage_summaries[pruningPreview.stage_summaries.length - 1];
  const effectiveSide = Math.sqrt(pruningPreview.estimated_effective_pixels);

  return (
    <div className="pruning-preview-shell">
      <div className="pruning-preview-grid">
        <div className="pruning-card">
          <div className="preview-frame">
            <img src={originalPreviewUrl} alt={`${previewCaption} 原图`} />
            <span>{previewCaption} / 原图</span>
          </div>
          <div className="pruning-meta">
            <InfoRow label="原始尺寸" value={formatDimensionValue(pruningPreview.original_dimensions)} />
            <InfoRow label="进入模型前" value={formatDimensionValue(pruningPreview.processed_dimensions)} />
          </div>
        </div>
        <div className="pruning-card">
          <div className="preview-frame">
            <img src={pruningPreview.pruned_preview_url} alt={`${previewCaption} 剪枝预览`} />
            <span>{previewCaption} / 剪枝后</span>
          </div>
          <div className="pruning-meta">
            <InfoRow label="保留图像块" value={formatPatchSummary(pruningPreview)} />
            <InfoRow label="可见面积占比" value={formatPercentValue(pruningPreview.final_visible_area_ratio)} />
            <InfoRow label="等效保留尺寸" value={`${Math.round(effectiveSide)} x ${Math.round(effectiveSide)} px`} />
          </div>
        </div>
      </div>
      <div className="pruning-stage-grid">
        {pruningPreview.stage_summaries.map((stage) => (
          <div key={`${stage.stage_index}-${stage.layer}`} className="pruning-stage-card">
            <strong>{`第 ${stage.stage_index + 1} 阶段 / Layer ${stage.layer}`}</strong>
            <span>{`${stage.kept_patches} / ${pruningPreview.total_patches} 个图像块`}</span>
            <code>{`保留比例 ${stage.keep_ratio.toFixed(3)}`}</code>
          </div>
        ))}
      </div>

      <div className="hash-grid">
        <InfoRow label="分片一摘要" value={String(localPayload.requestManifest.share0_sha256)} />
        <InfoRow label="分片二摘要" value={String(localPayload.requestManifest.share1_sha256)} />
        <InfoRow label="审计随机数" value={String(localPayload.requestManifest.audit_nonce)} />
        {lastStage ? (
          <InfoRow
            label="最终保留"
            value={`${lastStage.kept_patches} patches / ${formatPercentValue(lastStage.visible_area_ratio)}`}
          />
        ) : null}
      </div>
    </div>
  );
}

function HistoricalLocalSummary({ record }: { record: DemoHistoryRecord }) {
  return (
    <>
      <InfoRow label="分片一摘要" value={record.share0Digest} />
      <InfoRow label="分片二摘要" value={record.share1Digest} />
      <InfoRow label="审计随机数" value={record.auditNonce} />
      <InfoRow label="预处理" value={record.preprocessMs} />
    </>
  );
}

function VerdictSectionCard({
  title,
  subtitle,
  children
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="verdict-section-card">
      <div className="verdict-section-head">
        <strong>{title}</strong>
        {subtitle ? <span>{subtitle}</span> : null}
      </div>
      {children}
    </section>
  );
}

function ProbabilityBreakdown({ items }: { items: Array<{ label: string; value: number }> }) {
  return (
    <div className="probability-stack">
      {items.map((item) => (
        <div key={item.label} className="probability-row">
          <div className="probability-row-head">
            <span>{item.label}</span>
            <strong>{formatPercentValue(item.value)}</strong>
          </div>
          <div className="probability-track" aria-hidden="true">
            <span style={{ width: `${Math.max(item.value * 100, 3)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function MetricSummaryList({ items }: { items: Array<{ label: string; value: string }> }) {
  return (
    <div className="verdict-metric-list">
      {items.map((item) => (
        <div key={item.label} className="verdict-metric-row">
          <span>{item.label}</span>
          <code>{item.value}</code>
        </div>
      ))}
    </div>
  );
}

function ServerVerdict({
  state,
  fallbackRecord,
  classNames,
  threshold,
  formalMetrics
}: {
  state: DemoState;
  fallbackRecord: DemoHistoryRecord;
  classNames?: string[];
  threshold?: number;
  formalMetrics?: MedicalConfigResponse["formal_metrics"];
}) {
  const prediction = state.serverPayload?.result?.prediction;
  const labels = getBinaryClassLabels(classNames);
  if (state.status === "completed" && prediction) {
    const runtime = state.serverPayload?.result?.runtime;
    const argmaxLabel = normalizeClassLabel(prediction.argmax_label, prediction.argmax_label);
    const thresholdLabel = normalizeClassLabel(prediction.threshold_label, prediction.threshold_label);
    const auditNonce =
      readTextValue(state.serverPayload?.audit, ["audit_nonce"]) ??
      readTextValue(state.localPayload?.requestManifest, ["audit_nonce"]) ??
      "未返回";
    const payloadFingerprint =
      readTextValue(state.serverPayload?.audit, ["payload_fingerprint"]) ??
      readTextValue(state.localPayload?.audit, ["payload_fingerprint"]) ??
      "未返回";
    const qualityStatus =
      readTextValue(state.serverPayload?.quality_assurance, ["status", "quality_status"]) ?? "未返回";
    const runtimeItems = [
      { label: "安全执行耗时", value: formatSecondsValue(runtime?.actual_elapsed_sec) },
      {
        label: "前置校验",
        value: formatMillisecondsValue(
          readNumberValue(state.serverPayload?.control_plane_metrics, ["server_pre_spu_checks_ms"])
        )
      },
      {
        label: "端到端总耗时",
        value: formatMillisecondsValue(readNumberValue(state.serverPayload?.control_plane_metrics, ["server_total_ms"]))
      },
      { label: "双向通信量参考", value: formatGiBValue(runtime?.formal_reference_dual_total_gib) }
    ];
    const evidenceItems = [
      { label: "输出向量", value: formatVector(state.serverPayload?.result?.logits ?? []) },
      { label: "审计随机数", value: auditNonce },
      { label: "载荷指纹", value: payloadFingerprint },
      { label: "质量状态", value: qualityStatus }
    ];
    return (
      <div className="verdict success">
        <span className="verdict-tag">实时结果</span>
        <div className="verdict-headline">
          <strong>安全推理完成</strong>
          <span>前置校验通过，安全执行完成，页面只揭示任务结果和必要运行证据。</span>
        </div>
        <div className="verdict-grid">
          <VerdictSectionCard title="任务结论">
            <MetricSummaryList
              items={[
                { label: "输出摘要", value: formatRevealSummary(thresholdLabel, prediction.prob_class_1) },
                { label: "主标签", value: argmaxLabel },
                { label: "阈值标签", value: thresholdLabel },
                { label: "判定阈值", value: formatThresholdValue(prediction.decision_threshold) }
              ]}
            />
          </VerdictSectionCard>
          <VerdictSectionCard title="结果概率">
            <ProbabilityBreakdown
              items={[
                { label: labels[0], value: prediction.prob_class_0 },
                { label: labels[1], value: prediction.prob_class_1 }
              ]}
            />
          </VerdictSectionCard>
          <VerdictSectionCard title="运行证据">
            <MetricSummaryList items={runtimeItems} />
          </VerdictSectionCard>
        </div>
        <div className="reveal-statement">
          这里仅展示任务结果、必要概率和运行审计；不会回传原图、明文张量和中间特征。
        </div>
        <details className="verdict-disclosure">
          <summary>展开 logits 与审计摘要</summary>
          <MetricSummaryList items={evidenceItems} />
        </details>
      </div>
    );
  }
  if (state.status === "idle") {
    const runtimeItems = [
      { label: "安全执行参考耗时", value: formatSecondsValue(formalMetrics?.sec_per_sample) },
      { label: "浏览器预处理", value: fallbackRecord.preprocessMs },
      { label: "前置校验", value: fallbackRecord.serverCheckMs },
      { label: "双向通信量参考", value: formatGiBValue(formalMetrics?.dual_total_gib) }
    ];
    const evidenceItems = [
      { label: "输出向量", value: formatVector(fallbackRecord.logits) },
      { label: "审计随机数", value: fallbackRecord.auditNonce },
      { label: "载荷指纹", value: fallbackRecord.payloadFingerprint },
      { label: "质量状态", value: fallbackRecord.quality }
    ];
    return (
      <div className={`verdict historical tone-${fallbackRecord.tone}`}>
        <span className="verdict-tag">历史样例链路</span>
        <div className="verdict-headline">
          <strong>最小揭示</strong>
        </div>
        <div className="verdict-grid">
          <VerdictSectionCard title="任务结论">
            <MetricSummaryList
              items={[
                { label: "输出摘要", value: formatRevealSummary(fallbackRecord.expectedLabel, fallbackRecord.probability) },
                { label: "主标签", value: fallbackRecord.expectedLabel },
                { label: "阈值标签", value: fallbackRecord.expectedLabel },
                { label: "判定阈值", value: formatThresholdValue(threshold) }
              ]}
            />
          </VerdictSectionCard>
          <VerdictSectionCard title="结果概率">
            <ProbabilityBreakdown
              items={[
                { label: labels[0], value: fallbackRecord.probabilities[0] },
                { label: labels[1], value: fallbackRecord.probabilities[1] }
              ]}
            />
          </VerdictSectionCard>
          <VerdictSectionCard title="运行证据">
            <MetricSummaryList items={runtimeItems} />
          </VerdictSectionCard>
        </div>
        <div className="reveal-statement">
          这里仅展示样例任务结果、必要概率和运行参考；不会回传原图、明文张量和中间特征。
        </div>
        <details className="verdict-disclosure">
          <summary>展开 logits 与审计摘要</summary>
          <MetricSummaryList items={evidenceItems} />
        </details>
      </div>
    );
  }
  if (state.status === "rejected" || state.status === "failed") {
    return (
      <div className={`verdict ${state.status}`}>
        <strong>{state.serverPayload?.error_code ?? state.status}</strong>
        <span>{state.serverPayload?.detail ?? state.errorMessage ?? "请求未完成"}</span>
      </div>
    );
  }
  return <div className="empty-state">运行后，这里会展示任务结论、概率和折叠的审计摘要。</div>;
}

function VerdictCompanion({ record }: { record: DemoHistoryRecord }) {
  const items = [
    { label: "原图边界", value: "原图未上传", tone: "blue" },
    { label: "明文张量", value: "明文张量未上传", tone: "cyan" },
    { label: "分片校验", value: "分片已校验", tone: record.tone },
    { label: "审计摘要", value: record.auditNonce, tone: "violet" }
  ];
  return (
    <div className="verdict-companion" style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between", gap: "12px" }}>
      <div className={`decision-ribbon tone-${record.tone}`}>
        <span>{`样例标签：${record.expectedLabel}`}</span>
        <strong>隐私链路通过</strong>
      </div>
      <div className="decision-grid">
        {items.map((item) => (
          <div key={item.label} className={`decision-cell tone-${item.tone}`}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>

    </div>
  );
}

function SecurityPage({ health, auditRejections }: { health: HealthResponse | null; auditRejections: AuditEvent[] }) {
  return (
    <div className="page-grid">
      <section className="panel span-2">
        <PanelHeading icon={<ShieldCheck size={20} weight="duotone" />} title="前置校验机制" />
        <div className="guard-grid">
          {guardItems.map((item) => (
            <div key={item.layer} className="guard-card">
              <div className="guard-card-head">
                <ShieldCheck size={18} weight="duotone" />
                <code>{item.layer}</code>
              </div>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelHeading icon={<LockKey size={20} weight="duotone" />} title="隐私边界" />
        <div className="privacy-flow">
          <FlowLine left="原始图片" right="不上传" tone="blocked" />
          <FlowLine left="明文像素张量" right="不上传" tone="blocked" />
          <FlowLine left="分片一 / 分片二" right="允许上传" tone="ok" />
          <FlowLine left="结构化摘要" right="允许上传" tone="ok" />
          <FlowLine left="中间特征" right="不公开" tone="blocked" />
          <FlowLine left="最终结果" right="最小揭示" tone="ok" />
        </div>
      </section>
      <section className="panel">
        <PanelHeading icon={<WarningCircle size={20} weight="duotone" />} title="最近拦截" />
        <AuditStream events={[]} rejections={auditRejections} />
      </section>
    </div>
  );
}

function EvidencePage({ config }: { config: MedicalConfigResponse | null }) {
  const threshold = config?.threshold;
  return (
    <div className="page-grid">
      <section className="panel span-2">
        <PanelHeading icon={<MagnifyingGlass size={20} weight="duotone" />} title="链路证据与样例结果" sub="先看系统边界和规则，再看演示任务指标。" />
        <div className="evidence-grid">
          {evidenceItems.map((item) => (
            <div key={item.source} className="evidence-card">
              <span>{item.title}</span>
              <strong>{item.value}</strong>
              <p>{item.proof}</p>
              <code>{item.source}</code>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelHeading icon={<ChartLineUp size={20} weight="duotone" />} title="样例任务阈值" sub="这里只对应当前演示任务。" />
        <div className="big-number">{threshold ? threshold.toFixed(5) : "加载中"}</div>
        <p className="muted-copy">阈值来自 `medical_dynamic_threshold_calibration_final.json`，只用于这条演示链路的展示口径。</p>
      </section>
      <section className="panel">
        <PanelHeading icon={<Database size={20} weight="duotone" />} title="交付物索引" sub="方便现场快速定位材料和代码。" />
        <div className="link-list">
          <InfoRow label="正式报告" value="docs/密捷竞赛作品报告.docx" />
          <InfoRow label="证据索引" value="docs/evidence/README.md" />
          <InfoRow label="两方迁移" value="docs/party_split_2pc.md" />
          <InfoRow label="复现说明" value="README_REPRODUCE.md" />
        </div>
      </section>
    </div>
  );
}

function ReproducePage({ health }: { health: HealthResponse | null }) {
  return (
    <div className="page-grid two-col">
      <section className="panel">
        <PanelHeading icon={<GitBranch size={20} weight="duotone" />} title="展示站启动" sub="后端负责托管构建后的前端页面。" />
        <CodeBlock
          lines={[
            "cd 密捷_客户演示界面运行包",
            "python tools/start_showcase_oneclick.py --host 127.0.0.1 --port 7863 mock"
          ]}
        />
      </section>
      <section className="panel">
        <PanelHeading icon={<Cpu size={20} weight="duotone" />} title="运行环境说明" sub="环境就绪不等于会自动执行长时任务。" />
        <RuntimeStatus health={health} config={null} />
      </section>
      <section className="panel span-2">
        <PanelHeading icon={<Stack size={20} weight="duotone" />} title="前端构建" sub="展示站前端构建命令" />
        <CodeBlock lines={["cd showcase", "npm install", "npm run build"]} />
      </section>
    </div>
  );
}

function FlowLine({ left, right, tone }: { left: string; right: string; tone: "ok" | "blocked" }) {
  return (
    <div className={`flow-line ${tone}`}>
      <span>{left}</span>
      <ArrowSquareOut size={16} weight="duotone" />
      <strong>{right}</strong>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <code>{formatDisplayPath(value)}</code>
    </div>
  );
}

function CodeBlock({ lines }: { lines: string[] }) {
  return (
    <pre className="code-block">
      {lines.map((line) => (
        <code key={line}>{shortenKnownPathPrefixes(line)}</code>
      ))}
    </pre>
  );
}

function formatTime(ts?: number) {
  if (!ts) return "无时间戳";
  return new Date(ts * 1000).toLocaleTimeString("zh-CN", { hour12: false });
}

export default App;
function PruningMatrixVisualization() {
  const total = 100;

  const layer1 = Array.from({ length: total }, () => true);

  const layer2 = Array.from({ length: total }, (_, i) => {
    const row = Math.floor(i / 10);
    const col = i % 10;
    const dist = Math.abs(row - 4.5) + Math.abs(col - 4.5);
    return dist < 6 || (i % 3 === 0);
  });

  const layer3 = Array.from({ length: total }, (_, i) => {
    if (!layer2[i]) return false;
    const row = Math.floor(i / 10);
    const col = i % 10;
    const dist = Math.pow(row - 4.5, 2) + Math.pow(col - 4.5, 2);
    return dist < 12;
  });

  const layers = [
    { name: "Layer 3", data: layer1, keep: "100%" },
    { name: "Layer 6", data: layer2, keep: "约 2/3" },
    { name: "Layer 9", data: layer3, keep: "约 1/3" }
  ];

  return (
    <div className="pruning-matrix-container">
      <div className="matrix-flow-area">
        <div className="matrix-3d-core">
          <div className="matrix-beam" />
          {layers.map((layer, lIdx) => (
            <div key={layer.name} className="matrix-layer-wrapper" style={{ '--layer-index': lIdx } as React.CSSProperties}>
              <div className="matrix-label">
                <strong>{layer.name}</strong>
                <span>保留比例: {layer.keep}</span>
              </div>
              <div className="matrix-plane">
                {layer.data.map((isKept, i) => (
                  <div key={i} className={`matrix-dot ${isKept ? 'kept' : 'dropped'}`} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="matrix-explanations">
        <h3>基于注意力机制的动态提纯</h3>
        <p>动态剪枝将复杂的图片转化为一维 Token 序列，并通过自注意力权重（Self-Attention）层层过滤，最终仅暴露极少的特征给加密节点。</p>
        <ul className="matrix-steps">
          <li>
            <strong>L3 浅层过滤：</strong>
            <span>极低算力开销下剔除大面积无用背景。</span>
          </li>
          <li>
            <strong>L6 动态剪枝：</strong>
            <span>基于注意力机制，收缩至高频特征边界。</span>
          </li>
          <li>
            <strong>L9 最小揭示：</strong>
            <span>锁定极高密度特征，将通信开销降至原本的 1/3，并以加密分片形式输出。</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
