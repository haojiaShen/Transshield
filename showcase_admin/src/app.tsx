import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

type ViewKey = "overview" | "jobs" | "models" | "results" | "system";

type AdminSession = {
  username: string;
  display_name: string;
  created_at: number;
  expires_at: number;
};

type QueueStats = Record<string, number>;

type MetricsSummary = {
  argmaxAccuracy?: number | null;
  bestThreshold?: number | null;
  bestThresholdAccuracy?: number | null;
  auc?: number | null;
  sampleCount?: number | null;
  argmax_accuracy?: number | null;
  best_threshold?: number | null;
  best_threshold_accuracy?: number | null;
  sample_count?: number | null;
};

type TrainingJobRecord = {
  job_id: string;
  name: string;
  preset_id?: string | null;
  mode: string;
  status: string;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  current_step?: string | null;
  step_index?: number | null;
  command?: string | null;
  command_sequence?: string[];
  output_dir?: string | null;
  bundle_dir?: string | null;
  log_paths?: { stdout?: string; stderr?: string };
  metrics_summary?: MetricsSummary | null;
  artifacts?: Record<string, string | null>;
  error_message?: string | null;
  cancel_requested?: boolean;
  readonly?: boolean;
  source?: string;
};

type TrainingJobLog = {
  job_id: string;
  stdout: string;
  stderr: string;
};

type ModelRecord = {
  id: string;
  name: string;
  domain: string;
  bundle_name: string;
  bundle_dir: string;
  source_run?: string;
  dataset_path?: string;
  status: string;
  base_rate?: number | null;
  secure_static_train_depth?: number | null;
  cls_distill_weight?: number | null;
  token_distill_weight?: number | null;
  threshold_accuracy?: number | null;
  argmax_accuracy?: number | null;
  auc?: number | null;
  teacher_checkpoint_path?: string;
  manifest_path?: string;
  args_snapshot_path?: string;
  threshold_path?: string;
};

type ResultSection = {
  label: string;
  path: string;
  payload: Record<string, unknown>;
};

type ResultsCatalog = {
  formal_metrics: {
    threshold_accuracy: number;
    auc: number;
    sec_per_sample: number;
    dual_total_gib: number;
    threshold: number;
    bundle_dir: string;
  };
  sections: ResultSection[];
};

type TrainingPreset = {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, string | number | boolean>;
};

type TrainingSystemConfig = {
  repo_root: string;
  python_bin: string;
  job_root: string;
  train_output_root: string;
  bundle_output_root: string;
  default_train_data_path: string;
  default_eval_data_path: string;
  default_device: string;
  default_batch_size: number;
  default_num_workers: number;
  max_concurrent_train_jobs: number;
  runtime_mode: string;
  admin_display_name: string;
  training_presets: TrainingPreset[];
};

type OverviewPayload = {
  queue: QueueStats;
  recent_completed: TrainingJobRecord[];
  formal_model?: ModelRecord | null;
  gpu: {
    available: boolean;
    message: string;
    devices: Array<{
      index: string;
      name: string;
      memory_total_mib: string;
      utilization_gpu_percent: string;
    }>;
  };
  environment: {
    python_bin: string;
    python_version: string;
    repo_root: string;
    train_output_root: string;
    bundle_output_root: string;
    job_root: string;
    default_train_data_path: string;
    default_eval_data_path: string;
    max_concurrent_train_jobs: number;
    runtime_mode: string;
  };
};

type HealthPayload = {
  status: string;
  runtime_mode?: string;
  bundle_present?: boolean;
  spu_config_present?: boolean;
  runner_present?: boolean;
  dist_present?: boolean;
  admin_job_root_present?: boolean;
};

type ConnectionPreset = {
  id: string;
  label: string;
  apiBase: string;
  note: string;
};

const STORAGE_API_BASE = "mijie-admin-api-base";
const STORAGE_TOKEN = "mijie-admin-session-token";
const DEFAULT_API_BASE = "http://127.0.0.1:7863";
const ACTIVE_JOB_STATUSES = new Set(["queued", "starting", "running", "postprocessing"]);

function getCompanionApiBase(): string {
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return `${window.location.protocol}//${window.location.hostname}:7863`;
  }
  return DEFAULT_API_BASE;
}

const connectionPresets: ConnectionPreset[] = [
  {
    id: "local",
    label: "本机训练服务",
    apiBase: "http://127.0.0.1:7863",
    note: "适合本地联调与现场展示。"
  },
  {
    id: "remote",
    label: "远端训练服务器",
    apiBase: "http://10.204.68.253:7863",
    note: "用于切换到服务器侧训练环境。"
  }
];

const viewLabels: Record<ViewKey, string> = {
  overview: "总览",
  jobs: "训练任务",
  models: "已训练模型",
  results: "结果证据",
  system: "环境设置"
};

const viewIndexes: Record<ViewKey, string> = {
  overview: "01",
  jobs: "02",
  models: "03",
  results: "04",
  system: "05"
};

const jobStatusLabels: Record<string, string> = {
  queued: "待排队",
  starting: "启动中",
  running: "训练中",
  postprocessing: "整理结果",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消"
};

const presetNames: Record<string, string> = {
  medical_dynamic_mainline: "医疗动态主线",
  secure_static_depth12_control: "静态部署对齐控制线",
  secure_static_depth12_aanone: "静态 uniform/fixed-square 复现实验"
};

const modelNames: Record<string, string> = {
  frozen_bundle_medical_dynamic_mainline: "医疗主线 bundle",
  frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_20260430: "静态安全对照 bundle",
  frozen_bundle_secure_static_depth12_uniform_fixed_square_epoch8_aanone_20260507: "静态复现实验 bundle",
  frozen_bundle_finance_fraud_v3_20260511: "金融欺诈 bundle",
  frozen_bundle_finance_lrd_rank192_20260515: "金融风险排序 bundle"
};

const resultLabels: Record<string, string> = {
  "medical_dynamic_threshold_calibration_final.json": "正式阈值校准",
  "medical_dynamic_auc_reference_final.json": "正式 AUC 参考",
  "mainline_communication_profile_final.json": "通信量剖面",
  "distill_compensation_pair_compare.json": "蒸馏补偿配对实验",
  "secure_static_train_depth_pair_compare.json": "部署对齐控制实验",
  "protocol_fuzz_final.json": "协议层模糊测试",
  "guard_stress_final.json": "控制面准入压力测试"
};

const editableFields: Array<{
  key: string;
  label: string;
  type: "text" | "number" | "boolean";
  step?: string;
}> = [
  { key: "train_data_path", label: "训练集路径", type: "text" },
  { key: "eval_data_path", label: "验证集路径", type: "text" },
  { key: "epochs", label: "训练轮次", type: "number", step: "1" },
  { key: "batch_size", label: "批大小", type: "number", step: "1" },
  { key: "num_workers", label: "数据线程", type: "number", step: "1" },
  { key: "device", label: "训练设备", type: "text" },
  { key: "base_rate", label: "基础保留率", type: "number", step: "0.1" },
  { key: "secure_static_train_depth", label: "secure_static_train_depth", type: "number", step: "1" },
  { key: "cls_distill_weight", label: "分类蒸馏权重", type: "number", step: "0.01" },
  { key: "token_distill_weight", label: "token 蒸馏权重", type: "number", step: "0.01" },
  { key: "use_mask_pruning", label: "启用动态剪枝", type: "boolean" },
  { key: "inference_friendly_ops", label: "启用安全友好算子", type: "boolean" },
  { key: "export_bundle", label: "完成后导出 bundle", type: "boolean" },
  { key: "finetune", label: "finetune 检查点", type: "text" },
  { key: "teacher_checkpoint_path", label: "教师模型检查点", type: "text" }
];

function normalizeApiBase(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

function formatPercent(raw?: number | null, digits = 2): string {
  if (raw === undefined || raw === null || Number.isNaN(raw)) {
    return "—";
  }
  const value = raw <= 1 ? raw * 100 : raw;
  return `${value.toFixed(digits)}%`;
}

function formatNumber(raw?: number | null, digits = 4): string {
  if (raw === undefined || raw === null || Number.isNaN(raw)) {
    return "—";
  }
  return raw.toFixed(digits);
}

function formatShortNumber(raw?: number | null, digits = 2): string {
  if (raw === undefined || raw === null || Number.isNaN(raw)) {
    return "—";
  }
  return raw.toFixed(digits);
}

function formatDateTime(raw?: string | null): string {
  if (!raw) {
    return "—";
  }
  return raw.replace("T", " ");
}

function statusLabel(status: string): string {
  return jobStatusLabels[status] ?? status;
}

function presetLabel(presetId?: string | null): string {
  if (!presetId) {
    return "自定义任务";
  }
  return presetNames[presetId] ?? presetId;
}

const PROJECT_ROOT_MARKERS = [
  "密捷_管理员控制台运行包",
  "密捷_客户演示界面运行包",
  "Transshield_final",
  "Transshield",
  "源代码·"
];

const LEGACY_PROJECT_PATH_ALIASES: Array<[string, string]> = [
  [
    "/data/wyb/pneumoniamnist_imagefolder_subset/train",
    "data/pneumoniamnist_imagefolder_subset/train"
  ],
  [
    "/data/wyb/pneumoniamnist_imagefolder_subset/val",
    "data/pneumoniamnist_imagefolder_subset/val"
  ],
  [
    "/data/wyb/pneumoniamnist_imagefolder_subset/test",
    "data/pneumoniamnist_imagefolder_subset/test"
  ]
];

const PROJECT_PATH_ROOTS = [
  "archive",
  "artifacts",
  "configs",
  "data",
  "docs",
  "integrations",
  "licenses",
  "logs",
  "models",
  "results",
  "scripts",
  "showcase",
  "showcase_admin",
  "showcase_api",
  "spu_vendored",
  "tests",
  "tmp",
  "tools",
  "training_core",
  "training_compat"
];

const PATH_PARAMETER_KEYS = new Set([
  "train_data_path",
  "eval_data_path",
  "finetune",
  "teacher_checkpoint_path"
]);

function findProjectPathStart(text: string): number {
  let matchedIndex = -1;
  for (const root of PROJECT_PATH_ROOTS) {
    let index = text.indexOf(`${root}/`);
    while (index >= 0) {
      if (index === 0) {
        if (matchedIndex < 0 || index < matchedIndex) {
          matchedIndex = index;
        }
        break;
      }
      index = text.indexOf(`${root}/`, index + root.length + 1);
    }
  }
  return matchedIndex;
}

function toProjectRelativePath(value: string): string {
  let text = value.trim().replaceAll("\\", "/");
  if (!text || text === "—") return text || "—";
  if (/^(?:https?|wss?):\/\//i.test(text)) return text;
  if (/(?:^|\/)python(?:\d+(?:\.\d+)*)?(?:\.exe)?$/i.test(text)) return "python";

  for (const [legacyPrefix, relativePrefix] of LEGACY_PROJECT_PATH_ALIASES) {
    if (text === legacyPrefix || text.startsWith(`${legacyPrefix}/`)) {
      return `${relativePrefix}${text.slice(legacyPrefix.length)}`;
    }
  }

  for (const marker of PROJECT_ROOT_MARKERS) {
    const markerToken = `${marker}/`;
    const markerIndex = text.indexOf(markerToken);
    if (markerIndex >= 0) {
      text = text.slice(markerIndex + markerToken.length);
      break;
    }
    if (text.endsWith(marker)) return ".";
  }

  const projectPathStart = findProjectPathStart(text);
  if (projectPathStart >= 0) {
    return text.slice(projectPathStart).replace(/^\.\/+/, "");
  }

  if (/^(?:[A-Za-z]:\/|\/)/.test(text)) {
    const segments = text.split("/").filter(Boolean);
    return segments.length > 0 ? `./${segments[segments.length - 1]}` : ".";
  }

  return text.replace(/^\.\/+/, "") || ".";
}

function sanitizeDisplayText(value: string): string {
  let text = value.replaceAll("\\", "/");
  for (const [legacyPrefix, relativePrefix] of LEGACY_PROJECT_PATH_ALIASES) {
    text = text.replaceAll(legacyPrefix, relativePrefix);
  }
  for (const marker of PROJECT_ROOT_MARKERS) {
    const markerToken = `${marker}/`;
    let markerIndex = text.indexOf(markerToken);
    while (markerIndex >= 0) {
      let tokenStart = markerIndex;
      while (tokenStart > 0 && !/[\s"'=([{]/.test(text[tokenStart - 1])) {
        tokenStart -= 1;
      }
      text = text.slice(0, tokenStart) + text.slice(markerIndex + markerToken.length);
      markerIndex = text.indexOf(markerToken);
    }
  }

  const rootsPattern = PROJECT_PATH_ROOTS.join("|");
  const absolutePrefix = new RegExp(`(?:[A-Za-z]:)?/(?:[^\\s"'=<>]+/)*?(?=(?:${rootsPattern})/)`, "g");
  return text.replace(absolutePrefix, "");
}

function formatDisplayPath(value?: string | null): string {
  if (!value) return "—";
  return toProjectRelativePath(String(value)) || "—";
}

function formatDisplayJson(value: unknown): unknown {
  if (typeof value === "string") {
    return sanitizeDisplayText(value);
  }
  if (Array.isArray(value)) {
    return value.map(formatDisplayJson);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, formatDisplayJson(item)]));
  }
  return value;
}

function normalizePathParameters(
  parameters: Record<string, string | number | boolean>
): Record<string, string | number | boolean> {
  return Object.fromEntries(
    Object.entries(parameters).map(([key, value]) => [
      key,
      PATH_PARAMETER_KEYS.has(key) && typeof value === "string" ? toProjectRelativePath(value) : value
    ])
  );
}

function friendlyModelName(model: ModelRecord): string {
  return modelNames[model.id] ?? model.name ?? model.id;
}

function friendlyJobName(job: TrainingJobRecord): string {
  if (job.source === "console" && job.preset_id && presetNames[job.preset_id]) {
    return `${presetNames[job.preset_id]} / ${job.name}`;
  }
  return job.name || job.job_id;
}

function friendlySectionLabel(section: ResultSection): string {
  const fileName = section.path.split(/[/\\]/).pop() || "";
  return resultLabels[fileName] ?? section.label ?? fileName;
}

function summarizeSection(section: ResultSection): Array<{ key: string; value: string }> {
  const payload = section.payload as Record<string, unknown>;
  const items: Array<{ key: string; value: string }> = [];

  if (typeof payload.sample_count === "number") {
    items.push({ key: "样本数", value: String(payload.sample_count) });
  }
  if (typeof payload.best_threshold_accuracy === "number") {
    items.push({ key: "阈值精度", value: formatPercent(payload.best_threshold_accuracy as number) });
  }
  if (typeof payload.argmax_accuracy === "number") {
    items.push({ key: "argmax 精度", value: formatPercent(payload.argmax_accuracy as number) });
  }
  if (typeof payload.auc === "number") {
    items.push({ key: "AUC", value: formatNumber(payload.auc as number) });
  }
  if (typeof payload.best_threshold === "number") {
    items.push({ key: "最佳阈值", value: formatNumber(payload.best_threshold as number) });
  }

  if (payload.medical && typeof payload.medical === "object") {
    const medical = payload.medical as Record<string, unknown>;
    if (typeof medical.sec_per_sample === "number") {
      items.push({ key: "秒/样本", value: formatShortNumber(medical.sec_per_sample as number, 2) });
    }
    if (typeof medical.dual_total_gib === "number") {
      items.push({ key: "总通信量", value: `${formatShortNumber(medical.dual_total_gib as number, 2)} GiB` });
    }
  }

  const validationRows = Array.isArray(payload.results)
    ? payload.results
    : Array.isArray(payload.checks)
      ? payload.checks
      : null;
  if (validationRows) {
    const passedCount = validationRows.filter(
      (row) => row && typeof row === "object" && (row as Record<string, unknown>).passed === true
    ).length;
    items.push({ key: "验证项", value: String(validationRows.length) });
    items.push({ key: "已通过", value: String(passedCount) });
    items.push({ key: "未通过", value: String(validationRows.length - passedCount) });
    if (typeof payload.passed === "boolean") {
      items.push({ key: "总体结论", value: payload.passed ? "全部通过" : "存在未通过项" });
    }
  }

  if (items.length === 0) {
    items.push({ key: "顶层字段", value: String(Object.keys(payload).length) });
    items.push({ key: "证据格式", value: "结构化 JSON" });
  }

  return items.slice(0, 5);
}

function evidenceVisualPercent(item: { key: string; value: string }): number | null {
  const numeric = Number.parseFloat(item.value.replace("%", ""));
  if (!Number.isFinite(numeric)) {
    return null;
  }
  if (item.value.trim().endsWith("%")) {
    return Math.max(0, Math.min(100, numeric));
  }
  if (item.key === "AUC" && numeric >= 0 && numeric <= 1) {
    return numeric * 100;
  }
  return null;
}

async function requestJson<T>(
  apiBase: string,
  path: string,
  options: { method?: string; token?: string; body?: unknown } = {}
): Promise<T> {
  const headers: Record<string, string> = {};
  if (options.token) {
    headers["x-admin-session"] = options.token;
  }
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${normalizeApiBase(apiBase)}${path}`, {
    method: options.method ?? "GET",
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined
  });

  let payload: Record<string, unknown> = {};
  try {
    payload = (await response.json()) as Record<string, unknown>;
  } catch {
    payload = {};
  }

  if (!response.ok) {
    throw new Error(String(payload.detail ?? payload.message ?? `HTTP ${response.status}`));
  }

  return payload as T;
}

function App() {
  const [apiBase, setApiBase] = useState(() =>
    normalizeApiBase(localStorage.getItem(STORAGE_API_BASE) || getCompanionApiBase())
  );
  const [token, setToken] = useState(() => localStorage.getItem(STORAGE_TOKEN) || "");
  const [session, setSession] = useState<AdminSession | null>(null);
  const [booting, setBooting] = useState(true);
  const [activeView, setActiveView] = useState<ViewKey>("overview");
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [jobs, setJobs] = useState<TrainingJobRecord[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [results, setResults] = useState<ResultsCatalog | null>(null);
  const [systemConfig, setSystemConfig] = useState<TrainingSystemConfig | null>(null);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [selectedJob, setSelectedJob] = useState<TrainingJobRecord | null>(null);
  const [selectedLog, setSelectedLog] = useState<TrainingJobLog | null>(null);
  const [globalError, setGlobalError] = useState("");

  useEffect(() => {
    localStorage.setItem(STORAGE_API_BASE, apiBase);
  }, [apiBase]);

  useEffect(() => {
    let cancelled = false;

    async function restoreSession() {
      if (!token) {
        setBooting(false);
        return;
      }

      try {
        const payload = await requestJson<{ status: string; session: AdminSession }>(
          apiBase,
          "/api/admin/session",
          { token }
        );
        if (!cancelled) {
          setSession(payload.session);
        }
      } catch {
        localStorage.removeItem(STORAGE_TOKEN);
        if (!cancelled) {
          setToken("");
          setSession(null);
        }
      } finally {
        if (!cancelled) {
          setBooting(false);
        }
      }
    }

    restoreSession();
    return () => {
      cancelled = true;
    };
  }, [apiBase, token]);

  async function refreshConsoleData(currentToken = token, currentApiBase = apiBase) {
    if (!currentToken) {
      return;
    }
    try {
      const [overviewPayload, jobsPayload, modelsPayload, resultsPayload, configPayload] = await Promise.all([
        requestJson<{ status: string; overview: OverviewPayload }>(currentApiBase, "/api/admin/overview", {
          token: currentToken
        }),
        requestJson<{ status: string; jobs: TrainingJobRecord[] }>(currentApiBase, "/api/admin/train/jobs", {
          token: currentToken
        }),
        requestJson<{ status: string; models: ModelRecord[] }>(currentApiBase, "/api/admin/models", {
          token: currentToken
        }),
        requestJson<{ status: string; results: ResultsCatalog }>(currentApiBase, "/api/admin/results", {
          token: currentToken
        }),
        requestJson<{ status: string; config: TrainingSystemConfig }>(currentApiBase, "/api/admin/system/config", {
          token: currentToken
        })
      ]);

      setOverview(overviewPayload.overview);
      setJobs(jobsPayload.jobs);
      setModels(modelsPayload.models);
      setResults(resultsPayload.results);
      setSystemConfig(configPayload.config);
      setGlobalError("");

      if (!selectedJobId && jobsPayload.jobs.length > 0) {
        setSelectedJobId(jobsPayload.jobs[0].job_id);
      }
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "读取管理端数据失败。");
    }
  }

  useEffect(() => {
    if (!session || !token) {
      return;
    }
    refreshConsoleData(token);
    const handle = window.setInterval(() => {
      refreshConsoleData(token);
    }, 6000);
    return () => window.clearInterval(handle);
  }, [session, token, apiBase]);

  useEffect(() => {
    if (!session || !token || !selectedJobId) {
      setSelectedJob(null);
      setSelectedLog(null);
      return;
    }

    let cancelled = false;

    async function refreshJobDetail() {
      try {
        const [jobPayload, logPayload] = await Promise.all([
          requestJson<{ status: string; job: TrainingJobRecord }>(
            apiBase,
            `/api/admin/train/jobs/${selectedJobId}`,
            { token }
          ),
          requestJson<{ status: string; log: TrainingJobLog }>(
            apiBase,
            `/api/admin/train/jobs/${selectedJobId}/log`,
            { token }
          )
        ]);

        if (!cancelled) {
          setSelectedJob(jobPayload.job);
          setSelectedLog(logPayload.log);
        }
      } catch {
        if (!cancelled) {
          const fallback = jobs.find((job) => job.job_id === selectedJobId) || null;
          setSelectedJob(fallback);
        }
      }
    }

    refreshJobDetail();
    const handle = window.setInterval(refreshJobDetail, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [session, token, apiBase, selectedJobId, jobs]);

  async function handleLogin(username: string, password: string) {
    const companionApiBase = normalizeApiBase(getCompanionApiBase());
    const candidates = Array.from(new Set([normalizeApiBase(apiBase), companionApiBase]));
    let payload: { status: string; token: string; session: AdminSession } | null = null;
    let successfulApiBase = candidates[0];
    let lastError: unknown = null;

    for (const candidate of candidates) {
      try {
        payload = await requestJson<{ status: string; token: string; session: AdminSession }>(
          candidate,
          "/api/admin/login",
          {
            method: "POST",
            body: { username, password }
          }
        );
        successfulApiBase = candidate;
        break;
      } catch (error) {
        lastError = error;
        if (!(error instanceof TypeError)) {
          throw error;
        }
      }
    }

    if (!payload) {
      if (lastError instanceof TypeError) {
        throw new Error(`无法连接管理员服务：${candidates.join(" 或 ")}`);
      }
      throw lastError;
    }

    localStorage.setItem(STORAGE_TOKEN, payload.token);
    localStorage.setItem(STORAGE_API_BASE, successfulApiBase);
    setApiBase(successfulApiBase);
    setToken(payload.token);
    setSession(payload.session);
    setActiveView("overview");
    await refreshConsoleData(payload.token, successfulApiBase);
  }

  async function handleLogout() {
    try {
      if (token) {
        await requestJson<Record<string, unknown>>(apiBase, "/api/admin/logout", {
          method: "POST",
          token
        });
      }
    } catch {
      // ignore
    } finally {
      localStorage.removeItem(STORAGE_TOKEN);
      setToken("");
      setSession(null);
      setOverview(null);
      setJobs([]);
      setModels([]);
      setResults(null);
      setSystemConfig(null);
      setSelectedJobId("");
      setSelectedJob(null);
      setSelectedLog(null);
    }
  }

  async function handleCreateJob(request: {
    name: string;
    mode: string;
    preset_id?: string | null;
    parameters: Record<string, unknown>;
  }) {
    if (!token) {
      throw new Error("当前未登录。");
    }
    await requestJson<Record<string, unknown>>(apiBase, "/api/admin/train/jobs", {
      method: "POST",
      token,
      body: request
    });
    await refreshConsoleData(token);
    setActiveView("jobs");
  }

  async function handleCancelJob(jobId: string) {
    if (!token) {
      return;
    }
    await requestJson<Record<string, unknown>>(apiBase, `/api/admin/train/jobs/${jobId}/cancel`, {
      method: "POST",
      token
    });
    await refreshConsoleData(token);
  }

  async function handleChangePassword(oldPassword: string, newPassword: string) {
    if (!token) {
      throw new Error("当前未登录。");
    }
    await requestJson<Record<string, unknown>>(apiBase, "/api/admin/change-password", {
      method: "POST",
      token,
      body: {
        old_password: oldPassword,
        new_password: newPassword
      }
    });
  }

  if (booting) {
    return (
      <div className="screen-center">
        <div className="loading-panel">
          <div className="loading-orbit" aria-hidden="true">
            <span />
          </div>
          <div className="loading-title">密捷管理控制台</div>
          <div className="loading-text">正在恢复管理员会话…</div>
          <div className="loading-tracks" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
        </div>
      </div>
    );
  }

  if (!session) {
    return <LoginScreen apiBase={apiBase} onApiBaseChange={setApiBase} onLogin={handleLogin} />;
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">
            MJ
            <span className="brand-mark-pulse" aria-hidden="true" />
          </div>
          <div>
            <div className="brand-title">密捷管理控制台</div>
            <div className="brand-subtitle">训练、证据与部署资产统一入口</div>
          </div>
        </div>

        <div className="server-strip">
          <div className="server-strip-head">
            <div className="server-strip-label">当前训练服务</div>
            <div className="server-live"><span aria-hidden="true" /> ONLINE</div>
          </div>
          <div className="server-strip-value">{apiBase}</div>
          <div className="server-signal" aria-hidden="true">
            {Array.from({ length: 12 }, (_, index) => <span key={index} />)}
          </div>
        </div>

        <nav className="nav-list">
          {(Object.keys(viewLabels) as ViewKey[]).map((view) => (
            <button
              key={view}
              className={`nav-item ${activeView === view ? "is-active" : ""}`}
              onClick={() => setActiveView(view)}
              aria-current={activeView === view ? "page" : undefined}
            >
              <span className="nav-label">
                <span className="nav-index">{viewIndexes[view]}</span>
                <span>{viewLabels[view]}</span>
              </span>
              {view === "jobs" && overview ? (
                <span className="nav-count">
                  {(overview.queue.running || 0) + (overview.queue.starting || 0) + (overview.queue.queued || 0)}
                </span>
              ) : null}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="session-card">
            <div className="session-name">{session.display_name}</div>
            <div className="session-meta">{session.username}</div>
          </div>
          <button className="ghost-button full-width" onClick={handleLogout}>
            退出登录
          </button>
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <div className="eyebrow">ADMIN WORKSTATION</div>
            <h1>{viewLabels[activeView]}</h1>
          </div>
          <div className="topbar-meta">
            <div className="meta-chip">{systemConfig?.runtime_mode === "mock" ? "演示后端" : "真实后端"}</div>
            <div className="meta-chip">
              GPU {overview?.gpu.available ? `${overview.gpu.devices.length} 张` : "未检测到"}
            </div>
            <LiveClock />
          </div>
        </header>

        <SystemRibbon
          runtimeMode={systemConfig?.runtime_mode || overview?.environment.runtime_mode || "—"}
          activeJobs={
            (overview?.queue.queued || 0) +
            (overview?.queue.starting || 0) +
            (overview?.queue.running || 0) +
            (overview?.queue.postprocessing || 0)
          }
          modelCount={models.length}
          evidenceCount={results?.sections.length || 0}
        />

        {globalError ? <div className="banner-error">{globalError}</div> : null}

        <div className="view-stage" key={activeView}>
          {activeView === "overview" && overview ? (
            <OverviewView overview={overview} formalModel={overview.formal_model ?? null} />
          ) : null}

          {activeView === "jobs" ? (
            <JobsView
              jobs={jobs}
              selectedJobId={selectedJobId}
              selectedJob={selectedJob}
              selectedLog={selectedLog}
              systemConfig={systemConfig}
              onSelectJob={setSelectedJobId}
              onCreateJob={handleCreateJob}
              onCancelJob={handleCancelJob}
            />
          ) : null}

          {activeView === "models" ? <ModelsView models={models} /> : null}

          {activeView === "results" && results ? <ResultsView results={results} /> : null}

          {activeView === "system" && systemConfig ? (
            <SystemView
              config={systemConfig}
              apiBase={apiBase}
              onApiBaseChange={setApiBase}
              onChangePassword={handleChangePassword}
            />
          ) : null}
        </div>
      </main>
    </div>
  );
}

function LiveClock() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const handle = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(handle);
  }, []);

  return (
    <div className="live-clock" aria-label={`系统时间 ${now.toLocaleTimeString("zh-CN", { hour12: false })}`}>
      <span className="live-clock-status"><span aria-hidden="true" /> LIVE</span>
      <time dateTime={now.toISOString()}>
        {now.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}
      </time>
    </div>
  );
}

function SystemRibbon({
  runtimeMode,
  activeJobs,
  modelCount,
  evidenceCount
}: {
  runtimeMode: string;
  activeJobs: number;
  modelCount: number;
  evidenceCount: number;
}) {
  const items = [
    ["RUNTIME", runtimeMode.toUpperCase()],
    ["TRAIN JOBS", String(activeJobs)],
    ["MODEL ASSETS", String(modelCount)],
    ["EVIDENCE SETS", String(evidenceCount)],
    ["CONTROL CHANNEL", "CONNECTED"]
  ];

  return (
    <div className="system-ribbon" aria-label="控制台实时状态">
      <div className="system-ribbon-tag"><span aria-hidden="true" /> SYSTEM FEED</div>
      <div className="system-ribbon-viewport">
        <div className="system-ribbon-track">
          {[0, 1].map((group) => (
            <div className="system-ribbon-group" key={group} aria-hidden={group === 1 ? "true" : undefined}>
              {items.map(([label, value]) => (
                <span className="system-ribbon-item" key={`${group}-${label}`}>
                  <small>{label}</small>
                  <strong>{value}</strong>
                  <i aria-hidden="true" />
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LoginScreen({
  apiBase,
  onApiBaseChange,
  onLogin
}: {
  apiBase: string;
  onApiBaseChange: (next: string) => void;
  onLogin: (username: string, password: string) => Promise<void>;
}) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  return (
    <div className="login-shell">
      <div className="login-panel">
        <div className="login-copy">
          <div className="login-scan" aria-hidden="true" />
          <div className="eyebrow">密捷 ADMIN</div>
          <h1>连接训练服务，进入管理工作台</h1>
          <p>
            这个入口只面向管理员，用于查看真实训练任务、模型资产、阈值校准结果和部署输出路径。
          </p>
          <div className="login-telemetry" aria-hidden="true">
            <div className="telemetry-head">
              <span>SECURE CONTROL CHANNEL</span>
              <span className="telemetry-live"><i /> ACTIVE</span>
            </div>
            <div className="telemetry-track">
              {Array.from({ length: 20 }, (_, index) => <span key={index} />)}
            </div>
          </div>
          <div className="callout-grid">
            <div className="callout-card">
              <div className="callout-title">真实任务</div>
              <div className="callout-text">创建训练、轮询状态、查看 stdout/stderr 与结果资产。</div>
            </div>
            <div className="callout-card">
              <div className="callout-title">证据闭环</div>
              <div className="callout-text">模型、结果 JSON 与输出目录可以在同一处回溯。</div>
            </div>
          </div>
        </div>

        <div className="login-card">
          <div className="card-title">管理员登录</div>
          <form
            className="form-stack"
            onSubmit={async (event) => {
              event.preventDefault();
              setSubmitting(true);
              setError("");
              try {
                await onLogin(username, password);
              } catch (err) {
                setError(err instanceof Error ? err.message : "登录失败。");
              } finally {
                setSubmitting(false);
              }
            }}
          >
            <label className="field">
              <span>账号</span>
              <input value={username} onChange={(event) => setUsername(event.target.value)} />
            </label>
            <label className="field">
              <span>密码</span>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
            {error ? <div className="inline-error">{error}</div> : null}
            <button className="primary-button" disabled={submitting}>
              {submitting ? "登录中…" : "进入工作台"}
            </button>
          </form>

          <ConnectionPanel apiBase={apiBase} onApiBaseChange={onApiBaseChange} compact />
        </div>
      </div>
    </div>
  );
}

function OverviewView({ overview, formalModel }: { overview: OverviewPayload; formalModel: ModelRecord | null }) {
  const activeCount =
    (overview.queue.queued || 0) +
    (overview.queue.starting || 0) +
    (overview.queue.running || 0) +
    (overview.queue.postprocessing || 0);

  return (
    <div className="view-grid overview-grid">
      <OverviewCommandHero overview={overview} formalModel={formalModel} activeCount={activeCount} />
      <OverviewPipeline overview={overview} formalModel={formalModel} activeCount={activeCount} />

      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">正式主线模型</div>
            <div className="section-subtitle">当前主线 bundle 与其核心指标。</div>
          </div>
        </div>
        {formalModel ? (
          <div className="key-value-grid">
            <KeyValue label="模型名称" value={friendlyModelName(formalModel)} />
            <KeyValue label="领域" value={formalModel.domain === "medical" ? "医疗" : "金融"} />
            <KeyValue label="base_rate" value={formatShortNumber(formalModel.base_rate, 2)} />
            <KeyValue label="阈值精度" value={formatPercent(formalModel.threshold_accuracy)} />
            <KeyValue label="argmax 精度" value={formatPercent(formalModel.argmax_accuracy)} />
            <KeyValue label="AUC" value={formatNumber(formalModel.auc)} />
            <KeyValue label="bundle 路径" value={formatDisplayPath(formalModel.bundle_dir)} mono />
            <KeyValue label="来源 run" value={formatDisplayPath(formalModel.source_run || "—")} mono />
          </div>
        ) : (
          <div className="empty-state">未读取到正式主线 bundle。</div>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">最近完成任务</div>
            <div className="section-subtitle">优先展示已完成的真实训练或历史归档任务。</div>
          </div>
        </div>
        <div className="stack-list">
          {overview.recent_completed.slice(0, 6).map((job) => (
            <div className="row-card" key={job.job_id}>
              <div>
                <div className="row-title">{friendlyJobName(job)}</div>
                <div className="row-meta">
                  {presetLabel(job.preset_id)} · {formatDateTime(job.finished_at)}
                </div>
              </div>
              <div className="row-right">
                <span className={`status-pill ${job.status}`}>{statusLabel(job.status)}</span>
                <div className="row-metric">{formatPercent(job.metrics_summary?.bestThresholdAccuracy)}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">环境摘要</div>
            <div className="section-subtitle">当前后端使用的 Python、目录和并发限制。</div>
          </div>
        </div>
        <div className="key-value-grid">
          <KeyValue label="Python" value={`${formatDisplayPath(overview.environment.python_bin)} (${overview.environment.python_version})`} mono />
          <KeyValue label="运行模式" value={overview.environment.runtime_mode} />
          <KeyValue label="仓库根目录" value={formatDisplayPath(overview.environment.repo_root)} mono />
          <KeyValue label="训练输出目录" value={formatDisplayPath(overview.environment.train_output_root)} mono />
          <KeyValue label="bundle 输出目录" value={formatDisplayPath(overview.environment.bundle_output_root)} mono />
          <KeyValue label="任务目录" value={formatDisplayPath(overview.environment.job_root)} mono />
          <KeyValue label="默认训练集" value={formatDisplayPath(overview.environment.default_train_data_path)} mono />
          <KeyValue label="默认验证集" value={formatDisplayPath(overview.environment.default_eval_data_path)} mono />
          <KeyValue label="并发上限" value={String(overview.environment.max_concurrent_train_jobs)} />
        </div>
      </section>
    </div>
  );
}

type PipelineStageState = "live" | "ready" | "idle" | "missing";

type PipelineStage = {
  id: string;
  code: string;
  label: string;
  status: string;
  state: PipelineStageState;
  description: string;
  facts: Array<{ label: string; value: string; mono?: boolean }>;
};

function OverviewPipeline({
  overview,
  formalModel,
  activeCount
}: {
  overview: OverviewPayload;
  formalModel: ModelRecord | null;
  activeCount: number;
}) {
  const [selectedStageId, setSelectedStageId] = useState("secure");
  const dataReady = Boolean(
    overview.environment.default_train_data_path && overview.environment.default_eval_data_path
  );
  const completedCount = overview.queue.completed || 0;
  const failedCount = overview.queue.failed || 0;
  const stages: PipelineStage[] = [
    {
      id: "data",
      code: "01",
      label: "训练数据",
      status: dataReady ? "已配置" : "未配置",
      state: dataReady ? "ready" : "missing",
      description: "训练集与验证集通过项目相对路径接入，迁移环境时不依赖本机绝对目录。",
      facts: [
        { label: "训练集", value: formatDisplayPath(overview.environment.default_train_data_path || "—"), mono: true },
        { label: "验证集", value: formatDisplayPath(overview.environment.default_eval_data_path || "—"), mono: true }
      ]
    },
    {
      id: "train",
      code: "02",
      label: "ViT 模型训练",
      status: activeCount > 0 ? "训练中" : completedCount > 0 ? "已有训练记录" : "等待训练任务",
      state: activeCount > 0 ? "live" : completedCount > 0 ? "ready" : "idle",
      description: "这里统计的是模型训练任务。任务进入后台训练队列后，依次执行 ViT 训练、阈值处理和 Bundle 导出，不包含 SPU 推理请求。",
      facts: [
        { label: "活动训练任务", value: String(activeCount) },
        { label: "已完成训练", value: String(completedCount) },
        { label: "训练失败记录", value: String(failedCount) },
        { label: "训练并发上限", value: String(overview.environment.max_concurrent_train_jobs) }
      ]
    },
    {
      id: "prune",
      code: "03",
      label: "动态剪枝",
      status: typeof formalModel?.base_rate === "number" ? "已固化" : "待模型",
      state: typeof formalModel?.base_rate === "number" ? "ready" : "idle",
      description: "动态 ViT 根据样本保留高价值 token，正式 bundle 固化对应剪枝率与安全静态深度。",
      facts: [
        { label: "base_rate", value: formatShortNumber(formalModel?.base_rate, 2) },
        { label: "安全静态深度", value: formalModel?.secure_static_train_depth?.toString() || "—" },
        { label: "当前模型", value: formalModel ? friendlyModelName(formalModel) : "—" }
      ]
    },
    {
      id: "calibrate",
      code: "04",
      label: "阈值校准",
      status: typeof formalModel?.threshold_accuracy === "number" ? "已校准" : "待结果",
      state: typeof formalModel?.threshold_accuracy === "number" ? "ready" : "idle",
      description: "在验证结果上确定分类阈值，并同步保留 argmax 精度与 AUC，构成模型效果证据。",
      facts: [
        { label: "阈值精度", value: formatPercent(formalModel?.threshold_accuracy) },
        { label: "argmax 精度", value: formatPercent(formalModel?.argmax_accuracy) },
        { label: "AUC", value: formatNumber(formalModel?.auc) }
      ]
    },
    {
      id: "bundle",
      code: "05",
      label: "模型 Bundle",
      status: formalModel?.bundle_dir ? "已冻结" : "待生成",
      state: formalModel?.bundle_dir ? "ready" : "idle",
      description: "将模型、阈值和运行参数收束为可追踪 bundle，作为展示与复现的正式模型资产。",
      facts: [
        { label: "模型名称", value: formalModel ? friendlyModelName(formalModel) : "—" },
        { label: "Bundle", value: formatDisplayPath(formalModel?.bundle_dir || "—"), mono: true },
        { label: "来源 run", value: formatDisplayPath(formalModel?.source_run || "—"), mono: true }
      ]
    },
    {
      id: "secure",
      code: "06",
      label: "SPU 推理接入",
      status: formalModel ? "Bundle 已接入" : "等待正式模型",
      state: formalModel ? "ready" : "idle",
      description: "正式 Bundle 已接入 SPU 隐私推理环境，模型资产、运行模式与结果证据保持一致。",
      facts: [
        { label: "运行模式", value: overview.environment.runtime_mode.toUpperCase() },
        {
          label: "Bundle 状态",
          value: formalModel?.bundle_dir ? "正式 Bundle 已加载" : "等待 Bundle"
        },
        {
          label: "Bundle 路径",
          value: formatDisplayPath(formalModel?.bundle_dir || "—"),
          mono: true
        },
        { label: "正式模型", value: formalModel ? friendlyModelName(formalModel) : "—" }
      ]
    }
  ];
  const selectedStage = stages.find((stage) => stage.id === selectedStageId) || stages[0];

  return (
    <section className="pipeline-canvas">
      <div className="pipeline-header">
        <div>
          <div className="pipeline-kicker">模型资产链路 / 01—06</div>
          <div className="section-title">模型训练、固化与 SPU 推理接入</div>
          <div className="section-subtitle">点击节点查看训练队列、正式模型资产与 SPU 运行环境。</div>
        </div>
        <div className="pipeline-activity">
          <span className={activeCount > 0 ? "is-live" : ""} />
          {activeCount > 0 ? `${activeCount} 个训练任务活动中` : "当前无活动训练任务"}
        </div>
      </div>

      <div className="pipeline-scope-strip" aria-label="任务统计口径说明">
        <div>
          <strong>01—05</strong>
          <span>模型训练与资产固化流程，任务数量来自训练队列</span>
        </div>
        <i aria-hidden="true" />
        <div>
          <strong>06</strong>
          <span>正式 Bundle 与 SPU 隐私推理运行环境</span>
        </div>
      </div>

      <div className="pipeline-track-shell">
        <div className="pipeline-track">
          {stages.map((stage, index) => (
            <div className="pipeline-stage-wrap" key={stage.id}>
              <button
                type="button"
                className={`pipeline-node state-${stage.state}${selectedStage.id === stage.id ? " is-selected" : ""}`}
                onClick={() => setSelectedStageId(stage.id)}
                aria-pressed={selectedStage.id === stage.id}
              >
                <span className="pipeline-node-top">
                  <span className="pipeline-node-index">{stage.code}</span>
                  <span className="pipeline-node-dot" />
                </span>
                <strong>{stage.label}</strong>
                <span className="pipeline-node-status">{stage.status}</span>
              </button>
              {index < stages.length - 1 ? (
                <span className="pipeline-connector" aria-hidden="true"><i /></span>
              ) : null}
            </div>
          ))}
        </div>
      </div>

      <div className="pipeline-detail" key={selectedStage.id}>
        <div className="pipeline-detail-copy">
          <span>{selectedStage.code} / 当前节点</span>
          <h3>{selectedStage.label}</h3>
          <p>{selectedStage.description}</p>
        </div>
        <div className="pipeline-detail-grid">
          {selectedStage.facts.map((fact) => (
            <div className="pipeline-detail-item" key={fact.label}>
              <span>{fact.label}</span>
              <strong className={fact.mono ? "mono" : ""}>{fact.value}</strong>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function OverviewCommandHero({
  overview,
  formalModel,
  activeCount
}: {
  overview: OverviewPayload;
  formalModel: ModelRecord | null;
  activeCount: number;
}) {
  return (
    <section className="command-hero">
      <div className="command-hero-scan" aria-hidden="true" />
      <div className="command-hero-copy">
        <div className="command-kicker"><span aria-hidden="true" /> SECURE INFERENCE OPERATIONS</div>
        <h2>双向隐私推理<br />运行态势</h2>
        <p>集中呈现模型训练队列、正式资产与 SPU 隐私推理运行环境。</p>

        <div className="command-facts">
          <div>
            <span>运行模式</span>
            <strong>{overview.environment.runtime_mode.toUpperCase()}</strong>
          </div>
          <div>
            <span>活动训练任务</span>
            <strong>{activeCount}</strong>
          </div>
          <div>
            <span>GPU 通道</span>
            <strong>{overview.gpu.available ? `${overview.gpu.devices.length} READY` : "STANDBY"}</strong>
          </div>
        </div>

        <div className="command-model-line">
          <span>当前主线</span>
          <strong>{formalModel ? friendlyModelName(formalModel) : "等待模型资产"}</strong>
        </div>
      </div>

      <div className="command-orbit" aria-hidden="true">
        <div className="orbit-coordinates orbit-coordinates-x"><span>−128</span><span>000</span><span>+128</span></div>
        <div className="orbit-coordinates orbit-coordinates-y"><span>+64</span><span>000</span><span>−64</span></div>
        <div className="orbit-ring orbit-ring-outer">
          <i className="orbit-node node-a" />
          <i className="orbit-node node-b" />
          <i className="orbit-node node-c" />
        </div>
        <div className="orbit-ring orbit-ring-middle">
          <i className="orbit-node node-a" />
          <i className="orbit-node node-b" />
        </div>
        <div className="orbit-ring orbit-ring-inner" />
        <div className="orbit-axis orbit-axis-horizontal" />
        <div className="orbit-axis orbit-axis-vertical" />
        <div className="orbit-core">
          <span>CHANNEL</span>
          <strong>ONLINE</strong>
          <i />
        </div>
        <div className="orbit-sweep" />
      </div>

      <div className="command-edge-data" aria-hidden="true">
        {Array.from({ length: 28 }, (_, index) => <span key={index} />)}
      </div>
    </section>
  );
}

function JobsView({
  jobs,
  selectedJobId,
  selectedJob,
  selectedLog,
  systemConfig,
  onSelectJob,
  onCreateJob,
  onCancelJob
}: {
  jobs: TrainingJobRecord[];
  selectedJobId: string;
  selectedJob: TrainingJobRecord | null;
  selectedLog: TrainingJobLog | null;
  systemConfig: TrainingSystemConfig | null;
  onSelectJob: (jobId: string) => void;
  onCreateJob: (request: {
    name: string;
    mode: string;
    preset_id?: string | null;
    parameters: Record<string, unknown>;
  }) => Promise<void>;
  onCancelJob: (jobId: string) => Promise<void>;
}) {
  const [showCreate, setShowCreate] = useState(false);
  const [statusFilter, setStatusFilter] = useState("all");
  const filteredJobs = jobs.filter((job) => statusFilter === "all" || job.status === statusFilter);

  return (
    <div className="jobs-layout">
      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">训练任务列表</div>
            <div className="section-subtitle">已包含历史真实训练记录与控制台创建的新任务。</div>
          </div>
          <div className="section-actions">
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} className="select">
              <option value="all">全部状态</option>
              {Object.keys(jobStatusLabels).map((status) => (
                <option key={status} value={status}>
                  {statusLabel(status)}
                </option>
              ))}
            </select>
            <button className="primary-button" onClick={() => setShowCreate(true)}>
              新建训练任务
            </button>
          </div>
        </div>

        <div className="table-shell">
          <table className="data-table">
            <thead>
              <tr>
                <th>任务名称</th>
                <th>来源</th>
                <th>状态</th>
                <th>阈值精度</th>
                <th>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.map((job) => (
                <tr
                  key={job.job_id}
                  className={`${selectedJobId === job.job_id ? "is-selected" : ""} ${
                    ACTIVE_JOB_STATUSES.has(job.status) ? "is-live" : ""
                  }`.trim()}
                  onClick={() => onSelectJob(job.job_id)}
                >
                  <td>
                    <div className="cell-title">{friendlyJobName(job)}</div>
                    <div className="cell-subtitle">{job.job_id}</div>
                  </td>
                  <td>{job.source === "historical" ? "历史真实训练" : presetLabel(job.preset_id)}</td>
                  <td>
                    <span className={`status-pill ${job.status}`}>{statusLabel(job.status)}</span>
                  </td>
                  <td>{formatPercent(job.metrics_summary?.bestThresholdAccuracy)}</td>
                  <td>{formatDateTime(job.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-card detail-panel">
        <div className="section-header">
          <div>
            <div className="section-title">任务详情</div>
            <div className="section-subtitle">命令、日志、输出资产与指标摘要。</div>
          </div>
          {selectedJob && !selectedJob.readonly && ACTIVE_JOB_STATUSES.has(selectedJob.status) ? (
            <button
              className="danger-button"
              onClick={async () => {
                if (window.confirm("确认终止当前任务？")) {
                  await onCancelJob(selectedJob.job_id);
                }
              }}
            >
              终止任务
            </button>
          ) : null}
        </div>

        {selectedJob ? (
          <>
            <div className="key-value-grid compact">
              <KeyValue label="任务名称" value={friendlyJobName(selectedJob)} />
              <KeyValue label="状态" value={statusLabel(selectedJob.status)} />
              <KeyValue label="当前阶段" value={selectedJob.current_step || "—"} />
              <KeyValue label="任务来源" value={selectedJob.source === "historical" ? "历史真实训练" : "控制台创建"} />
              <KeyValue label="创建时间" value={formatDateTime(selectedJob.created_at)} />
              <KeyValue label="完成时间" value={formatDateTime(selectedJob.finished_at)} />
              <KeyValue label="输出目录" value={formatDisplayPath(selectedJob.output_dir)} mono />
              <KeyValue label="bundle 目录" value={formatDisplayPath(selectedJob.bundle_dir)} mono />
            </div>

            <div className="split-grid">
              <div className="subcard">
                <div className="subcard-title">核心指标</div>
                <div className="metric-grid-two">
                  <KeyValue label="argmax 精度" value={formatPercent(selectedJob.metrics_summary?.argmaxAccuracy)} />
                  <KeyValue label="阈值精度" value={formatPercent(selectedJob.metrics_summary?.bestThresholdAccuracy)} />
                  <KeyValue label="最佳阈值" value={formatNumber(selectedJob.metrics_summary?.bestThreshold)} />
                  <KeyValue label="AUC" value={formatNumber(selectedJob.metrics_summary?.auc)} />
                </div>
              </div>
              <div className="subcard">
                <div className="subcard-title">输出资产</div>
                <div className="stack-list small">
                  {Object.entries(selectedJob.artifacts || {}).map(([key, value]) => (
                    <div key={key} className="artifact-row">
                      <span>{key}</span>
                      <code>{formatDisplayPath(value)}</code>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="subcard">
              <div className="subcard-title">命令链</div>
              <div className="stack-list small">
                {(selectedJob.command_sequence || []).map((command, index) => (
                  <pre key={index} className="code-block command-block">
                    {sanitizeDisplayText(command)}
                  </pre>
                ))}
              </div>
            </div>

            <div className="split-grid logs">
              <div className="subcard">
                <div className="subcard-title">stdout</div>
                <pre className="code-block log-block">
                  {sanitizeDisplayText(selectedLog?.stdout || "暂无 stdout 输出。")}
                </pre>
              </div>
              <div className="subcard">
                <div className="subcard-title">stderr</div>
                <pre className="code-block log-block">
                  {sanitizeDisplayText(
                    selectedLog?.stderr || selectedJob.error_message || "暂无 stderr 输出。"
                  )}
                </pre>
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state">从左侧任务列表选择一条记录。</div>
        )}
      </section>

      {showCreate && systemConfig ? (
        <CreateJobModal
          config={systemConfig}
          onClose={() => setShowCreate(false)}
          onSubmit={async (request) => {
            await onCreateJob(request);
            setShowCreate(false);
          }}
        />
      ) : null}
    </div>
  );
}

function CreateJobModal({
  config,
  onClose,
  onSubmit
}: {
  config: TrainingSystemConfig;
  onClose: () => void;
  onSubmit: (request: {
    name: string;
    mode: string;
    preset_id?: string | null;
    parameters: Record<string, unknown>;
  }) => Promise<void>;
}) {
  const defaultPreset = config.training_presets[0];
  const [name, setName] = useState("现场展示训练任务");
  const [mode, setMode] = useState("preset");
  const [presetId, setPresetId] = useState(defaultPreset?.id || "");
  const [parameters, setParameters] = useState<Record<string, string | number | boolean>>(
    normalizePathParameters(defaultPreset?.parameters || {})
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  function updateFromPreset(nextPresetId: string) {
    setPresetId(nextPresetId);
    const matched = config.training_presets.find((item) => item.id === nextPresetId);
    setParameters(normalizePathParameters(matched?.parameters || {}));
  }

  return createPortal(
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="create-job-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="section-header modal-header">
          <div>
            <div className="section-title" id="create-job-title">新建训练任务</div>
            <div className="section-subtitle">默认采用现有训练预设，可局部改写关键参数。</div>
          </div>
          <button className="ghost-button" onClick={onClose}>
            关闭
          </button>
        </div>

        <form
          className="form-stack modal-form"
          onSubmit={async (event) => {
            event.preventDefault();
            setSubmitting(true);
            setError("");
            try {
              await onSubmit({
                name,
                mode,
                preset_id: presetId || null,
                parameters
              });
            } catch (err) {
              setError(err instanceof Error ? err.message : "创建任务失败。");
            } finally {
              setSubmitting(false);
            }
          }}
        >
          <div className="modal-form-scroll">
            <div className="form-grid">
              <label className="field">
                <span>任务名称</span>
                <input value={name} onChange={(event) => setName(event.target.value)} />
              </label>
              <label className="field">
                <span>启动方式</span>
                <select value={mode} onChange={(event) => setMode(event.target.value)} className="select">
                  <option value="preset">主线预设</option>
                  <option value="custom">自定义参数</option>
                </select>
              </label>
            </div>

            <label className="field">
              <span>预设模板</span>
              <select
                value={presetId}
                className="select"
                onChange={(event) => updateFromPreset(event.target.value)}
              >
                {config.training_presets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {presetNames[preset.id] ?? preset.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="preset-note">
              {config.training_presets.find((item) => item.id === presetId)?.description || "—"}
            </div>

            <div className="form-grid wide">
              {editableFields.map((field) => {
                const currentValue = parameters[field.key];
                if (field.type === "boolean") {
                  return (
                    <label key={field.key} className="toggle-field">
                      <input
                        type="checkbox"
                        checked={Boolean(currentValue)}
                        onChange={(event) =>
                          setParameters((previous) => ({
                            ...previous,
                            [field.key]: event.target.checked
                          }))
                        }
                      />
                      <span>{field.label}</span>
                    </label>
                  );
                }

                return (
                  <label key={field.key} className="field">
                    <span>{field.label}</span>
                    <input
                      type={field.type === "number" ? "number" : "text"}
                      step={field.step}
                      value={String(currentValue ?? "")}
                      onChange={(event) =>
                        setParameters((previous) => ({
                          ...previous,
                          [field.key]:
                            field.type === "number" ? Number(event.target.value) : event.target.value
                        }))
                      }
                    />
                  </label>
                );
              })}
            </div>

            {error ? <div className="inline-error">{error}</div> : null}
          </div>

          <div className="modal-actions">
            <button type="button" className="ghost-button" onClick={onClose}>
              取消
            </button>
            <button className="primary-button" disabled={submitting}>
              {submitting ? "创建中…" : "提交任务"}
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body
  );
}

function ModelsView({ models }: { models: ModelRecord[] }) {
  return (
    <div className="view-grid">
      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">已训练模型</div>
            <div className="section-subtitle">从真实 bundle 与参数快照中汇总，不是手工录入。</div>
          </div>
        </div>

        <div className="cards-grid">
          {models.map((model) => (
            <div className="model-card" key={model.id}>
              <div className="model-card-top">
                <div>
                  <div className="model-name">{friendlyModelName(model)}</div>
                  <div className="model-meta">
                    {model.domain === "medical" ? "医疗" : "金融"} · {model.status}
                  </div>
                </div>
                <span className={`status-pill ${model.status.includes("主线") ? "completed" : "starting"}`}>
                  {model.status}
                </span>
              </div>
              <div className="metric-grid-two">
                <KeyValue label="base_rate" value={formatShortNumber(model.base_rate, 2)} />
                <KeyValue label="depth" value={model.secure_static_train_depth?.toString() || "—"} />
                <KeyValue label="阈值精度" value={formatPercent(model.threshold_accuracy)} />
                <KeyValue label="AUC" value={formatNumber(model.auc)} />
              </div>
              <div className="stack-list small">
                <div className="artifact-row">
                  <span>bundle</span>
                  <code>{formatDisplayPath(model.bundle_dir)}</code>
                </div>
                <div className="artifact-row">
                  <span>args_snapshot</span>
                  <code>{formatDisplayPath(model.args_snapshot_path)}</code>
                </div>
                <div className="artifact-row">
                  <span>manifest</span>
                  <code>{formatDisplayPath(model.manifest_path)}</code>
                </div>
                <div className="artifact-row">
                  <span>source_run</span>
                  <code>{formatDisplayPath(model.source_run || "—")}</code>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ResultsView({ results }: { results: ResultsCatalog }) {
  const [selectedPath, setSelectedPath] = useState(results.sections[0]?.path || "");
  const selectedSection =
    results.sections.find((section) => section.path === selectedPath) || results.sections[0] || null;
  const selectedSummary = selectedSection ? summarizeSection(selectedSection) : [];

  useEffect(() => {
    if (!results.sections.some((section) => section.path === selectedPath)) {
      setSelectedPath(results.sections[0]?.path || "");
    }
  }, [results.sections, selectedPath]);

  return (
    <div className="view-grid">
      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">正式指标口径</div>
            <div className="section-subtitle">直接来自后端读取的正式结果文件与通信量剖面。</div>
          </div>
        </div>
        <div className="stats-grid">
          <MetricCard
            label="阈值精度"
            value={formatPercent(results.formal_metrics.threshold_accuracy)}
            detail="formal threshold accuracy"
            tone="green"
          />
          <MetricCard
            label="AUC"
            value={formatNumber(results.formal_metrics.auc)}
            detail="formal auc"
            tone="blue"
          />
          <MetricCard
            label="秒/样本"
            value={formatShortNumber(results.formal_metrics.sec_per_sample, 2)}
            detail="SPU 侧采样通信剖面"
            tone="slate"
          />
          <MetricCard
            label="总通信量"
            value={`${formatShortNumber(results.formal_metrics.dual_total_gib, 2)} GiB`}
            detail="dual_total_gib"
            tone="amber"
          />
        </div>
        <div className="artifact-row top-spacing">
          <span>正式 bundle</span>
          <code>{formatDisplayPath(results.formal_metrics.bundle_dir)}</code>
        </div>
      </section>

      <section className="evidence-workbench">
        <div className="evidence-workbench-header">
          <div>
            <div className="pipeline-kicker">EVIDENCE DESK / READ ONLY</div>
            <div className="section-title">结果证据工作台</div>
            <div className="section-subtitle">选择结果文件，同时核对指标摘要、来源路径与原始 JSON。</div>
          </div>
          <div className="evidence-file-count">{results.sections.length.toString().padStart(2, "0")} FILES</div>
        </div>

        {selectedSection ? (
          <div className="evidence-workspace-grid">
            <aside className="evidence-tree-pane">
              <div className="evidence-pane-label">证据文件</div>
              <div className="evidence-file-list">
                {results.sections.map((section, index) => {
                  return (
                    <button
                      type="button"
                      className={`evidence-file-button${selectedSection.path === section.path ? " is-active" : ""}`}
                      key={section.path}
                      onClick={() => setSelectedPath(section.path)}
                      aria-pressed={selectedSection.path === section.path}
                    >
                      <span className="evidence-file-index">{String(index + 1).padStart(2, "0")}</span>
                      <span>
                        <strong>{friendlySectionLabel(section)}</strong>
                        <code>{formatDisplayPath(section.path)}</code>
                      </span>
                    </button>
                  );
                })}
              </div>
            </aside>

            <div className="evidence-summary-pane" key={`summary-${selectedSection.path}`}>
              <div className="evidence-pane-label">指标摘要</div>
              <div className="evidence-summary-title">
                <span>SELECTED EVIDENCE</span>
                <h3>{friendlySectionLabel(selectedSection)}</h3>
                <code>{formatDisplayPath(selectedSection.path)}</code>
              </div>

              <div className="evidence-metric-list">
                {selectedSummary.map((item) => {
                  const visualPercent = evidenceVisualPercent(item);
                  return (
                    <div className="evidence-metric" key={item.key}>
                      <div>
                        <span>{item.key}</span>
                        <strong>{item.value}</strong>
                      </div>
                      {visualPercent !== null ? (
                        <span className="evidence-meter" aria-hidden="true">
                          <i style={{ transform: `scaleX(${visualPercent / 100})` }} />
                        </span>
                      ) : null}
                    </div>
                  );
                })}
              </div>

              <div className="evidence-structure">
                <span>结构信息</span>
                <div><strong>{Object.keys(selectedSection.payload).length}</strong><small>顶层字段</small></div>
                <div><strong>JSON</strong><small>原始格式</small></div>
              </div>
            </div>

            <div className="evidence-json-pane" key={`json-${selectedSection.path}`}>
              <div className="evidence-json-header">
                <div className="evidence-pane-label">原始 JSON</div>
                <span>READ ONLY</span>
              </div>
              <pre className="code-block evidence-json">{JSON.stringify(formatDisplayJson(selectedSection.payload), null, 2)}</pre>
            </div>
          </div>
        ) : (
          <div className="empty-state">未读取到结果证据文件。</div>
        )}
      </section>
    </div>
  );
}

function SystemView({
  config,
  apiBase,
  onApiBaseChange,
  onChangePassword
}: {
  config: TrainingSystemConfig;
  apiBase: string;
  onApiBaseChange: (next: string) => void;
  onChangePassword: (oldPassword: string, newPassword: string) => Promise<void>;
}) {
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  return (
    <div className="view-grid">
      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">训练服务连接</div>
            <div className="section-subtitle">切换训练服务器地址，并验证 `/api/health` 可达性。</div>
          </div>
        </div>
        <ConnectionPanel apiBase={apiBase} onApiBaseChange={onApiBaseChange} />
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">后端配置</div>
            <div className="section-subtitle">当前后端暴露给管理员工作台的核心环境参数。</div>
          </div>
        </div>
        <div className="key-value-grid">
          <KeyValue label="仓库根目录" value={formatDisplayPath(config.repo_root)} mono />
          <KeyValue label="Python 路径" value={formatDisplayPath(config.python_bin)} mono />
          <KeyValue label="任务目录" value={formatDisplayPath(config.job_root)} mono />
          <KeyValue label="训练输出目录" value={formatDisplayPath(config.train_output_root)} mono />
          <KeyValue label="bundle 输出目录" value={formatDisplayPath(config.bundle_output_root)} mono />
          <KeyValue label="默认训练集" value={formatDisplayPath(config.default_train_data_path)} mono />
          <KeyValue label="默认验证集" value={formatDisplayPath(config.default_eval_data_path)} mono />
          <KeyValue label="默认设备" value={config.default_device} />
          <KeyValue label="默认 batch size" value={String(config.default_batch_size)} />
          <KeyValue label="默认 num_workers" value={String(config.default_num_workers)} />
          <KeyValue label="并发上限" value={String(config.max_concurrent_train_jobs)} />
          <KeyValue label="运行模式" value={config.runtime_mode} />
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">训练预设</div>
            <div className="section-subtitle">用于快速发起主线、对照线和复现实验。</div>
          </div>
        </div>
        <div className="cards-grid">
          {config.training_presets.map((preset) => (
            <div className="model-card" key={preset.id}>
              <div className="model-name">{presetNames[preset.id] ?? preset.name}</div>
              <div className="model-meta">{preset.description}</div>
              <div className="metric-grid-two">
                <KeyValue label="base_rate" value={String(preset.parameters.base_rate ?? "—")} />
                <KeyValue label="epochs" value={String(preset.parameters.epochs ?? "—")} />
                <KeyValue label="batch" value={String(preset.parameters.batch_size ?? "—")} />
                <KeyValue label="depth" value={String(preset.parameters.secure_static_train_depth ?? "—")} />
              </div>
              <pre className="code-block result-json">
                {JSON.stringify(formatDisplayJson(preset.parameters), null, 2)}
              </pre>
            </div>
          ))}
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <div className="section-title">修改管理员密码</div>
            <div className="section-subtitle">密码最少 8 位，修改后立即生效。</div>
          </div>
        </div>
        <form
          className="form-stack narrow"
          onSubmit={async (event) => {
            event.preventDefault();
            setMessage("");
            setError("");
            try {
              await onChangePassword(oldPassword, newPassword);
              setMessage("管理员密码已更新。");
              setOldPassword("");
              setNewPassword("");
            } catch (err) {
              setError(err instanceof Error ? err.message : "修改密码失败。");
            }
          }}
        >
          <label className="field">
            <span>原密码</span>
            <input
              type="password"
              value={oldPassword}
              onChange={(event) => setOldPassword(event.target.value)}
            />
          </label>
          <label className="field">
            <span>新密码</span>
            <input
              type="password"
              value={newPassword}
              onChange={(event) => setNewPassword(event.target.value)}
            />
          </label>
          {message ? <div className="inline-success">{message}</div> : null}
          {error ? <div className="inline-error">{error}</div> : null}
          <button className="primary-button">提交修改</button>
        </form>
      </section>
    </div>
  );
}

function ConnectionPanel({
  apiBase,
  onApiBaseChange,
  compact = false
}: {
  apiBase: string;
  onApiBaseChange: (next: string) => void;
  compact?: boolean;
}) {
  const [draft, setDraft] = useState(apiBase);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    setDraft(apiBase);
  }, [apiBase]);

  async function testConnection(base: string) {
    setTesting(true);
    setMessage("");
    setHealth(null);
    try {
      const normalized = normalizeApiBase(base);
      const response = await fetch(`${normalized}/api/health`);
      const payload = (await response.json()) as HealthPayload;
      if (!response.ok || payload.status !== "ok") {
        throw new Error("health check failed");
      }
      setHealth(payload);
      setMessage("连接成功。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "连接失败。");
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className={`connection-panel ${compact ? "compact" : ""}`}>
      <div className="preset-row">
        {connectionPresets.map((preset) => (
          <button
            key={preset.id}
            className="preset-button"
            type="button"
            onClick={() => {
              setDraft(preset.apiBase);
              onApiBaseChange(preset.apiBase);
            }}
          >
            <strong>{preset.label}</strong>
            <span>{preset.note}</span>
          </button>
        ))}
      </div>

      <label className="field">
        <span>训练服务 API 地址</span>
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="http://127.0.0.1:7863"
        />
      </label>

      <div className="button-row">
        <button
          type="button"
          className="ghost-button"
          onClick={() => testConnection(draft)}
          disabled={testing}
        >
          {testing ? "测试中…" : "测试连接"}
        </button>
        <button
          type="button"
          className="primary-button"
          onClick={() => onApiBaseChange(normalizeApiBase(draft))}
        >
          保存并切换
        </button>
      </div>

      {message ? <div className={health ? "inline-success" : "inline-error"}>{message}</div> : null}
      {health ? (
        <div className="health-grid">
          <KeyValue label="运行模式" value={health.runtime_mode || "—"} />
          <KeyValue label="bundle" value={health.bundle_present ? "已就绪" : "缺失"} />
          <KeyValue label="SPU 配置" value={health.spu_config_present ? "已就绪" : "缺失"} />
          <KeyValue label="runner" value={health.runner_present ? "已就绪" : "缺失"} />
        </div>
      ) : null}
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  tone
}: {
  label: string;
  value: string;
  detail: string;
  tone: "blue" | "green" | "amber" | "slate";
}) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <div className="metric-signal" aria-hidden="true"><span /></div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}

function KeyValue({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="kv-item">
      <div className="kv-label">{label}</div>
      <div className={`kv-value ${mono ? "mono" : ""}`}>{value}</div>
    </div>
  );
}

export default App;
