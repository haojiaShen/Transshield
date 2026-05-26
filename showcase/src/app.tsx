import {
  ArrowSquareOut,
  Article,
  BugBeetle,
  CheckCircle,
  ClockCounterClockwise,
  Cpu,
  FileArrowUp,
  Gauge,
  Heartbeat,
  House,
  ImageSquare,
  Info,
  LockKey,
  PlayCircle,
  Pulse,
  Stack,
  ShieldCheck,
  Sparkle,
  WarningCircle
} from "@phosphor-icons/react";
import { AnimatePresence, motion } from "framer-motion";
import reportContent from "./generated/report_content.json";
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
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";

type RouteKey =
  | "/"
  | "/overview"
  | "/design"
  | "/implementation"
  | "/results"
  | "/innovation"
  | "/reproduce"
  | "/live-demo";

const API_BASE_URL = String(import.meta.env.VITE_TRANSSHIELD_API_BASE_URL ?? "").replace(/\/+$/, "");

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

type DemoStatus =
  | "idle"
  | "worker_preprocessing"
  | "uploading"
  | "server_precheck"
  | "spu_running"
  | "completed"
  | "rejected"
  | "failed";

type WorkerPayload = {
  requestManifest: Record<string, unknown>;
  qualityAssurance: Record<string, unknown>;
  audit: Record<string, unknown>;
  controlPlaneMetrics: Record<string, unknown>;
  share0: Uint8Array;
  share1: Uint8Array;
  previewUrl: string;
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
  formal_metrics: {
    threshold_accuracy: number;
    auc: number;
    sec_per_sample: number;
    dual_total_gib: number;
  };
  demo_boundary_note: string;
  limitations: string[];
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
    formal_metrics: {
      threshold_accuracy: number;
      auc: number;
      sec_per_sample: number;
      dual_total_gib: number;
    };
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

type HealthResponse = {
  status: string;
  runtime_mode: string;
  bundle_present: boolean;
  spu_config_present: boolean;
  runner_present: boolean;
  dist_present: boolean;
  inflight: Record<string, number>;
};

type DemoState = {
  status: DemoStatus;
  selectedFile: File | null;
  previewUrl: string | null;
  localPayload: WorkerPayload | null;
  serverPayload: MedicalRunResponse | null;
  errorMessage: string | null;
};

type DemoAction =
  | { type: "selectFile"; file: File | null; previewUrl: string | null }
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

const navItems: Array<{ path: RouteKey; label: string; icon: ReactNode }> = [
  { path: "/", label: "首页", icon: <House size={18} weight="duotone" /> },
  { path: "/overview", label: "概述", icon: <Article size={18} weight="duotone" /> },
  { path: "/design", label: "设计", icon: <Stack size={18} weight="duotone" /> },
  { path: "/implementation", label: "实现", icon: <Cpu size={18} weight="duotone" /> },
  { path: "/results", label: "结果", icon: <Gauge size={18} weight="duotone" /> },
  { path: "/innovation", label: "创新", icon: <Sparkle size={18} weight="duotone" /> },
  { path: "/reproduce", label: "复现", icon: <ShieldCheck size={18} weight="duotone" /> },
  { path: "/live-demo", label: "现场演示", icon: <PlayCircle size={18} weight="duotone" /> }
];

const workerStatusLabel: Record<DemoStatus, string> = {
  idle: "待命",
  worker_preprocessing: "浏览器侧预处理中",
  uploading: "上传数据分片与结构化摘要",
  server_precheck: "服务端前置校验中",
  spu_running: "安全处理器单元运行中",
  completed: "执行完成",
  rejected: "请求被拒绝",
  failed: "执行失败"
};

const pageTransitions = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -18 }
};

const sectionRouteMap = {
  overview: reportContent.sections.overview,
  design: reportContent.sections.design,
  implementation: reportContent.sections.implementation,
  results: reportContent.sections.results,
  innovation: reportContent.sections.innovation
};

const sectionBriefMap: Record<
  keyof typeof sectionRouteMap,
  {
    eyebrow: string;
    headline: string;
    bullets: string[];
    facts: Array<{ label: string; value: string }>;
    details: string[];
  }
> = {
  overview: {
    eyebrow: "章节摘要",
    headline: "本作品面向高敏感数据场景，构建了兼顾数据隐私与模型隐私的动态剪枝安全推理系统。",
    bullets: [
      "展示站按正式报告章节组织内容，并统一引用既有图件与结果数据。",
      "医疗场景承担现场演示任务，金融场景用于展示既有压力验证结果。",
      "跨越信任边界后仅传输数据分片与结构化摘要，不上传原始图像。"
    ],
    facts: [
      { label: "现场演示场景", value: "仅医疗" },
      { label: "展示范围", value: "医疗与金融" },
      { label: "数据依据", value: "正式结果 JSON" }
    ],
    details: [
      "本作品针对输入数据与模型参数均不宜明文暴露的应用环境，围绕双向隐私保护需求开展系统设计与实现。",
      "展示站保留正式报告的章节结构、图件编号与关键指标，便于评委快速核验系统目标、技术路线与实验结果。",
      "其中，医疗场景提供现场上传与运行入口，金融场景仅用于补充说明方法在另一类高敏感任务中的适配表现。"
    ]
  },
  design: {
    eyebrow: "章节摘要",
    headline: "系统设计围绕信任边界划分、前置校验机制与两方协同安全执行三项核心要求展开。",
    bullets: [
      "浏览器工作线程负责图像解码、裁剪、标准化、质量评估与数据分片生成。",
      "服务端在进入安全执行流程前完成协议结构、张量合法性与重放风险检查。",
      "安全处理器单元（Secure Processing Unit, SPU）仅返回最终分类结果，不暴露中间明文张量。"
    ],
    facts: [
      { label: "路由数", value: "8 个" },
      { label: "上传字段", value: "6 个固定字段" },
      { label: "结果公开范围", value: "仅最终分类输出" }
    ],
    details: [
      "系统按照本地明文处理域、跨边界传输域与两方安全执行域进行划分，明确原始图像的处理终止点与密态数据的传输起点。",
      "浏览器侧仅保留任务编排所需的最小明文处理步骤，服务端则承担结构校验、数据合法性检查与审计记录等控制面职责。",
      "通过上述设计，展示链路能够在保持交互可见性的同时，与正式推理流程保持一致。"
    ]
  },
  implementation: {
    eyebrow: "章节摘要",
    headline: "实现层在复用既有端到端推理链路的基础上，补充了评审所需的展示、演示与控制面能力。",
    bullets: [
      "前端采用独立展示工程承载章节浏览与现场演示，静态图件均来自正式报告抽取结果。",
      "后端保留原始报文预检流程，对 multipart 结构、字段完整性与数据分片一致性进行逐项检查。",
      "浏览器生成的两份数据分片将写入既有执行清单，并复用现有安全推理程序完成单通道执行。"
    ],
    facts: [
      { label: "前端", value: "Vite / React / Tailwind" },
      { label: "后端", value: "FastAPI / Uvicorn" },
      { label: "运行方式", value: "runtime=spu" }
    ],
    details: [
      "前端主线程负责文件选择、状态展示与请求编排，浏览器工作线程承担图像预处理、审计摘要生成与数据分片序列化任务。",
      "后端对原始请求体执行长度门、字段门、哈希校验、张量合法性校验与并发保护，确保进入安全执行流程的输入满足约束条件。",
      "在通过校验后，系统将数据分片转换为既有执行程序可接受的清单形式，并调用现有运行链路完成推理。"
    ]
  },
  results: {
    eyebrow: "章节摘要",
    headline: "结果页重点展示医疗场景的准确性、AUC、运行时延与通信开销，并与正式结果保持一致。",
    bullets: [
      "医疗场景验证结果显示，阈值精度为 92.7481%，AUC 为 0.9639。",
      "完整隐私推理的参考时延为 89.06 秒/样本，双向总通信量为 84.47 GiB。",
      "金融场景仅用于说明压力样本下的一致性与稳定性，不提供现场上传入口。"
    ],
    facts: [
      { label: "验证样本", value: "524 张" },
      { label: "部署批次", value: "32 张" },
      { label: "金融现场上传", value: "关闭" }
    ],
    details: [
      "医疗场景以阈值精度、AUC、端到端时延与通信开销为主要评价指标，相关数值均来自正式结果文件。",
      "金融场景仅用于观察压力样本下的稳定性与一致性，不作为与医疗场景对称的现场演示任务。",
      "展示页中的关键指标与图件均与正式报告保持一致，便于评委进行交叉核验。"
    ]
  },
  innovation: {
    eyebrow: "章节摘要",
    headline: "创新性主要体现在动态剪枝语义的安全化改写、控制面校验链路与可运行演示闭环的协同实现。",
    bullets: [
      "浏览器侧不上传原始图像，跨边界仅发送加法分片与结构化摘要。",
      "服务端执行重放保护、载荷指纹检查、并发限制与数据质量漂移校验。",
      "当前演示部署仍由单个协调服务接收两份数据分片，该边界已在页面中明确说明。"
    ],
    facts: [
      { label: "原始图像上传", value: "不接收" },
      { label: "演示部署边界", value: "单协调服务" },
      { label: "任务中断行为", value: "断连不保证立即终止" }
    ],
    details: [
      "本作品的核心特点在于将动态剪枝中的数据相关决策改写为适用于固定安全计算图的表达方式，以保留按样本变化的判别能力。",
      "在系统层面，作品补充了浏览器侧分片生成、服务端前置校验与审计记录机制，形成可闭环运行的演示路径。",
      "同时需要说明，当前演示部署仍属于展示级实现，尚未拆分为两个独立上传服务。"
    ]
  }
};

function demoReducer(state: DemoState, action: DemoAction): DemoState {
  switch (action.type) {
    case "selectFile":
      return {
        status: "idle",
        selectedFile: action.file,
        previewUrl: action.previewUrl,
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
        previewUrl: action.payload.previewUrl,
        errorMessage: null
      };
    case "complete":
      return { ...state, status: "completed", serverPayload: action.payload, errorMessage: null };
    case "reject":
      return { ...state, status: "rejected", serverPayload: action.payload, errorMessage: action.payload.detail ?? "请求被拒绝" };
    case "fail":
      return { ...state, status: "failed", serverPayload: action.payload ?? null, errorMessage: action.message };
    case "resetRunState":
      return { ...state, status: "idle", localPayload: null, serverPayload: null, errorMessage: null };
    default:
      return state;
  }
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(4)}%`;
}

function formatSeconds(value: number) {
  return `${value.toFixed(2)} 秒`;
}

function formatBytes(bytes: number) {
  const units = ["B", "KiB", "MiB", "GiB"];
  let index = 0;
  let current = bytes;
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }
  return `${current.toFixed(index === 0 ? 0 : 2)} ${units[index]}`;
}

function useMedicalConfig() {
  const [config, setConfig] = useState<MedicalConfigResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const [configResponse, healthResponse] = await Promise.all([
          fetch(apiUrl("/api/medical/config")),
          fetch(apiUrl("/api/health"))
        ]);
        if (!configResponse.ok) {
          throw new Error("无法加载医疗配置");
        }
        const configPayload = (await configResponse.json()) as MedicalConfigResponse;
        const healthPayload = (await healthResponse.json()) as HealthResponse;
        if (!mounted) return;
        setConfig(configPayload);
        setHealth(healthPayload);
      } catch (loadError) {
        if (!mounted) return;
        setError(loadError instanceof Error ? loadError.message : "接口不可用");
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, []);

  return { config, health, error };
}

function createWorker() {
  return new Worker(new URL("./workers/medicalControlPlaneWorker.ts", import.meta.url), {
    type: "module"
  });
}

function App() {
  const location = useLocation();

  return (
    <div className="min-h-[100dvh] bg-slate-50">
      <TopNavigation />
      <main className="mx-auto max-w-[1400px] px-4 pb-16 pt-28 md:px-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            variants={pageTransitions}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
          >
            <Routes location={location}>
              <Route path="/" element={<HomePage />} />
              <Route path="/overview" element={<SectionPage sectionKey="overview" />} />
              <Route path="/design" element={<SectionPage sectionKey="design" />} />
              <Route path="/implementation" element={<SectionPage sectionKey="implementation" />} />
              <Route path="/results" element={<SectionPage sectionKey="results" />} />
              <Route path="/innovation" element={<SectionPage sectionKey="innovation" />} />
              <Route path="/reproduce" element={<ReproducePage />} />
              <Route path="/live-demo" element={<LiveDemoPage />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

function TopNavigation() {
  return (
    <header className="fixed inset-x-0 top-0 z-20 border-b border-slate-200/80 bg-slate-50/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-6 px-4 py-4 md:px-8">
        <Link to="/" className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-teal-200 bg-teal-50 text-accent shadow-[inset_0_1px_0_rgba(255,255,255,0.65)]">
            <ShieldCheck size={24} weight="duotone" />
          </div>
          <div>
            <div className="text-sm uppercase tracking-[0.24em] text-slate-500">TransShield</div>
            <div className="text-base font-semibold tracking-tight text-slate-950">评委展示站与单通道安全推理现场演示</div>
          </div>
        </Link>
        <nav className="hidden items-center gap-1 rounded-full border border-slate-200 bg-white/80 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)] lg:flex">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                [
                  "flex items-center gap-2 rounded-full px-3 py-2 text-base transition-all duration-200",
                  isActive
                    ? "bg-slate-950 text-white"
                    : "text-slate-600 hover:-translate-y-[1px] hover:bg-slate-100 hover:text-slate-950"
                ].join(" ")
              }
            >
              {item.icon}
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <Link
          to="/live-demo"
          className="inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-4 py-2 text-base font-medium text-accent transition-transform duration-200 hover:-translate-y-[1px] active:translate-y-0"
        >
          <PlayCircle size={18} weight="duotone" />
          进入现场演示
        </Link>
      </div>
      <div className="mx-auto flex max-w-[1400px] gap-2 overflow-x-auto px-4 pb-4 lg:hidden">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              [
                "whitespace-nowrap rounded-full border px-3 py-2 text-base",
                isActive ? "border-slate-950 bg-slate-950 text-white" : "border-slate-200 bg-white text-slate-600"
              ].join(" ")
            }
          >
            {item.label}
          </NavLink>
        ))}
      </div>
    </header>
  );
}

function HomePage() {
  const medicalMetrics = [
    { label: "验证集阈值精度", value: "92.75%", note: "524 张验证样本" },
    { label: "判定阈值", value: "0.66196", note: "医疗场景部署使用" },
    { label: "参考时延", value: "89.06 秒/样本", note: "32 张部署验证批次" }
  ];
  const financeMetrics = [
    { label: "压力样本一致性", value: "100.0%", note: "8 条压力样本全部一致" },
    { label: "参数压缩比例", value: "68.39%", note: "压缩模型用于部署" },
    { label: "参考时延", value: "105.16 秒/样本", note: "8 条压力样本批量实测" }
  ];
  const homepageSignals = [
    "正式报告章节展示",
    "医疗场景现场演示",
    "金融场景压力验证"
  ];
  const homepageFlow = [
    { title: "浏览器侧处理", description: "完成预处理、标准化与数据分片生成" },
    { title: "跨边界传输", description: "仅上传数据分片与结构化摘要" },
    { title: "服务端校验", description: "检查字段结构、张量合法性与重放风险" },
    { title: "安全执行", description: "进入安全处理器单元并返回最终结果" }
  ];
  const homepageBoundary = [
    "展示结果统一引用正式结果文件，不额外引入新的实验结论。",
    "服务端不接收原始图像，仅接收两份数据分片及结构化摘要。",
    "安全处理器单元（Secure Processing Unit, SPU）执行阶段平均参考时延约为 89.06 秒/样本。"
  ];

  return (
    <div className="space-y-8">
      <section className="glass-panel overflow-hidden p-8 md:p-10">
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-[1.35fr_0.65fr]">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-sm uppercase tracking-[0.2em] text-accent">
              <ShieldCheck size={16} weight="duotone" />
              基于正式报告与实测结果构建
            </div>
            <div className="space-y-4">
              <h1 className="text-[2.15rem] font-semibold tracking-tight text-slate-950 md:text-[3.45rem] md:leading-[0.98]">
                面向双向隐私保护的动态剪枝安全推理展示系统
              </h1>
              <div className="flex flex-wrap gap-2">
                {homepageSignals.map((signal) => (
                  <span
                    key={signal}
                    className="rounded-full border border-slate-200 bg-white px-3 py-2 text-base text-slate-700"
                  >
                    {signal}
                  </span>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-[1.1fr_0.9fr]">
              <MetricTile
                eyebrow="医疗场景验证结果"
                value="92.7481%"
                note="阈值精度，524 张验证样本"
                icon={<Heartbeat size={20} weight="duotone" />}
              />
              <MetricTile
                eyebrow="完整隐私推理时延"
                value="89.06 秒/样本"
                note="32 张部署验证批次"
                icon={<Gauge size={20} weight="duotone" />}
              />
            </div>
          </div>

          <div className="space-y-4">
            <CompactStatement
              title="作品定位"
              value="集中展示正式报告内容、关键图件与可运行的现场演示闭环。"
            />
            <CompactStatement
              title="现场演示范围"
              value="仅开放医疗图像样本上传；金融场景仅展示既有压力验证结果。"
            />
            <div className="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-4 py-5">
              <div className="text-sm uppercase tracking-[0.18em] text-slate-500">建议起点</div>
              <div className="mt-2 text-base leading-relaxed text-slate-800">
                若希望先确认系统确已可运行，建议从现场演示页开始，再查看结果验证与系统设计。
              </div>
              <div className="mt-4 flex flex-wrap gap-3">
                <Link
                  to="/live-demo"
                  className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-4 py-3 text-base font-medium text-white transition-transform duration-200 hover:-translate-y-[1px] active:translate-y-0"
                >
                  先看现场演示
                  <ArrowSquareOut size={18} weight="duotone" />
                </Link>
                <Link
                  to="/results"
                  className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-3 text-base font-medium text-slate-800 transition-transform duration-200 hover:-translate-y-[1px] active:translate-y-0"
                >
                  查看结果验证
                  <ArrowSquareOut size={18} weight="duotone" />
                </Link>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <QuickFact icon={<LockKey size={18} weight="duotone" />} label="结果公开范围" value="仅最终分类输出" />
          <QuickFact icon={<Pulse size={18} weight="duotone" />} label="鲁棒性矩阵" value="17 / 17 通过" />
          <QuickFact icon={<Cpu size={18} weight="duotone" />} label="通信量" value="84.47 GiB" />
          <QuickFact icon={<ImageSquare size={18} weight="duotone" />} label="现场演示范围" value="仅开放医疗上传" />
        </div>
      </section>

      <section>
        <div className="rounded-[2rem] border border-slate-200 bg-slate-950 p-6 text-white md:p-8">
          <div className="space-y-5">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-sm uppercase tracking-[0.2em] text-teal-200">
              <PlayCircle size={16} weight="duotone" />
              演示流程
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
              {homepageFlow.map((step, index) => (
                <div key={step.title} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
                  <div className="text-sm uppercase tracking-[0.18em] text-slate-300">步骤 {index + 1}</div>
                  <div className="mt-2 text-base font-medium text-white">{step.title}</div>
                  <div className="mt-2 text-sm leading-relaxed text-slate-300">{step.description}</div>
                </div>
              ))}
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 text-base text-slate-300">
              <div className="mb-2 font-medium text-white">演示边界</div>
              <ul className="space-y-2 leading-relaxed">
                {homepageBoundary.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_1fr]">
        <DomainCard
          eyebrow="医疗场景"
          title="医疗影像现场演示"
          summaryLines={[
            "浏览器侧完成图像预处理、质量校验与两份数据分片生成。",
            "服务端仅接收数据分片与结构化摘要，并在安全处理器单元中返回最终分类结果。"
          ]}
          metrics={medicalMetrics}
          route="/live-demo"
          ctaLabel="进入医疗现场演示"
        />
        <DomainCard
          eyebrow="金融场景"
          title="金融风控压力验证展示"
          summaryLines={[
            "展示既有压力样本下的完整隐私推理结果与一致性验证情况。",
            "该场景不提供现场上传入口，仅用于说明方法在另一类高敏感任务中的适配性。"
          ]}
          metrics={financeMetrics}
          route="/results"
          ctaLabel="查看金融验证结果"
        />
      </section>
    </div>
  );
}

function SectionPage({ sectionKey }: { sectionKey: keyof typeof sectionRouteMap }) {
  const section = sectionRouteMap[sectionKey];
  const sectionBrief = sectionBriefMap[sectionKey];
  const figures = reportContent.figures.filter((item) => item.route === sectionKey);
  const [selectedFigure, setSelectedFigure] = useState<(typeof figures)[number] | null>(null);
  const showInlineDetails = sectionKey === "overview";
  const figureSectionClassName =
    figures.length === 1
      ? "grid grid-cols-1 gap-6"
      : figures.length === 2
        ? "grid grid-cols-1 gap-6 xl:grid-cols-2"
        : "grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3";
  const figureCardClassName =
    figures.length === 1
      ? "glass-panel group mx-auto max-w-[1080px] overflow-hidden text-left transition-transform duration-200 hover:-translate-y-[2px]"
      : figures.length === 2
        ? "glass-panel group overflow-hidden text-left transition-transform duration-200 hover:-translate-y-[2px]"
        : "glass-panel group overflow-hidden text-left transition-transform duration-200 hover:-translate-y-[2px]";
  const figureImageClassName =
    figures.length === 1 ? "aspect-[16/8.8] w-full object-cover" : figures.length === 2 ? "aspect-[16/10] w-full object-cover" : "aspect-[4/3] w-full object-cover";
  const figureTextClassName = figures.length === 1 ? "space-y-3 p-6 md:p-7" : "space-y-2 p-5";

  return (
    <div className="space-y-8">
      <section
        className={
          sectionKey === "overview"
            ? "grid grid-cols-1 gap-6"
            : "grid grid-cols-1 gap-6 lg:grid-cols-[1.2fr_0.8fr]"
        }
      >
        <div className="glass-panel p-8 md:p-10">
          <div className="space-y-6">
            <div className="text-sm uppercase tracking-[0.22em] text-slate-500">{sectionBrief.eyebrow}</div>
            <h1 className="section-title">{pageLabelFromKey(sectionKey)}</h1>
            <div className="text-[1.08rem] leading-relaxed text-slate-700">{sectionBrief.headline}</div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {sectionBrief.bullets.map((bullet) => (
                <SignalPanel key={bullet} text={bullet} />
              ))}
            </div>
            {showInlineDetails ? (
              <div className="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-5 py-4 text-base text-slate-600">
                <div className="mb-3 font-medium text-slate-900">章节说明</div>
                <div className="space-y-2 leading-relaxed">
                  {sectionBrief.details.map((paragraph) => (
                    <p key={paragraph}>{paragraph}</p>
                  ))}
                </div>
              </div>
            ) : (
              <details className="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-5 py-4 text-base text-slate-600">
                <summary className="cursor-pointer list-none font-medium text-slate-900">
                  展开章节说明
                </summary>
                <div className="mt-3 space-y-2 leading-relaxed">
                  {sectionBrief.details.map((paragraph) => (
                    <p key={paragraph}>{paragraph}</p>
                  ))}
                </div>
              </details>
            )}
          </div>
        </div>
        {sectionKey === "overview" ? (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <div className="glass-panel p-6 md:p-7">
              <div className="space-y-4">
                <div className="text-sm uppercase tracking-[0.2em] text-slate-500">展示范围</div>
                <div className="space-y-3 text-base text-slate-600">
                  {sectionBrief.facts.map((fact) => (
                    <InfoRow key={fact.label} label={fact.label} value={fact.value} />
                  ))}
                </div>
              </div>
            </div>
            <div className="glass-panel p-6 md:p-7">
              <div className="space-y-4">
                <div className="text-sm uppercase tracking-[0.2em] text-slate-500">医疗关键指标</div>
                <div className="space-y-3 text-base text-slate-600">
                  <InfoRow label="阈值精度" value="92.7481%" />
                  <InfoRow label="AUC" value="0.9639" />
                  <InfoRow label="参考时延" value="89.06 秒/样本" />
                </div>
              </div>
            </div>
            <div className="glass-panel p-6 md:p-7">
              <div className="space-y-4">
                <div className="text-sm uppercase tracking-[0.2em] text-slate-500">演示边界</div>
                <div className="space-y-3 text-base text-slate-600">
                  <InfoRow label="原始图像上传" value="不接收" />
                  <InfoRow label="上传内容" value="数据分片与结构化摘要" />
                  <InfoRow label="金融现场上传" value="关闭" />
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="glass-panel p-6 md:p-8">
            <div className="space-y-4">
              <div className="text-sm uppercase tracking-[0.2em] text-slate-500">章节定位</div>
              <div className="space-y-3 text-base text-slate-600">
                {sectionBrief.facts.map((fact) => (
                  <InfoRow key={fact.label} label={fact.label} value={fact.value} />
                ))}
                {sectionKey === "results" ? (
                  <>
                    <InfoRow label="医疗场景精度" value="92.7481%" />
                    <InfoRow label="医疗 AUC" value="0.9639" />
                    <InfoRow label="参考时延" value="89.06 秒/样本" />
                    <InfoRow label="双向通信量" value="84.47 GiB" />
                  </>
                ) : (
                  <>
                    <InfoRow label="章节来源" value={section.title} />
                    <InfoRow label="图件来源" value="当前正式 docx 内嵌图件抽取" />
                    <InfoRow label="内容组织原则" value="以正式报告与结果文件为依据" />
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </section>
      {figures.length > 0 ? (
        <section className={figureSectionClassName}>
          {figures.map((figure, index) => (
            <motion.button
              key={figure.id}
              type="button"
              layout
              className={figureCardClassName}
              onClick={() => setSelectedFigure(figure)}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.06, duration: 0.24 }}
            >
              <img src={figure.src} alt={figure.caption} className={figureImageClassName} />
              <div className={figureTextClassName}>
                <div className="text-sm uppercase tracking-[0.2em] text-slate-500">{figure.id}</div>
                <div className="text-base font-medium leading-relaxed text-slate-900">{figure.caption}</div>
                <div className="text-sm leading-relaxed text-slate-500">{figure.context}</div>
              </div>
            </motion.button>
          ))}
        </section>
      ) : null}
      <AnimatePresence>
        {selectedFigure ? (
          <motion.button
            type="button"
            className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4"
            onClick={() => setSelectedFigure(null)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              layoutId={selectedFigure.id}
              className="max-h-[92dvh] max-w-[1100px] overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950 p-4 shadow-diffusion"
              initial={{ scale: 0.96 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.96 }}
              onClick={(event) => event.stopPropagation()}
            >
              <img src={selectedFigure.src} alt={selectedFigure.caption} className="max-h-[78dvh] w-full rounded-[1.5rem] object-contain" />
              <div className="space-y-2 px-2 pb-2 pt-4 text-left">
                <div className="text-sm uppercase tracking-[0.2em] text-slate-400">{selectedFigure.id}</div>
                <div className="text-base text-slate-100">{selectedFigure.caption}</div>
              </div>
            </motion.div>
          </motion.button>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function ReproducePage() {
  const steps = [
    "python3 -m pip install -r requirements.txt",
    "cd showcase && npm install && npm run build",
    "uvicorn showcase_api.app:app --host 0.0.0.0 --port 7860",
    "若只跑前端开发态：cd showcase && npm run dev",
    "访问 /live-demo，使用 PNG 或 JPEG 触发医疗单样本演示"
  ];

  const notes = [
    "正式报告成品位于 docs/密捷竞赛作品报告.docx。",
    "当前现场演示页面是在既有控制面证据基础上的可运行重建，不代表新增实验结论。",
    "若本机缺少 SPU / JAX 运行栈，可先设置 TRANSSHIELD_SHOWCASE_RUNTIME_MODE=mock 验证前后端闭环。"
  ];

  return (
    <div className="space-y-8">
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="glass-panel p-8 md:p-10">
          <div className="space-y-5">
            <div className="text-sm uppercase tracking-[0.22em] text-slate-500">最小步骤</div>
            <h1 className="section-title">复现与启动</h1>
            <div className="text-[1.08rem] text-slate-700">只保留最短启动链路，不在这里重复解释整篇报告。</div>
            <div className="space-y-3 rounded-[1.75rem] border border-slate-200 bg-slate-950 p-5 font-mono text-base text-slate-100">
              {steps.map((step) => (
                <div key={step}>{step}</div>
              ))}
            </div>
          </div>
        </div>
        <div className="glass-panel p-6 md:p-8">
          <div className="space-y-4">
            <div className="text-sm uppercase tracking-[0.22em] text-slate-500">说明</div>
            {notes.map((note) => (
              <div key={note} className="flex gap-3 text-base leading-relaxed text-slate-600">
                <ShieldCheck size={18} weight="duotone" className="mt-0.5 shrink-0 text-accent" />
                <span>{note}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function LiveDemoPage() {
  const { config, health, error } = useMedicalConfig();
  const [state, dispatch] = useReducer(demoReducer, {
    status: "idle",
    selectedFile: null,
    previewUrl: null,
    localPayload: null,
    serverPayload: null,
    errorMessage: null
  });
  const [requestActive, setRequestActive] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const serverPrecheckTimerRef = useRef<number | null>(null);
  const spuTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      if (serverPrecheckTimerRef.current) window.clearTimeout(serverPrecheckTimerRef.current);
      if (spuTimerRef.current) window.clearTimeout(spuTimerRef.current);
    };
  }, []);

  const formalMetrics = config?.formal_metrics ?? {
    threshold_accuracy: reportContent.formal_metrics.medical_threshold_accuracy,
    auc: reportContent.formal_metrics.medical_auc,
    sec_per_sample: reportContent.formal_metrics.medical_sec_per_sample,
    dual_total_gib: reportContent.formal_metrics.medical_dual_total_gib
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      dispatch({ type: "selectFile", file: null, previewUrl: null });
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    dispatch({ type: "selectFile", file, previewUrl });
  };

  const handleRun = async (event: FormEvent) => {
    event.preventDefault();
    if (!config) {
      dispatch({ type: "fail", message: "医疗配置尚未加载完成" });
      return;
    }
    if (!state.selectedFile) {
      dispatch({ type: "fail", message: "请先选择一张 PNG 或 JPEG 医疗样本图像" });
      return;
    }
    dispatch({ type: "resetRunState" });
    dispatch({ type: "selectFile", file: state.selectedFile, previewUrl: state.previewUrl });
    dispatch({ type: "setStatus", status: "worker_preprocessing" });
    setRequestActive(true);
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
      serverPrecheckTimerRef.current = window.setTimeout(() => {
        dispatch({ type: "setStatus", status: "server_precheck" });
      }, 120);
      spuTimerRef.current = window.setTimeout(() => {
        dispatch({ type: "setStatus", status: "spu_running" });
      }, 1500);

      const form = new FormData();
      form.append("request_manifest", new Blob([JSON.stringify(message.payload.requestManifest)], { type: "application/json" }));
      form.append("quality_assurance", new Blob([JSON.stringify(message.payload.qualityAssurance)], { type: "application/json" }));
      form.append("audit", new Blob([JSON.stringify(message.payload.audit)], { type: "application/json" }));
      form.append("control_plane_metrics", new Blob([JSON.stringify(message.payload.controlPlaneMetrics)], { type: "application/json" }));
      form.append(
        "share0",
        new Blob([message.payload.share0.slice().buffer], { type: "application/octet-stream" }),
        "share0.bin"
      );
      form.append(
        "share1",
        new Blob([message.payload.share1.slice().buffer], { type: "application/octet-stream" }),
        "share1.bin"
      );

      try {
        const response = await fetch(apiUrl("/api/medical/live-run"), { method: "POST", body: form });
        const payload = (await response.json()) as MedicalRunResponse;
        if (!response.ok || payload.status === "rejected") {
          dispatch({ type: "reject", payload });
          return;
        }
        if (payload.status === "failed") {
          dispatch({ type: "fail", message: payload.detail ?? "SPU 运行失败", payload });
          return;
        }
        dispatch({ type: "complete", payload });
      } catch (uploadError) {
        dispatch({
          type: "fail",
          message: uploadError instanceof Error ? uploadError.message : "请求失败"
        });
      } finally {
        setRequestActive(false);
        worker.terminate();
        workerRef.current = null;
        if (serverPrecheckTimerRef.current) window.clearTimeout(serverPrecheckTimerRef.current);
        if (spuTimerRef.current) window.clearTimeout(spuTimerRef.current);
      }
    };

    worker.postMessage({
      file: state.selectedFile,
      config
    });
  };

  return (
    <div className="space-y-8">
      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1.05fr_0.95fr]">
        <div className="glass-panel p-8 md:p-10">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="space-y-6">
              <div className="space-y-3">
                <div className="inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-sm uppercase tracking-[0.2em] text-accent">
                  <Heartbeat size={16} weight="duotone" />
                  医疗场景现场演示
                </div>
                <h1 className="section-title">浏览器侧分片生成、服务端前置校验与单通道安全执行</h1>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <SignalPanel text="本页仅开放医疗图像样本的现场上传与运行。" />
                  <SignalPanel text="浏览器侧完成预处理、标准化与两份数据分片生成。" />
                  <SignalPanel text="服务端仅接收数据分片与结构化摘要，并进入安全处理器单元执行。" />
                </div>
              </div>

              <form className="space-y-5" onSubmit={(event) => void handleRun(event)}>
                <label className="flex flex-col gap-2">
                  <span className="text-base font-medium text-slate-900">选择医疗样本</span>
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    onChange={handleFileChange}
                    className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3 text-base text-slate-600 file:mr-4 file:rounded-full file:border-0 file:bg-slate-950 file:px-4 file:py-2 file:text-base file:font-medium file:text-white"
                  />
                  <span className="text-sm text-slate-500">
                    当前前端仅接受 PNG / JPEG；原始图像不会跨边界上传，跨边界仅传输数据分片与结构化摘要。
                  </span>
                </label>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <StatusBlock
                    icon={<ClockCounterClockwise size={18} weight="duotone" />}
                    label="正式参考时延"
                    value={`${formalMetrics.sec_per_sample.toFixed(2)} 秒/样本`}
                  />
                  <StatusBlock
                    icon={<LockKey size={18} weight="duotone" />}
                    label="双向通信量"
                    value={`${formalMetrics.dual_total_gib.toFixed(2)} GiB`}
                  />
                </div>

                <div className="rounded-[1.75rem] border border-amber-200 bg-amber-50 p-4 text-base leading-relaxed text-amber-900">
                  请求进入安全处理器单元后，当前演示原型不能保证中途断连即终止任务。页面会如实展示等待状态，不将长时任务包装为瞬时推理。
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="submit"
                    disabled={!state.selectedFile || !config || requestActive}
                    className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-5 py-3 text-base font-medium text-white transition-transform duration-200 hover:-translate-y-[1px] disabled:cursor-not-allowed disabled:opacity-55 active:translate-y-0"
                  >
                    <PlayCircle size={18} weight="duotone" />
                    运行医疗场景演示
                  </button>
                  <button
                    type="button"
                    onClick={() => dispatch({ type: "resetRunState" })}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-3 text-base font-medium text-slate-700 transition-transform duration-200 hover:-translate-y-[1px] active:translate-y-0"
                  >
                    <ArrowSquareOut size={18} weight="duotone" />
                    清空本次状态
                  </button>
                </div>
              </form>
            </div>

            <div className="space-y-4">
              <PreviewPanel previewUrl={state.previewUrl} />
              <HealthPanel config={config} health={health} error={error} />
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="glass-panel p-6 md:p-8">
            <div className="mb-4 flex items-center gap-3">
              <Pulse size={20} weight="duotone" className="text-accent" />
              <div>
                <div className="text-base font-medium text-slate-950">运行状态机</div>
                <div className="text-sm uppercase tracking-[0.2em] text-slate-500">{workerStatusLabel[state.status]}</div>
              </div>
            </div>
            <StatusTimeline activeStatus={state.status} />
          </div>

          <div className="glass-panel p-6 md:p-8">
            <div className="mb-4 flex items-center gap-3">
              <ShieldCheck size={20} weight="duotone" className="text-accent" />
              <div>
                <div className="text-base font-medium text-slate-950">正式结果指标</div>
                <div className="text-sm uppercase tracking-[0.2em] text-slate-500">正式结果文件</div>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <MetricTile eyebrow="阈值精度" value="92.7481%" note="医疗场景正式结果" icon={<Heartbeat size={18} weight="duotone" />} />
              <MetricTile eyebrow="AUC" value="0.9639" note="正式结果文件记录" icon={<Gauge size={18} weight="duotone" />} />
              <MetricTile eyebrow="时延" value="89.06 秒/样本" note="32 张部署验证批次" icon={<ClockCounterClockwise size={18} weight="duotone" />} />
              <MetricTile eyebrow="通信量" value="84.47 GiB" note="双向总通信量" icon={<Cpu size={18} weight="duotone" />} />
            </div>
          </div>

          <ServerVerdictPanel state={state} />
        </div>
      </section>
    </div>
  );
}

function DomainCard({
  eyebrow,
  title,
  summaryLines,
  metrics,
  route,
  ctaLabel
}: {
  eyebrow: string;
  title: string;
  summaryLines: string[];
  metrics: Array<{ label: string; value: string; note: string }>;
  route: RouteKey;
  ctaLabel: string;
}) {
  return (
    <div className="glass-panel p-8">
      <div className="space-y-5">
        <div className="text-sm uppercase tracking-[0.2em] text-slate-500">{eyebrow}</div>
        <div className="space-y-3">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-950">{title}</h2>
          <div className="space-y-2">
            {summaryLines.map((line) => (
              <div key={line} className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 text-base text-slate-700">
                {line}
              </div>
            ))}
          </div>
        </div>
        <div className="space-y-3 border-t border-slate-200 pt-5">
          {metrics.slice(0, 3).map((metric) => (
            <InfoRow key={metric.label} label={metric.label} value={`${metric.value} · ${metric.note}`} />
          ))}
        </div>
        <Link
          to={route}
          className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-4 py-3 text-base font-medium text-white transition-transform duration-200 hover:-translate-y-[1px] active:translate-y-0"
        >
          {ctaLabel}
          <ArrowSquareOut size={18} weight="duotone" />
        </Link>
      </div>
    </div>
  );
}

function CompactStatement({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-4 py-4">
      <div className="text-sm uppercase tracking-[0.18em] text-slate-500">{title}</div>
      <div className="mt-2 text-base leading-relaxed text-slate-800">{value}</div>
    </div>
  );
}

function SignalPanel({ text }: { text: string }) {
  return (
    <div className="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-4 py-4 text-base leading-relaxed text-slate-800">
      {text}
    </div>
  );
}

function MetricTile({
  eyebrow,
  value,
  note,
  icon
}: {
  eyebrow: string;
  value: string;
  note: string;
  icon: ReactNode;
}) {
  return (
    <div className="rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)]">
      <div className="mb-3 flex items-center gap-2 text-sm uppercase tracking-[0.2em] text-slate-500">
        <span className="text-accent">{icon}</span>
        {eyebrow}
      </div>
      <div className="text-2xl font-semibold tracking-tight text-slate-950">{value}</div>
      <div className="mt-2 text-base leading-relaxed text-slate-500">{note}</div>
    </div>
  );
}

function QuickFact({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-3 rounded-[1.35rem] border border-slate-200 bg-white px-4 py-3 text-base shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]">
      <span className="text-accent">{icon}</span>
      <div>
        <div className="text-sm uppercase tracking-[0.2em] text-slate-500">{label}</div>
        <div className="font-medium text-slate-900">{value}</div>
      </div>
    </div>
  );
}

function InfoRow({
  label,
  value,
  stacked = false
}: {
  label: string;
  value: string;
  stacked?: boolean;
}) {
  const pathLike =
    /[\\/]/.test(value) ||
    value.includes(".json") ||
    value.includes(".pth") ||
    value.includes(".pt") ||
    value.includes("artifact") ||
    value.includes("bundle");

  if (stacked) {
    return (
      <div className="border-t border-slate-200/80 pt-3 first:border-t-0 first:pt-0">
        <div className="mb-1 text-sm uppercase tracking-[0.18em] text-slate-500">{label}</div>
        <div
          className={[
            "w-full text-base font-medium leading-relaxed text-slate-900",
            pathLike ? "break-all font-mono text-[0.9rem] text-slate-700" : "break-words"
          ].join(" ")}
        >
          {value}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 border-t border-slate-200/80 pt-3 first:border-t-0 first:pt-0 sm:grid sm:grid-cols-[9rem_minmax(0,1fr)] sm:items-start sm:gap-4">
      <div className="shrink-0 text-base text-slate-500">{label}</div>
      <div
        className={[
          "min-w-0 max-w-full text-base font-medium leading-relaxed text-slate-900 sm:w-full sm:max-w-none sm:text-right",
          pathLike ? "break-all font-mono text-[0.9rem] text-slate-700 sm:text-[0.92rem]" : "break-words"
        ].join(" ")}
      >
        {value}
      </div>
    </div>
  );
}

function pageLabelFromKey(sectionKey: keyof typeof sectionRouteMap) {
  return {
    overview: "作品概述",
    design: "系统设计",
    implementation: "系统实现",
    results: "测试方案与结果分析",
    innovation: "创新性与局限性"
  }[sectionKey];
}

function PreviewPanel({ previewUrl }: { previewUrl: string | null }) {
  return (
    <div className="glass-panel overflow-hidden p-4">
      <div className="mb-3 flex items-center gap-2 text-sm uppercase tracking-[0.2em] text-slate-500">
        <ImageSquare size={18} weight="duotone" />
        样本预览
      </div>
      {previewUrl ? (
        <img src={previewUrl} alt="预览图像" className="aspect-square w-full rounded-[1.5rem] object-cover" />
      ) : (
        <div className="flex aspect-square items-center justify-center rounded-[1.5rem] border border-dashed border-slate-300 bg-slate-100 text-base text-slate-500">
          选择 PNG / JPEG 后在这里预览
        </div>
      )}
    </div>
  );
}

function HealthPanel({
  config,
  health,
  error
}: {
  config: MedicalConfigResponse | null;
  health: HealthResponse | null;
  error: string | null;
}) {
  if (error) {
    return (
      <div className="glass-panel p-4 text-base text-rose-700">
        后端健康检查失败：{error}
      </div>
    );
  }
  return (
    <div className="glass-panel p-4">
      <div className="mb-3 flex items-center gap-2 text-sm uppercase tracking-[0.2em] text-slate-500">
        <ShieldCheck size={18} weight="duotone" />
        运行可见性
      </div>
      <div className="space-y-3 text-base text-slate-600">
        <InfoRow stacked label="模型部署配置" value={config ? "医疗场景现场演示部署包" : "加载中"} />
        <InfoRow stacked label="SPU 配置" value={health?.spu_config_present ? "可见" : "缺失"} />
        <InfoRow stacked label="执行程序" value={health?.runner_present ? "可见" : "缺失"} />
        <InfoRow stacked label="运行模式" value={health?.runtime_mode ?? "未知"} />
        <InfoRow stacked label="当前排队数" value={String(health?.inflight.global_inflight ?? 0)} />
      </div>
    </div>
  );
}

function StatusBlock({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-[1.5rem] border border-slate-200 bg-white px-4 py-4">
      <div className="mb-2 flex items-center gap-2 text-sm uppercase tracking-[0.2em] text-slate-500">
        <span className="text-accent">{icon}</span>
        {label}
      </div>
      <div className="text-xl font-semibold tracking-tight text-slate-950">{value}</div>
    </div>
  );
}

function StatusTimeline({ activeStatus }: { activeStatus: DemoStatus }) {
  const steps: DemoStatus[] = [
    "idle",
    "worker_preprocessing",
    "uploading",
    "server_precheck",
    "spu_running",
    "completed"
  ];
  const activeIndex = steps.indexOf(activeStatus);
  const isRejected = activeStatus === "rejected";
  const isFailed = activeStatus === "failed";

  return (
    <div className="space-y-3">
      {steps.map((step, index) => {
        const done = activeIndex >= index && !isRejected && !isFailed;
        const current = activeStatus === step;
        return (
          <div
            key={step}
            className={[
              "flex items-center gap-3 rounded-[1.35rem] border px-4 py-3 text-base",
              current ? "border-teal-200 bg-teal-50 text-teal-900" : done ? "border-slate-200 bg-white text-slate-700" : "border-slate-200 bg-slate-50 text-slate-500"
            ].join(" ")}
          >
            {current ? (
              <Pulse size={18} weight="duotone" />
            ) : done ? (
              <CheckCircle size={18} weight="duotone" />
            ) : (
              <ClockCounterClockwise size={18} weight="duotone" />
            )}
            <span>{workerStatusLabel[step]}</span>
          </div>
        );
      })}
      {isRejected ? (
        <div className="flex items-center gap-3 rounded-[1.35rem] border border-amber-200 bg-amber-50 px-4 py-3 text-base text-amber-900">
          <WarningCircle size={18} weight="duotone" />
          请求在控制面被拦截
        </div>
      ) : null}
      {isFailed ? (
        <div className="flex items-center gap-3 rounded-[1.35rem] border border-rose-200 bg-rose-50 px-4 py-3 text-base text-rose-800">
          <BugBeetle size={18} weight="duotone" />
          SPU 或后端执行失败
        </div>
      ) : null}
    </div>
  );
}

function ServerVerdictPanel({ state }: { state: DemoState }) {
  const qualityStatus = String(state.serverPayload?.quality_assurance?.status ?? "pending");
  const prediction = state.serverPayload?.result?.prediction;
  const audit = state.serverPayload?.audit as Record<string, unknown> | null;
  const metrics = state.serverPayload?.control_plane_metrics as Record<string, unknown> | null;

  return (
    <div className="glass-panel p-6 md:p-8">
      <div className="mb-4 flex items-center gap-3">
        <ShieldCheck size={20} weight="duotone" className="text-accent" />
        <div>
          <div className="text-base font-medium text-slate-950">审计摘要与服务端裁决</div>
          <div className="text-sm uppercase tracking-[0.2em] text-slate-500">响应字段概览</div>
        </div>
      </div>

      {state.status === "completed" && prediction ? (
        <div className="space-y-5">
          <div className="rounded-[1.75rem] border border-teal-200 bg-teal-50 p-5">
            <div className="mb-2 text-sm uppercase tracking-[0.2em] text-accent">推理结果</div>
            <div className="text-2xl font-semibold tracking-tight text-slate-950">{prediction.threshold_label}</div>
            <div className="mt-2 text-base text-slate-600">
              最大响应类别：{prediction.argmax_label} · 阳性概率：{prediction.prob_class_1.toFixed(4)} · 判定阈值：
              {prediction.decision_threshold.toFixed(5)}
            </div>
          </div>
          <InfoRow label="质量校验结果" value={qualityStatus} />
          <InfoRow label="审计随机数" value={String(audit?.audit_nonce ?? "—")} />
          <InfoRow
            label="服务端预检耗时"
            value={`${Number(metrics?.server_pre_spu_checks_ms ?? 0).toFixed(3)} ms`}
          />
          <InfoRow
            label="本次实际总时长"
            value={formatSeconds(Number(state.serverPayload?.result?.runtime.actual_elapsed_sec ?? 0))}
          />
        </div>
      ) : state.status === "rejected" ? (
        <VerdictMessage
          icon={<WarningCircle size={20} weight="duotone" className="text-amber-700" />}
          title={state.serverPayload?.error_code ?? "请求被拒绝"}
          body={state.serverPayload?.detail ?? state.errorMessage ?? "控制面在前置层拦截了本次请求。"}
          tone="amber"
        />
      ) : state.status === "failed" ? (
        <VerdictMessage
          icon={<BugBeetle size={20} weight="duotone" className="text-rose-700" />}
          title={state.serverPayload?.error_code ?? "执行失败"}
          body={state.errorMessage ?? "后端或 SPU 运行失败。"}
          tone="rose"
        />
      ) : (
        <div className="space-y-4">
          <SkeletonLine widthClass="w-5/12" />
          <SkeletonLine widthClass="w-full" />
          <SkeletonLine widthClass="w-4/5" />
          <div className="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-4 text-base text-slate-500">
            运行完成后，此处将展示推理结果、质量校验结果、审计随机数、服务端预检耗时与本次实际总时长。
          </div>
        </div>
      )}
    </div>
  );
}

function VerdictMessage({
  icon,
  title,
  body,
  tone
}: {
  icon: ReactNode;
  title: string;
  body: string;
  tone: "amber" | "rose";
}) {
  const classes = tone === "amber" ? "border-amber-200 bg-amber-50 text-amber-900" : "border-rose-200 bg-rose-50 text-rose-900";
  return (
    <div className={`rounded-[1.75rem] border p-5 ${classes}`}>
      <div className="mb-3 flex items-center gap-3">
        {icon}
        <div className="font-medium">{title}</div>
      </div>
      <div className="text-base leading-relaxed">{body}</div>
    </div>
  );
}

function SkeletonLine({ widthClass }: { widthClass: string }) {
  return <div className={`h-3 rounded-full bg-slate-200/80 ${widthClass}`} />;
}

export default App;
