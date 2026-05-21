const SITE_NAV_ITEMS = [
  { id: 'home', href: '/index.html', label: '首页' },
  { id: 'design', href: '/design.html', label: '系统设计' },
  { id: 'implementation', href: '/implementation.html', label: '系统实现' },
  { id: 'results', href: '/results.html', label: '测试结果' },
  { id: 'demo', href: '/demo.html', label: '现场演示', cta: true },
  { id: 'innovation', href: '/innovation.html', label: '创新与声明' },
];

const FORMAL_EVIDENCE = {
  medicalAccuracy: '92.7481%',
  medicalAuc: '0.9639',
  medicalCommunication: '84.47 GiB',
  financeCommunication: '25.30 GiB',
  robustnessCaseCount: 17,
  robustnessExpectedPass: 17,
  medicalThreshold: '0.6619606018',
};

const PAGE_ENTRY_CARDS = [
  {
    kicker: 'System Design',
    title: '系统设计',
    summary: '查看信任边界、部署拓扑、端到端软件时序和当前安全边界表。',
    href: '/design.html',
    tags: ['图 2-1 部署拓扑', '图 2-2 软件时序', '威胁模型与不覆盖项'],
  },
  {
    kicker: 'Implementation',
    title: '系统实现',
    summary: '查看 DynamicViT 密态改写、控制面分工、快检链路与关键仓内落点。',
    href: '/implementation.html',
    tags: ['F_less / F_mux', '浏览器工作线程', '服务端权威快检'],
  },
  {
    kicker: 'Evidence',
    title: '测试结果',
    summary: '查看医疗正式主线、金融边界压力验证、基线对比与鲁棒性矩阵。',
    href: '/results.html',
    tags: ['524 张验证集', '32 张部署批次', '17 类黑盒用例'],
  },
  {
    kicker: 'Live Demo',
    title: '现场演示',
    summary: '直接触发医疗正式演示或金融边界压力验证，观察完整隐私推理闭环。',
    href: '/demo.html',
    tags: ['医疗上传演示', '金融内置压力样本', '真实运行状态返回'],
  },
  {
    kicker: 'Innovation & Compliance',
    title: '创新与声明',
    summary: '查看正式创新点、当前局限、最低复现路径和数据/模型/许可声明。',
    href: '/innovation.html',
    tags: ['复现路径', '局限性', '第三方许可映射'],
  },
];

const HOME_EVIDENCE_CARDS = [
  {
    title: '医疗正式主线',
    value: '92.7481%',
    note: '524 张全量验证集样本；正式部署阈值为 0.6619606018；AUC 为 0.9639。',
  },
  {
    title: '金融边界压力验证',
    value: '100.0%',
    note: '8 条压力样本逐样本一致；平均时延 105.16s/sample；只承担边界验证职责。',
  },
  {
    title: '协议与控制面守卫',
    value: '17 / 17',
    note: '首个拦截层与兜底层全部落盘；未观察到 FD 或 socket 持续泄漏；允许瞬时 RSS 抬升。',
  },
  {
    title: '最低复现路径',
    value: '10–20 分钟',
    note: '启动演示、运行协议 fuzz、执行至少一项守卫检查，即可验证完整计算流程。',
  },
];

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function pageId() {
  return document.body.dataset.page || 'home';
}

function renderShell() {
  const current = pageId();
  const header = document.getElementById('siteHeader');
  const footer = document.getElementById('siteFooter');
  if (header) {
    header.innerHTML = `
      <div class="shell-header-inner">
        <a class="brand" href="/index.html">
          <div class="brand-mark">密捷</div>
          <div class="brand-copy">
            <strong>密捷 TransShield</strong>
            <span>动态词元剪枝双向隐私安全推理系统</span>
          </div>
        </a>
        <nav class="top-nav" aria-label="主导航">
          ${SITE_NAV_ITEMS.map((item) => `
            <a href="${item.href}" class="${item.id === current ? 'active' : ''} ${item.cta ? 'cta' : ''}">${item.label}</a>
          `).join('')}
        </nav>
      </div>
    `;
  }
  if (footer) {
    footer.innerHTML = `
      <div class="shell-footer-inner">
        <div class="footer-card">
          <div class="card-head">
            <div>
              <div class="mini-kicker">当前正式展示口径</div>
              <h3>密捷 TransShield</h3>
            </div>
            <div class="status-chip">医疗正式主线 / 金融边界压力验证</div>
          </div>
          <p>前端页面与当前竞赛报告口径同步：医疗作为唯一正式主线，金融仅承担边界压力验证；动态剪枝决策链保留在两方安全执行域内，浏览器端只负责本地控制面、分片生成与演示编排。</p>
          <div class="footer-links">
            ${SITE_NAV_ITEMS.map((item) => `<a href="${item.href}">${item.label}</a>`).join('')}
          </div>
        </div>
      </div>
    `;
  }
}

async function loadSummary() {
  const candidates = [
    '/api/demo_summary',
    '/artifacts/web_demo_assets/demo_content_summary.json',
    '../artifacts/web_demo_assets/demo_content_summary.json',
  ];
  for (const url of candidates) {
    try {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) continue;
      return await response.json();
    } catch (error) {
      continue;
    }
  }
  return null;
}

function getDomains(summary) {
  const items = Array.isArray(summary?.showcase_domains?.items) ? summary.showcase_domains.items : [];
  return {
    medical: items.find((item) => item.id === 'medical') || null,
    finance: items.find((item) => item.id === 'finance') || null,
  };
}

function dataCard(label, value, note) {
  return `
    <div class="data-card">
      <strong>${escapeHtml(label)}</strong>
      <span class="data-value">${escapeHtml(value)}</span>
      <p class="data-note">${escapeHtml(note)}</p>
    </div>
  `;
}

function formatNumber(value, digits = 4) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return '—';
  return parsed.toFixed(digits);
}

function privacyLineToHuman(line) {
  const mapping = {
    'host_plaintext_pixel_values_materialized=false': '服务端不接收明文像素值',
    'host_model_params_materialized=false': '模型参数不向数据使用方以明文暴露',
    'reveal_policy=final_logits_only': '对外仅返回最终分类结果',
    'spu_params_mode=secret': '模型参数以密态方式参与执行',
    'input_mode=party_local_debug_share_load': '输入以分片形式进入安全执行域',
  };
  return mapping[line] || line;
}

function renderHome(summary) {
  const { medical, finance } = getDomains(summary);
  const heroMetrics = document.getElementById('homeHeroMetrics');
  if (heroMetrics && medical && finance) {
    heroMetrics.innerHTML = [
      dataCard('医疗正式阈值精度', FORMAL_EVIDENCE.medicalAccuracy, '动态路径单独校准，524 张全量验证集样本'),
      dataCard('医疗正式效率', medical.metrics?.[2]?.value || '86.91s/sample', '32 张正式验证批次'),
      dataCard('金融压力一致性', finance.metrics?.[0]?.value || '100.0%', '8 条压力样本逐样本一致'),
      dataCard('黑盒验证覆盖', `${FORMAL_EVIDENCE.robustnessCaseCount} 类`, `当前 ${FORMAL_EVIDENCE.robustnessExpectedPass}/${FORMAL_EVIDENCE.robustnessCaseCount} 按预期返回`),
    ].join('');
  }

  const routeGrid = document.getElementById('homeRouteGrid');
  if (routeGrid) {
    routeGrid.innerHTML = PAGE_ENTRY_CARDS.map((item) => `
      <article class="card route-card">
        <div class="card-head">
          <div>
            <div class="mini-kicker">${escapeHtml(item.kicker)}</div>
            <h3>${escapeHtml(item.title)}</h3>
          </div>
        </div>
        <p class="lead route-summary">${escapeHtml(item.summary)}</p>
        <div class="route-tags">
          ${item.tags.map((tag) => `<span class="route-tag">${escapeHtml(tag)}</span>`).join('')}
        </div>
        <a class="route-link" href="${escapeHtml(item.href)}">进入本页</a>
      </article>
    `).join('');
  }

  const domainCards = document.getElementById('homeDomainCards');
  if (domainCards && medical && finance) {
    domainCards.innerHTML = [medical, finance].map((item) => `
      <article class="card">
        <div class="card-head">
          <div>
            <div class="mini-kicker">${escapeHtml(item.eyebrow || '')}</div>
            <h3>${escapeHtml(item.label)}：${escapeHtml(item.headline || '')}</h3>
          </div>
          <div class="status-chip ${item.id === 'finance' ? 'warn' : ''}">${item.id === 'medical' ? '正式主线' : '边界压力验证'}</div>
        </div>
        <p class="lead">${escapeHtml(item.summary || '')}</p>
        <div class="metric-grid cols-2">
          ${((item.metrics || []).slice(0, 4)).map((metric, index) => {
            if (item.id === 'medical' && index === 0) {
              return dataCard('验证集阈值精度', FORMAL_EVIDENCE.medicalAccuracy, '动态路径单独校准，524 张全量验证集样本');
            }
            if (item.id === 'medical' && index === 1) {
              return dataCard('正式判类阈值', FORMAL_EVIDENCE.medicalThreshold, '用于医疗正式部署');
            }
            return dataCard(metric.label, metric.value, metric.note);
          }).join('')}
        </div>
        <div class="page-card-grid">
          <div class="scope-box">
            <strong>展示定位</strong>
            <p>${escapeHtml(item.display_notes?.[0] || '')}</p>
          </div>
          <div class="note-box">
            <strong>隐私边界</strong>
            <ul class="list tight">
              ${(item.privacy || []).map((line) => `<li>${escapeHtml(privacyLineToHuman(line))}</li>`).join('')}
            </ul>
          </div>
        </div>
      </article>
    `).join('');
  }

  const evidenceGrid = document.getElementById('homeEvidenceGrid');
  if (evidenceGrid) {
    evidenceGrid.innerHTML = HOME_EVIDENCE_CARDS.map((item) => `
      <article class="card evidence-card">
        <div class="mini-kicker">Formal Evidence</div>
        <h3>${escapeHtml(item.title)}</h3>
        <div class="data-value">${escapeHtml(item.value)}</div>
        <p class="data-note">${escapeHtml(item.note)}</p>
      </article>
    `).join('');
  }

  const narrative = document.getElementById('homeNarrativeSteps');
  if (narrative) {
    const steps = Array.isArray(summary?.competition_narrative?.steps) ? summary.competition_narrative.steps : [];
    narrative.innerHTML = steps.map((step, index) => `
      <article class="timeline-card">
        <div class="timeline-no">${index + 1}</div>
        <div class="timeline-copy">
          <h4>${escapeHtml(step.title || `步骤 ${index + 1}`)}</h4>
          <p class="muted">${escapeHtml(step.summary || '')}</p>
        </div>
      </article>
    `).join('');
  }

  const benchmark = document.getElementById('homeBenchmarkStrip');
  if (benchmark) {
    benchmark.innerHTML = `
      <div class="comparison-row">
        <div>
          <span>核心算法</span>
          <strong>协议友好重写</strong>
        </div>
        <div>
          <span>当前做法</span>
          <p>把删 token、阈值比较和数据相关 Top-K 分别改写为掩码化表达、安全比较与编码键双调排序。</p>
        </div>
        <div>
          <span>工程意义</span>
          <p>保留按样本变化的动态剪枝能力，而不是退回固定结构安全推理。</p>
        </div>
      </div>
      <div class="comparison-row">
        <div>
          <span>系统形态</span>
          <strong>双向隐私 2PC 原型</strong>
        </div>
        <div>
          <span>当前做法</span>
          <p>服务端不接收明文像素值，模型参数不向数据使用方以明文暴露，动态决策链保留在安全执行域内。</p>
        </div>
        <div>
          <span>工程意义</span>
          <p>避免“外部先算路径、内部只回放”的半隐私运行。</p>
        </div>
      </div>
      <div class="comparison-row">
        <div>
          <span>展示闭环</span>
          <strong>控制面 + 审计链</strong>
        </div>
        <div>
          <span>当前做法</span>
          <p>浏览器工作线程、本地 DQA、服务端权威快检、协议 fuzz 和重放守卫共同纳入交付范围。</p>
        </div>
        <div>
          <span>工程意义</span>
          <p>让评审能同时看到分类结果、阻断位置和资源回落状态。</p>
        </div>
      </div>
    `;
  }
}

function renderImplementation(summary) {
  const operatorCards = document.getElementById('implementationOperatorCards');
  if (operatorCards) {
    const items = Array.isArray(summary?.operator_replacements?.items) ? summary.operator_replacements.items : [];
    operatorCards.innerHTML = items.map((item) => `
      <article class="data-card">
        <strong>${escapeHtml(item.name || '')}</strong>
        <span class="data-value">${escapeHtml(item.after || '')}</span>
        <p class="data-note">原始形式：${escapeHtml(item.before || '')}</p>
        <p class="data-note">${escapeHtml(item.effect || '')}</p>
        <p class="caption">适用范围：${escapeHtml(item.scope || '')}</p>
      </article>
    `).join('');
  }
}

function renderResults(summary) {
  const { medical, finance } = getDomains(summary);
  const comparisonBody = document.getElementById('resultsComparisonRows');
  if (comparisonBody) {
    const rows = Array.isArray(summary?.external_comparison?.additional_rows) ? summary.external_comparison.additional_rows : [];
    const extraColumns = {
      'Transshield（ours）': ['是', '是'],
      'Transshield（static control）': ['是', '否'],
      'MPCViT [3]': ['否', '否'],
      'DeiT-S Static': ['否', '否'],
      'Original DynamicViT': ['否', '是'],
    };
    comparisonBody.innerHTML = rows.map((row) => {
      const [privacy, dynamic] = extraColumns[row.method] || ['—', '—'];
      return `
        <tr>
          <td>${escapeHtml(row.method || '')}</td>
          <td>${escapeHtml(privacy)}</td>
          <td>${escapeHtml(dynamic)}</td>
          <td>${escapeHtml(formatNumber(row.threshold_accuracy, 4))}%</td>
          <td>${escapeHtml(formatNumber(row.auc, 4))}</td>
          <td>${escapeHtml(row.note || '')}</td>
        </tr>
      `;
    }).join('');
  }

  const medicalKpis = document.getElementById('resultsMedicalKpis');
  if (medicalKpis && medical) {
    medicalKpis.innerHTML = [
      dataCard('全量验证集阈值精度', FORMAL_EVIDENCE.medicalAccuracy, '正式主线主指标'),
      dataCard('部署阈值', FORMAL_EVIDENCE.medicalThreshold, '动态路径单独阈值校准'),
      dataCard('32 条样本平均时延', medical.metrics?.[2]?.value || '86.91s/sample', '批次规模 8，深度 10'),
      dataCard('32 条样本双向通信量', FORMAL_EVIDENCE.medicalCommunication, '同配置重测记录'),
    ].join('');
  }

  const financeKpis = document.getElementById('resultsFinanceKpis');
  if (financeKpis && finance) {
    financeKpis.innerHTML = [
      dataCard('8 条压力样本一致性', finance.metrics?.[0]?.value || '100.0%', '与明文参考逐样本对齐'),
      dataCard('压力样本平均时延', finance.metrics?.[2]?.value || '105.16s/sample', '仅作为边界压力验证口径'),
      dataCard('双向总通信量', FORMAL_EVIDENCE.financeCommunication, '同配置重测口径'),
      dataCard('参数规模保留比例', '68.39%', '体现低秩压缩收益'),
    ].join('');
  }

  const proxyBody = document.getElementById('resultsProxyRows');
  if (proxyBody) {
    const comparisons = Array.isArray(summary?.standardized_secure_benchmark?.comparisons) ? summary.standardized_secure_benchmark.comparisons : [];
    proxyBody.innerHTML = comparisons.map((item) => `
      <tr>
        <td>${escapeHtml(item.comparison_group || '')}</td>
        <td>${escapeHtml(item.left_display_name || '')}</td>
        <td>${escapeHtml(item.right_display_name || '')}</td>
        <td>${escapeHtml((item.module_comm_ratio_left_over_right ?? 0).toFixed(3))}×</td>
        <td>${escapeHtml((item.time_ratio_left_over_right ?? 0).toFixed(3))}×</td>
        <td>${escapeHtml(item.scope_note || '')}</td>
      </tr>
    `).join('');
  }

  const robustnessStats = document.getElementById('resultsRobustnessKpis');
  if (robustnessStats) {
    robustnessStats.innerHTML = [
      dataCard('黑盒用例总数', `${FORMAL_EVIDENCE.robustnessCaseCount}`, '协议层异常输入与控制面守卫合计'),
      dataCard('按预期返回', `${FORMAL_EVIDENCE.robustnessExpectedPass}/${FORMAL_EVIDENCE.robustnessCaseCount}`, '全部用例均返回预期拦截或限制结果'),
      dataCard('句柄回落', '未观察到 FD / Socket 净增长', '句柄类资源回到基线附近'),
      dataCard('资源状态描述', '存在瞬时 RSS 抬升', '不包装为零成本防护'),
    ].join('');
  }
}

function renderInnovation() {
  const innovationCore = document.getElementById('innovationCore');
  if (innovationCore) {
    innovationCore.innerHTML = `
      <article class="data-card">
        <strong>算法层</strong>
        <span class="data-value">pruning boundary 的协议友好重写</span>
        <p class="data-note">将删 token、阈值比较与 Top-K 选择改写为 F_mux、F_less 与编码键双调排序，保留按样本变化的动态剪枝能力。</p>
      </article>
      <article class="data-card">
        <strong>系统层</strong>
        <span class="data-value">PredictorLG in-SPU + 双向隐私 runtime</span>
        <p class="data-note">动态剪枝预测器、阈值判断与并列分数决断留在安全执行域内，避免退化为半隐私运行。</p>
      </article>
      <article class="data-card">
        <strong>工程层</strong>
        <span class="data-value">浏览器工作线程控制面 + 服务端权威快检</span>
        <p class="data-note">前端本地 DQA、审计哈希链、服务端张量快检、协议 fuzz 与重放守卫共同形成可验证交付闭环。</p>
      </article>
    `;
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  renderShell();
  const summary = await loadSummary();
  if (!summary) return;
  const current = pageId();
  if (current === 'home') renderHome(summary);
  if (current === 'implementation') renderImplementation(summary);
  if (current === 'results') renderResults(summary);
  if (current === 'innovation') renderInnovation(summary);
});
