/**
 * ML Solver - Frontend Logic
 * Author: Musharib Ramzan
 */

// ── ALGO INFO ─────────────────────────────────────────────
const ALGO_INFO = {};
const ALGO_PARAMS = {
  polynomial_regression: ['degree'],
  kmeans: ['k'],
  knn: ['k'],
  random_forest: ['task'],
  svm: ['task'],
  neural_network: ['task'],
};

// ── SAMPLE DATA ───────────────────────────────────────────
const SAMPLES = {
  linear_regression: "1,2\n2,4\n3,5.5\n4,8\n5,10\n6,11.5\n7,14\n8,16\n9,18\n10,20",
  polynomial_regression: "1,1\n2,4\n3,9\n4,16\n5,25\n6,36\n7,49\n8,64\n9,81\n10,100",
  logistic_regression: "1.2,0\n2.1,0\n2.8,0\n3.5,1\n4.2,1\n5.0,1\n1.5,0\n3.1,1\n4.8,1\n2.5,0",
  decision_tree: "1,2,0\n2,3,0\n3,4,1\n5,6,1\n6,7,1\n1,5,0\n4,2,1\n3,3,0\n7,8,1\n2,6,0",
  random_forest: "1,2,0\n2,3,0\n3,4,1\n5,6,1\n4,2,1\n6,8,1\n1,7,0\n3,5,0\n7,9,1\n2,4,0",
  knn: "1,2,0\n2,3,0\n3,4,1\n5,6,1\n4,3,1\n6,7,1\n1,1,0\n8,9,1\n2,5,0\n7,8,1",
  svm: "1,2,0\n2,1,0\n3,4,1\n4,3,1\n5,6,1\n1,3,0\n6,7,1\n2,4,0\n7,8,1\n3,5,1",
  kmeans: "1.1,2.2\n1.5,1.8\n2.0,2.5\n8.0,8.5\n8.5,9.0\n9.0,8.2\n5.0,5.5\n4.8,5.2\n5.3,4.9\n1.3,1.5",
  naive_bayes: "1,2,0\n3,4,1\n5,6,1\n2,3,0\n7,8,1\n1,1,0\n4,5,1\n2,2,0\n6,7,1\n3,3,0",
  statistics: "12 15 18 22 25 28 30 14 16 20 24 26 19 21 17 23 27 29 13 11",
  neural_network: "1,2,0\n2,3,0\n3,4,1\n5,6,1\n4,3,1\n6,7,1\n1,1,0\n8,9,1\n7,8,1\n2,2,0",
};

// ── STATE ─────────────────────────────────────────────────
let mode = 'text', level = 'beginner', currentResult = null, selectedFile = null;

// ── INIT ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadAlgorithms();
  bindEvents();
  loadHistory();
  restoreTheme();
});

function restoreTheme() {
  const saved = localStorage.getItem('mlsolver-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
}

// ── EVENT BINDINGS ─────────────────────────────────────────
function bindEvents() {
  // Tabs
  document.querySelectorAll('.nav-btn[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab, btn));
  });

  // Theme toggle
  document.getElementById('themeToggle').addEventListener('click', toggleTheme);

  // Input mode
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => switchInputMode(btn.dataset.mode, btn));
  });

  // Level toggle
  document.querySelectorAll('.toggle-opt').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle-opt').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      level = btn.dataset.level;
      if (currentResult) renderAdvanced(currentResult);
    });
  });

  // Algo select
  document.getElementById('algoSelect').addEventListener('change', onAlgoChange);

  // Data input debounce detect
  let detectTimer;
  document.getElementById('inputData').addEventListener('input', e => {
    document.getElementById('charCount').textContent = `${e.target.value.length} chars`;
    clearTimeout(detectTimer);
    detectTimer = setTimeout(() => autoDetect(e.target.value), 600);
  });

  // Clear input
  document.getElementById('clearInput').addEventListener('click', () => {
    document.getElementById('inputData').value = '';
    document.getElementById('charCount').textContent = '0 chars';
    document.getElementById('detectBar').style.display = 'none';
  });

  // Sample button
  document.getElementById('sampleBtn').addEventListener('click', loadSample);

  // File upload
  const uploadZone = document.getElementById('uploadZone');
  uploadZone.addEventListener('click', () => document.getElementById('fileInput').click());
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); handleFile(e.dataTransfer.files[0]); });
  document.getElementById('fileInput').addEventListener('change', e => handleFile(e.target.files[0]));

  // Run
  document.getElementById('runBtn').addEventListener('click', runAnalysis);

  // Export / Copy
  document.getElementById('exportBtn')?.addEventListener('click', exportResults);
  document.getElementById('copyBtn')?.addEventListener('click', copyMetrics);

  // History
  document.getElementById('refreshHistory').addEventListener('click', loadHistory);
  document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);
}

// ── TABS ──────────────────────────────────────────────────
function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${tabId}`).classList.add('active');
  btn.classList.add('active');
  if (tabId === 'history') loadHistory();
}

// ── THEME ─────────────────────────────────────────────────
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('mlsolver-theme', next);
}

// ── INPUT MODE ─────────────────────────────────────────────
function switchInputMode(m, btn) {
  mode = m;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('input-text-mode').style.display = m === 'text' ? 'block' : 'none';
  document.getElementById('input-file-mode').style.display = m === 'file' ? 'block' : 'none';
}

// ── LOAD ALGORITHMS ────────────────────────────────────────
async function loadAlgorithms() {
  try {
    const r = await fetch('/api/ml/algorithms');
    const d = await r.json();
    Object.assign(ALGO_INFO, d.algorithms);
  } catch (e) { console.warn('Could not load algorithm info', e); }
}

// ── ALGO CHANGE ───────────────────────────────────────────
function onAlgoChange() {
  const algo = document.getElementById('algoSelect').value;
  const info = ALGO_INFO[algo];
  document.getElementById('algoDesc').textContent = info?.description || '';

  // Show/hide extra params
  const extraParams = document.getElementById('extraParams');
  const params = ALGO_PARAMS[algo] || [];
  const show = params.length > 0;
  extraParams.style.display = show ? 'block' : 'none';
  document.getElementById('degreeParam').style.display = params.includes('degree') ? 'block' : 'none';
  document.getElementById('kParam').style.display = params.includes('k') ? 'block' : 'none';
  document.getElementById('taskParam').style.display = params.includes('task') ? 'block' : 'none';
}

// ── AUTO DETECT ───────────────────────────────────────────
async function autoDetect(text) {
  if (!text.trim() || text.length < 5) { document.getElementById('detectBar').style.display = 'none'; return; }
  try {
    const r = await fetch('/api/ml/detect', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ input_data: text })
    });
    const d = await r.json();
    if (d.suggestions?.length) {
      const bar = document.getElementById('detectBar');
      const chips = document.getElementById('detectChips');
      chips.innerHTML = '';
      d.suggestions.slice(0, 3).forEach(s => {
        const chip = document.createElement('button');
        chip.className = 'detect-chip';
        chip.textContent = ALGO_INFO[s]?.label || s;
        chip.addEventListener('click', () => {
          document.getElementById('algoSelect').value = s;
          onAlgoChange();
        });
        chips.appendChild(chip);
      });
      bar.style.display = 'flex';
    }
  } catch (e) { /* silent */ }
}

// ── FILE HANDLING ─────────────────────────────────────────
function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  document.getElementById('fileInfo').style.display = 'block';
  document.getElementById('fileInfo').textContent = `✓ ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
}

// ── SAMPLE DATA ───────────────────────────────────────────
function loadSample() {
  const algo = document.getElementById('algoSelect').value || 'linear_regression';
  const sample = SAMPLES[algo] || SAMPLES.statistics;
  document.getElementById('inputData').value = sample;
  document.getElementById('charCount').textContent = `${sample.length} chars`;
  autoDetect(sample);
}

// ── RUN ANALYSIS ───────────────────────────────────────────
async function runAnalysis() {
  const algo = document.getElementById('algoSelect').value;
  if (!algo) { showError('Please select an algorithm.'); return; }

  hideError();
  showLoading();

  try {
    const formData = new FormData();
    formData.append('algorithm', algo);

    if (mode === 'file' && selectedFile) {
      formData.append('file', selectedFile);
    } else {
      const text = document.getElementById('inputData').value.trim();
      if (!text) { showError('Please enter some data.'); hideLoading(); return; }
      formData.append('input_data', text);
    }

    // Extra params
    if (algo === 'polynomial_regression') formData.append('degree', document.getElementById('degreeInput').value);
    if (['kmeans','knn'].includes(algo)) formData.append('k', document.getElementById('kInput').value);
    if (ALGO_PARAMS[algo]?.includes('task')) formData.append('task', document.getElementById('taskSelect').value);

    const r = await fetch('/api/ml/run', { method: 'POST', body: formData });
    const d = await r.json();

    if (!r.ok || d.error) { showError(d.error || 'Analysis failed.'); hideLoading(); return; }

    currentResult = d.result;
    hideLoading();
    renderResults(d.result);

  } catch (e) {
    showError('Network error: ' + e.message);
    hideLoading();
  }
}

// ── RENDER RESULTS ─────────────────────────────────────────
function renderResults(result) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('loadingState').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';

  document.getElementById('resultAlgoName').textContent = result.algorithm || '';
  document.getElementById('resultExplain').textContent = result.explanation || '';

  // Metrics
  const mg = document.getElementById('metricsGrid');
  mg.innerHTML = '';
  if (result.metrics) {
    Object.entries(result.metrics).forEach(([k, v], i) => {
      const card = document.createElement('div');
      card.className = 'metric-card';
      card.style.animationDelay = `${i * 0.05}s`;
      card.innerHTML = `<div class="metric-label">${k}</div><div class="metric-value">${v}</div>`;
      mg.appendChild(card);
    });
  }

  // Charts
  const cs = document.getElementById('chartsSection');
  cs.innerHTML = '';
  (result.charts || []).forEach(chart => {
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `<div class="chart-title">${chart.title}</div><img class="chart-img" src="data:image/png;base64,${chart.image}" alt="${chart.title}" loading="lazy"/>`;
    cs.appendChild(card);
  });

  // Equation
  if (result.equation) {
    document.getElementById('equationBox').style.display = 'flex';
    document.getElementById('equationText').textContent = result.equation;
  } else {
    document.getElementById('equationBox').style.display = 'none';
  }

  renderAdvanced(result);
}

function renderAdvanced(result) {
  const adv = document.getElementById('advancedDetails');
  if (level === 'advanced') {
    adv.style.display = 'block';
    const details = {};
    if (result.confusion_matrix) details['Confusion Matrix'] = result.confusion_matrix;
    if (result.classification_report) details['Classification Report'] = result.classification_report;
    if (result.feature_importances) details['Feature Importances'] = result.feature_importances;
    if (result.predictions) details['Sample Predictions (first 20)'] = result.predictions;
    if (result.cluster_labels) details['Cluster Labels (first 30)'] = result.cluster_labels.slice(0,30);
    if (result.cluster_centers) details['Cluster Centers'] = result.cluster_centers;
    document.getElementById('advancedPre').textContent = JSON.stringify(details, null, 2);
  } else {
    adv.style.display = 'none';
  }
}

// ── LOADING / ERROR ───────────────────────────────────────
function showLoading() {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'none';
  document.getElementById('loadingState').style.display = 'flex';
  document.getElementById('runBtn').disabled = true;
}
function hideLoading() {
  document.getElementById('loadingState').style.display = 'none';
  document.getElementById('runBtn').disabled = false;
  if (!currentResult) document.getElementById('resultsPlaceholder').style.display = 'flex';
}
function showError(msg) {
  const box = document.getElementById('errorBox');
  box.textContent = '⚠ ' + msg;
  box.style.display = 'block';
}
function hideError() {
  document.getElementById('errorBox').style.display = 'none';
}

// ── EXPORT ────────────────────────────────────────────────
function exportResults() {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `mlsolver_${currentResult.algorithm?.replace(/\s+/g,'_')}_${Date.now()}.json`;
  a.click();
}

function copyMetrics() {
  if (!currentResult?.metrics) return;
  const text = Object.entries(currentResult.metrics).map(([k,v]) => `${k}: ${v}`).join('\n');
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✓ Copied!';
    setTimeout(() => btn.textContent = '⎘ Copy', 1500);
  });
}

// ── HISTORY ───────────────────────────────────────────────
async function loadHistory() {
  try {
    const r = await fetch('/api/history/');
    const d = await r.json();
    renderHistory(d.history || []);
  } catch (e) {
    document.getElementById('historyList').innerHTML = '<div class="history-empty">Could not load history.</div>';
  }
}

function renderHistory(items) {
  const list = document.getElementById('historyList');
  if (!items.length) {
    list.innerHTML = '<div class="history-empty">No history yet. Run an analysis first.</div>';
    return;
  }
  list.innerHTML = '';
  items.forEach(item => {
    const el = document.createElement('div');
    el.className = 'history-item';
    const metricsHtml = Object.entries(item.metrics || {}).slice(0,4).map(([k,v]) =>
      `<span class="history-metric">${k}: ${v}</span>`).join('');
    el.innerHTML = `
      <span class="history-badge">${item.id}</span>
      <div class="history-content">
        <div class="history-algo">${item.algorithm}</div>
        <div class="history-input">${item.input}</div>
        <div class="history-metrics">${metricsHtml}</div>
      </div>
      <span class="history-time">${item.timestamp}</span>
      <button class="history-del" data-id="${item.id}" title="Delete">✕</button>
    `;
    el.querySelector('.history-del').addEventListener('click', () => deleteHistoryItem(item.id));
    list.appendChild(el);
  });
}

async function deleteHistoryItem(id) {
  await fetch(`/api/history/${id}`, { method: 'DELETE' });
  loadHistory();
}

async function clearHistory() {
  if (!confirm('Clear all history?')) return;
  await fetch('/api/history/clear', { method: 'DELETE' });
  loadHistory();
}
