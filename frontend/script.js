// All predictions come from POST /api/predict (Flask + saved ML model). No calculations of demand happen here.
const API_BASE = (location.protocol === "file:" || location.port !== "5000") ? "http://127.0.0.1:5000" : "";
const WATER = "Drinking Water";
const $ = (id) => document.getElementById(id);

let dash = null;          // last /api/dashboard response
let itemsInfo = [];       // /api/items response
const charts = {};

async function api(path, options) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch (e) {
    throw new Error("Cannot reach the backend at " + (API_BASE || location.origin) + ". Is `python app.py` running?");
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    const msg = data ? [data.error].concat(data.details || []).join(" ") : "Request failed (" + res.status + ")";
    throw new Error(msg);
  }
  return data;
}

const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 });
const badgeClass = (s) => (s === "Sufficient" ? "sufficient" : s === "Critical" ? "critical" : "low");
const badge = (s) => `<span class="badge ${badgeClass(s)}">${s}</span>`;
const showError = (msg) => { const el = $("apiError"); el.textContent = msg; el.hidden = !msg; };

/* ---------- navigation ---------- */
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  document.querySelectorAll(".sidebar nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  $("viewTitle").textContent = document.querySelector(`.sidebar nav a[data-view="${name}"]`).firstChild.textContent.trim();
  $("sidebar").classList.remove("open");
  if (dash) renderCharts(); // charts sized inside hidden views need a redraw
}
document.querySelectorAll(".sidebar nav a").forEach((a) =>
  a.addEventListener("click", (e) => { e.preventDefault(); history.replaceState(null, "", "#" + a.dataset.view); showView(a.dataset.view); }));
$("menuBtn").addEventListener("click", () => $("sidebar").classList.toggle("open"));
$("includeWater").addEventListener("change", () => dash && renderCharts());

/* ---------- charts ---------- */
const BLUE = "#1b64d1", NAVY = "#0b2a55", ORANGE = "#e08a1e";
function draw(id, config) {
  if (charts[id]) charts[id].destroy();
  const canvas = $(id);
  if (!canvas) return;
  config.options = Object.assign({ responsive: true, maintainAspectRatio: false }, config.options || {});
  charts[id] = new Chart(canvas, config);
}
const chartItems = () => dash.items.filter((i) => $("includeWater").checked || i.item !== WATER);

function renderCharts() {
  const items = chartItems();
  const labels = items.map((i) => i.item);
  const stockDemand = {
    type: "bar",
    data: { labels, datasets: [
      { label: "Current Stock", data: items.map((i) => i.current_stock), backgroundColor: BLUE },
      { label: "Predicted Demand", data: items.map((i) => i.predicted_next_week_demand), backgroundColor: ORANGE },
    ] },
    options: { scales: { y: { beginAtZero: true } } },
  };
  draw("chStockDemand", stockDemand);
  draw("anStockDemand", JSON.parse(JSON.stringify(stockDemand)));

  const demandCfg = () => {
    const sorted = [...items].sort((a, b) => b.predicted_next_week_demand - a.predicted_next_week_demand);
    return { type: "bar", data: { labels: sorted.map((i) => i.item),
      datasets: [{ label: "Predicted demand (units)", data: sorted.map((i) => i.predicted_next_week_demand), backgroundColor: BLUE }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } } } };
  };
  draw("chDemand", demandCfg());
  draw("anTop", demandCfg());

  const t = dash.trend;
  const labelsT = t.labels.concat([t.forecast_label]);
  const pad = (arr) => arr.concat([null]);
  const trendCfg = () => {
    const ds = [
      { label: "Total demand excl. Drinking Water", data: pad(t.total_excl_water), borderColor: BLUE, backgroundColor: BLUE, tension: .25, pointRadius: 2, yAxisID: "y" },
      { label: "Forecast (excl. water)", data: labelsT.map((_, i) => (i >= labelsT.length - 2 ? (i === labelsT.length - 1 ? t.forecast_excl_water : t.total_excl_water[t.total_excl_water.length - 1]) : null)),
        borderColor: ORANGE, borderDash: [6, 4], pointRadius: 4, yAxisID: "y" },
    ];
    const scales = { y: { beginAtZero: true, title: { display: true, text: "Units (excl. water)" } } };
    if ($("includeWater").checked) {
      ds.push({ label: "Drinking Water", data: pad(t.drinking_water), borderColor: NAVY, tension: .25, pointRadius: 2, yAxisID: "y2" });
      scales.y2 = { position: "right", beginAtZero: true, grid: { drawOnChartArea: false }, title: { display: true, text: "Drinking Water" } };
    }
    return { type: "line", data: { labels: labelsT, datasets: ds }, options: { scales } };
  };
  draw("chTrend", trendCfg());
  draw("anTrend", trendCfg());

  const cats = dash.category_distribution;
  draw("anCat", { type: "doughnut", data: { labels: Object.keys(cats),
    datasets: [{ data: Object.values(cats), backgroundColor: [BLUE, NAVY, ORANGE, "#5aa0f0", "#9db8de"] }] } });
}

/* ---------- dashboard / tables / alerts ---------- */
function renderDashboard() {
  const s = dash.summary;
  $("kpiItems").textContent = s.total_items;
  $("kpiAttention").textContent = s.items_requiring_attention;
  $("kpiStock").textContent = fmt(s.total_current_stock);
  $("kpiDemand").textContent = fmt(s.total_predicted_demand);
  $("kpiWeek").textContent = "(next week)";
  $("alertCount").textContent = dash.alerts.length;
  $("modelInfo").innerHTML = `Model: <b>${dash.model.name}</b><br>R² ${dash.model.metrics.R2} · MAE ${dash.model.metrics.MAE}`;

  $("invTable").querySelector("tbody").innerHTML = dash.items.map((i) => `
    <tr><td><b>${i.item}</b></td><td>${fmt(i.current_stock)}</td><td>${fmt(i.previous_week_consumption)}</td>
    <td>${fmt(i.predicted_next_week_demand)}</td><td>${badge(i.status)}</td><td>${i.recommended_action}</td></tr>`).join("");

  renderRecent(dash.recent_predictions);

  $("alertList").innerHTML = dash.alerts.length
    ? dash.alerts.map((a) => `<li class="${badgeClass(a.severity)}"><span><b>${a.message}</b><br><small>Suggested purchase: ${fmt(a.shortfall)} units</small></span>${badge(a.severity)}</li>`).join("")
    : `<li class="ok"><span><b>All items have sufficient stock for next week.</b></span>${badge("Sufficient")}</li>`;
  renderCharts();
}

function renderRecent(list) {
  $("recentTable").querySelector("tbody").innerHTML = list.length
    ? list.slice(0, 8).map((r) => `<tr><td>${r.item}</td><td>${fmt(r.predicted_next_week_demand)}</td><td>${fmt(r.current_stock)}</td><td>${badge(r.status)}</td></tr>`).join("")
    : `<tr><td class="empty" colspan="4">No predictions yet. Use “Predict Demand”.</td></tr>`;
}

async function loadDashboard() {
  try { dash = await api("/api/dashboard"); showError(""); renderDashboard(); }
  catch (e) { showError(e.message); }
}

/* ---------- prediction form ---------- */
function fillForm(info) {
  const l = info.latest;
  $("week").value = l.week; $("prev1").value = l.previous_week_consumption; $("prev2").value = l.two_weeks_ago_consumption;
  $("prev3").value = l.three_weeks_ago_consumption; $("stock").value = l.current_stock;
  $("occupancy").value = l.hostel_occupancy; $("month").value = l.month;
}

async function loadItems() {
  const data = await api("/api/items");
  itemsInfo = data.items;
  $("item").innerHTML = itemsInfo.map((i) => `<option value="${i.name}">${i.name}</option>`).join("");
  fillForm(itemsInfo[0]);
}
$("fillBtn").addEventListener("click", () => fillForm(itemsInfo.find((i) => i.name === $("item").value)));
$("item").addEventListener("change", () => $("fillBtn").click());

$("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("formError"); err.hidden = true;
  const payload = {
    week: $("week").value, item: $("item").value,
    previous_week_consumption: $("prev1").value, two_weeks_ago_consumption: $("prev2").value,
    three_weeks_ago_consumption: $("prev3").value, current_stock: $("stock").value,
    hostel_occupancy: $("occupancy").value, month: $("month").value,
  };
  for (const k of Object.keys(payload)) if (k !== "item") payload[k] = payload[k] === "" ? null : Number(payload[k]);
  const btn = $("predictBtn"); btn.disabled = true; btn.textContent = "Predicting…";
  try {
    const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    $("resultEmpty").hidden = true; $("resultBody").hidden = false;
    $("rPred").textContent = r.predicted_next_week_demand;
    $("rItem").textContent = r.item;
    $("rStock").textContent = r.current_stock + " units";
    $("rDiff").textContent = (r.stock_difference > 0 ? "+" : "") + r.stock_difference + " units";
    $("rBuy").textContent = r.recommended_purchase + " units";
    const st = $("rStatus"); st.textContent = r.status; st.className = "badge " + badgeClass(r.status);
    loadDashboard(); // refresh "Recent predictions"
  } catch (e2) {
    err.textContent = e2.message; err.hidden = false;
  } finally { btn.disabled = false; btn.textContent = "Predict Demand"; }
});

/* ---------- init ---------- */
(async function init() {
  const hash = location.hash.replace("#", "");
  if (document.querySelector(`.sidebar nav a[data-view="${hash}"]`)) showView(hash);
  try { await loadItems(); } catch (e) { showError(e.message); }
  loadDashboard();
})();
