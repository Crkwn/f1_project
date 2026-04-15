"use strict";

// -------------------------------------------------------------------
// Bootstrap
// -------------------------------------------------------------------
let DATA = null;

async function load() {
  const res = await fetch("data.json", { cache: "no-cache" });
  DATA = await res.json();
  hydrateMeta();
  hydrateStats();
  hydrateRaces();
  hydrateBuckets();
  hydrateCalibration();
  hydrateMethods();
  bindControls();
}

const pct = (x, d = 1) =>
  x == null ? "—" : `${(x * 100).toFixed(d)}%`;
const fixed = (x, d) =>
  x == null ? "—" : Number(x).toFixed(d);

// -------------------------------------------------------------------
// Top-of-page meta + stats
// -------------------------------------------------------------------
function hydrateMeta() {
  const m = DATA.meta;
  document.getElementById("n-races").textContent = m.n_races;
  document.getElementById("year-range").textContent =
    `${m.holdout_start}–${m.holdout_end}`;
  document.getElementById("foot-years").textContent =
    `${m.holdout_start}–${m.holdout_end}`;
  document.getElementById("foot-tau").textContent = m.tau;
  document.getElementById("foot-mode").textContent = m.mode;
}

function hydrateStats() {
  const m = DATA.meta;
  document.getElementById("stat-pwinner").textContent = pct(m.mean_p_winner, 1);
  document.getElementById("stat-top1").textContent    = pct(m.top_1_rate, 1);
  document.getElementById("stat-top3").textContent    = pct(m.top_3_rate, 1);
  document.getElementById("stat-avgrank").textContent = m.avg_rank_winner;

  const nRaces = m.n_races;
  document.getElementById("stat-top1-total").textContent = nRaces;
  document.getElementById("stat-top1-count").textContent =
    Math.round(m.top_1_rate * nRaces);
}

// -------------------------------------------------------------------
// Race picker
// -------------------------------------------------------------------
function hydrateRaces() {
  const sel = document.getElementById("race-select");
  DATA.races.forEach(r => {
    const opt = document.createElement("option");
    opt.value = r.raceId;
    opt.textContent = `${r.year} R${String(r.round).padStart(2, "0")}  —  ${r.name}`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => selectRace(parseInt(sel.value, 10)));

  // Default selection: first "best" race (model got the winner with highest conf)
  const defaultId = DATA.best_raceids?.[0] ?? DATA.races[0].raceId;
  sel.value = defaultId;
  selectRace(defaultId);
}

function bindControls() {
  document.getElementById("btn-best").addEventListener("click", () => {
    const ids = DATA.best_raceids || [];
    const current = parseInt(document.getElementById("race-select").value, 10);
    const idx = ids.indexOf(current);
    const next = ids[(idx + 1) % ids.length] ?? ids[0];
    if (next != null) pickRace(next);
  });
  document.getElementById("btn-worst").addEventListener("click", () => {
    const ids = DATA.worst_raceids || [];
    const current = parseInt(document.getElementById("race-select").value, 10);
    const idx = ids.indexOf(current);
    const next = ids[(idx + 1) % ids.length] ?? ids[0];
    if (next != null) pickRace(next);
  });
  document.getElementById("btn-random").addEventListener("click", () => {
    const races = DATA.races;
    const next = races[Math.floor(Math.random() * races.length)].raceId;
    pickRace(next);
  });
}

function pickRace(raceId) {
  document.getElementById("race-select").value = raceId;
  selectRace(raceId);
}

function selectRace(raceId) {
  const race = DATA.races.find(r => r.raceId === raceId);
  if (!race) return;
  renderRaceMeta(race);
  renderRaceTable(race);
}

function renderRaceMeta(race) {
  const hit   = race.top_pick_hit;
  const ll    = race.model_log_loss;
  const conf  = race.top_pick_conf;
  const chip  = hit === true  ? `<span class="field-value ok">✓ Correct</span>`
              : hit === false ? `<span class="field-value bad">✗ Missed</span>`
              :                 "—";
  const wn    = race.winner ?? "—";
  const rml   = document.getElementById("race-meta");
  rml.innerHTML = `
    <div><div class="field-label">Actual winner</div>
         <div class="field-value">${wn}</div></div>
    <div><div class="field-label">Model's top pick</div>
         <div class="field-value">${race.top_pick} (${pct(conf, 1)})</div></div>
    <div><div class="field-label">Top pick</div>${chip}</div>
    <div><div class="field-label">Log-loss on winner</div>
         <div class="field-value mono">${fixed(ll, 3)}</div></div>
  `;
}

// -------------------------------------------------------------------
// Driver table for a race
// -------------------------------------------------------------------
function renderRaceTable(race) {
  const body = document.getElementById("race-body");
  body.innerHTML = "";
  race.drivers.forEach((d, i) => {
    const tr = document.createElement("tr");
    if (d.is_winner) tr.classList.add("winner");
    if (i === 0) tr.classList.add("top-pick");

    const actual = d.actual_finish == null ? "—"
      : (d.is_winner ? `<span class="badge win">P${d.actual_finish}</span>`
        : d.actual_finish <= 3 ? `<span class="badge pod">P${d.actual_finish}</span>`
        : `<span>P${d.actual_finish}</span>`);

    const fairCell = d.fair_odds >= 999
      ? `<span class="small">—</span>`
      : fixed(d.fair_odds, 2);
    const bookCell = d.book_odds >= 999
      ? `<span class="small">—</span>`
      : fixed(d.book_odds, 2);

    tr.innerHTML = `
      <td class="col-rank">${i + 1}</td>
      <td class="col-driver">${d.driver}</td>
      <td class="col-team">${d.constructor}</td>
      <td class="col-grid">${d.grid > 0 ? d.grid : "PL"}</td>
      <td class="col-num">${fixed(d.ordinal, 0)}</td>
      <td class="col-num">${fixed(d.reliability, 2)}</td>
      <td class="col-prob">${barPct(d.p_win)}</td>
      <td class="col-prob">${barPct(d.p_podium)}</td>
      <td class="col-prob">${barPct(d.p_points)}</td>
      <td class="col-odds">${fairCell}</td>
      <td class="col-odds">${bookCell}</td>
      <td class="col-result">${actual}</td>
    `;
    body.appendChild(tr);
  });
}

function barPct(p) {
  const w = Math.max(0, Math.min(100, p * 100));
  return `<span class="prob-bar"><span class="prob-bar-fill" style="width:${w}%"></span></span>`
       + `<span class="prob-val">${pct(p, 1)}</span>`;
}

// -------------------------------------------------------------------
// Bucket table
// -------------------------------------------------------------------
function hydrateBuckets() {
  const body = document.querySelector("#bucket-table tbody");
  body.innerHTML = "";
  DATA.bucket_stats.forEach(b => {
    if (b.n === 0) return;
    const tr = document.createElement("tr");
    const hrPct = b.hit_rate == null ? "—" : pct(b.hit_rate, 0);
    const barW  = Math.max(0, Math.min(100, (b.hit_rate ?? 0) * 100));
    tr.innerHTML = `
      <td class="mono">${b.bucket}</td>
      <td class="mono">${b.n}</td>
      <td>
        <div class="bar-cell">
          <div class="bar-track"><span class="bar-fill" style="width:${barW}%"></span></div>
          <span class="mono">${hrPct}</span>
        </div>
      </td>
    `;
    body.appendChild(tr);
  });
}

// -------------------------------------------------------------------
// Calibration plot (SVG)
// -------------------------------------------------------------------
function hydrateCalibration() {
  const svg = document.getElementById("calibration-plot");
  const W = 400, H = 400;
  const pad = { l: 40, r: 14, t: 14, b: 36 };
  const xw = W - pad.l - pad.r, yh = H - pad.t - pad.b;
  const sx = x => pad.l + x * xw;
  const sy = y => pad.t + (1 - y) * yh;

  let svgContent = "";

  // Axes
  svgContent += `<line class="axis-line" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(0)}"/>`;
  svgContent += `<line class="axis-line" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(0)}" y2="${sy(1)}"/>`;

  // Gridlines + labels
  [0, 0.25, 0.5, 0.75, 1.0].forEach(t => {
    svgContent += `<line class="axis-line" x1="${sx(t)}" y1="${sy(0)}" x2="${sx(t)}" y2="${sy(0) + 4}"/>`;
    svgContent += `<text x="${sx(t)}" y="${sy(0) + 18}" text-anchor="middle">${(t*100).toFixed(0)}%</text>`;
    svgContent += `<line class="axis-line" x1="${sx(0)}" y1="${sy(t)}" x2="${sx(0) - 4}" y2="${sy(t)}"/>`;
    svgContent += `<text x="${sx(0) - 8}" y="${sy(t) + 3}" text-anchor="end">${(t*100).toFixed(0)}%</text>`;
  });

  // Ideal line
  svgContent += `<line class="ideal-line" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}"/>`;

  // Points (sized by n)
  const maxN = Math.max(...DATA.calibration.map(b => b.n || 0));
  DATA.calibration.forEach(b => {
    if (b.n === 0 || b.mean_pred == null || b.observed == null) return;
    const r = 3 + 6 * Math.sqrt((b.n || 0) / maxN);
    svgContent += `<circle class="point" cx="${sx(b.mean_pred)}" cy="${sy(b.observed)}" r="${r}">`
               + `<title>predicted ${(b.mean_pred*100).toFixed(1)}% → observed ${(b.observed*100).toFixed(1)}% (n=${b.n})</title>`
               + `</circle>`;
  });

  // Axis labels
  svgContent += `<text x="${W/2}" y="${H - 6}" text-anchor="middle">predicted P(win)</text>`;
  svgContent += `<text x="12" y="${H/2}" text-anchor="middle" transform="rotate(-90 12 ${H/2})">observed win rate</text>`;

  svg.innerHTML = svgContent;
}

// -------------------------------------------------------------------
// Methods comparison
// -------------------------------------------------------------------
function hydrateMethods() {
  const body = document.querySelector("#methods-table tbody");
  body.innerHTML = "";
  // Sort so model rows appear first
  const order = [...DATA.methods].sort((a, b) => {
    const isModel = s => s.method.startsWith("model");
    if (isModel(a) && !isModel(b)) return -1;
    if (!isModel(a) && isModel(b)) return 1;
    return b.mean_p_winner - a.mean_p_winner;
  });

  const maxP = Math.max(...order.map(r => r.mean_p_winner));
  order.forEach(r => {
    const tr = document.createElement("tr");
    const isModel = r.method.startsWith("model");
    if (isModel) tr.style.background = "rgba(225, 6, 0, 0.06)";
    const barW = Math.max(0, Math.min(100, (r.mean_p_winner / maxP) * 100));
    tr.innerHTML = `
      <td>${r.method}</td>
      <td>
        <div class="bar-cell">
          <div class="bar-track"><span class="bar-fill" style="width:${barW}%"></span></div>
          <span class="mono">${pct(r.mean_p_winner, 1)}</span>
        </div>
      </td>
      <td class="mono">${pct(r.top_1, 1)}</td>
      <td class="mono">${pct(r.top_3, 1)}</td>
      <td class="mono">${fixed(r.avg_rank, 2)}</td>
    `;
    body.appendChild(tr);
  });
}

// -------------------------------------------------------------------
load();
