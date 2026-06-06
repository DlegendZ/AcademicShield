// Academic Shield — fully client-side inference + UI.
// All prediction happens in the browser; zero network on the hot path.
import { scoreModelA } from "./models/model_a.js";
import { scoreModelB } from "./models/model_b.js";

// ---------------------------------------------------------------------------
// Config — mirrors src/config.py exactly
// ---------------------------------------------------------------------------
const STRESS_MAPPING_A = { Low: 2.29, Moderate: 4.80, High: 7.42 };
const STRESS_MAPPING_B = { Low: 0, Moderate: 1, High: 2 };
const BURNOUT_CLASSES = { 0: "Healthy", 1: "Mildly Burnout", 2: "Burnout" };
const BURNOUT_SCORE_WEIGHTS = [20, 55, 85];
const GPA_LABELS = [
  [3.5, "Excellent! Keep it up."],
  [3.0, "Good job!"],
  [2.5, "Not so great."],
  [0.0, "Needs improvement."],
];
const INSIGHTS = {
  2: "Your burnout level is high. Try increasing your sleep to at least 7 hours, reduce study overload, and schedule regular breaks throughout the day.",
  1: "You're showing early signs of stress. Make sure to schedule breaks, maintain social connections, and balance your workload intentionally.",
  0: "You're in good shape! Keep maintaining this balanced lifestyle and stay consistent with your habits.",
};
const COLORS = {
  accent: "#D97757", text: "#2D2D2D", muted: "#9B9B9B",
  healthy: "#4AA785", mild: "#D4A843", burnout: "#C75050",
};
// select_slider rating -> numeric (Page_2.py RATING_TO_SCORE)
const RATING_TO_SCORE = { "Very Low": 0.0, "Low": 2.5, "Moderate": 5.0, "High": 7.5, "Very High": 10.0 };
const RATING_OPTIONS = ["Very Low", "Low", "Moderate", "High", "Very High"];
// Model B clamp ranges: key -> [lo, hi, trainingMean]
const B_RANGES = {
  study_hours: [5.0, 10.0, 7.48], sleep_hours: [5.0, 10.0, 7.50],
  eca_hours: [0.0, 4.0, 1.99], social_hours: [0.0, 6.0, 2.70],
  physical_hours: [0.0, 13.0, 4.33],
};

// ---------------------------------------------------------------------------
// Inference — ports of src/models.py predict_burnout / predict_gpa
// ---------------------------------------------------------------------------
function predictBurnout(inp) {
  const stress = STRESS_MAPPING_A[inp.stress_level_category];
  const f = {
    study_hours_per_day: inp.study_hours, sleep_hours: inp.sleep_hours,
    exam_pressure: inp.exam_pressure, stress_level: stress,
    financial_stress: inp.financial_stress, social_support: inp.social_support,
    anxiety_score: inp.anxiety_score, depression_score: inp.depression_score,
    family_expectation: inp.family_expectation, physical_activity: inp.physical_hours,
  };
  // engineer_features_a (src/features.py) — order MUST match training
  const vec = [
    f.study_hours_per_day, f.sleep_hours, f.exam_pressure, f.stress_level,
    f.financial_stress, f.social_support, f.anxiety_score, f.depression_score,
    f.family_expectation, f.physical_activity,
    f.stress_level * f.anxiety_score,                                  // stress_x_anxiety
    f.stress_level * f.depression_score,                              // stress_x_depression
    f.stress_level * f.exam_pressure,                                 // stress_x_exam
    f.anxiety_score * f.depression_score,                             // anxiety_x_depression
    f.study_hours_per_day / (f.sleep_hours + 1),                      // study_sleep_ratio
    ((f.exam_pressure + f.financial_stress + f.family_expectation) / 3) - f.social_support, // pressure_support_gap
    f.stress_level ** 2,                                              // stress_level_sq
    f.anxiety_score ** 2,                                             // anxiety_score_sq
    f.sleep_hours < 5 ? 1 : 0,                                        // sleep_deprived
    f.stress_level > 7 ? 1 : 0,                                       // high_stress
  ];
  const proba = scoreModelA(vec);
  const classId = proba.indexOf(Math.max(...proba));
  let score = proba.reduce((s, p, i) => s + p * BURNOUT_SCORE_WEIGHTS[i], 0);
  score = Math.round(Math.max(0, Math.min(100, score)) * 10) / 10;
  return { classId, className: BURNOUT_CLASSES[classId], probabilities: proba, score };
}

function predictGpa(inp) {
  const clamp = (k) => {
    const [lo, hi, mean] = B_RANGES[k];
    const v = inp[k];
    return v < lo || v > hi ? mean : v;
  };
  const vec = [
    clamp("study_hours"), clamp("eca_hours"), clamp("sleep_hours"),
    clamp("social_hours"), clamp("physical_hours"),
    STRESS_MAPPING_B[inp.stress_level_category],
  ];
  let gpa = Math.round(scoreModelB(vec) * 100) / 100;
  gpa = Math.max(0, Math.min(4, gpa));
  let label = GPA_LABELS[GPA_LABELS.length - 1][1];
  for (const [t, l] of GPA_LABELS) { if (gpa >= t) { label = l; break; } }
  return { gpa, label };
}

// ---------------------------------------------------------------------------
// SVG chart helpers
// ---------------------------------------------------------------------------
function polar(cx, cy, r, deg) {
  const a = (deg - 90) * Math.PI / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}
function arcPath(cx, cy, r, startDeg, endDeg) {
  const [sx, sy] = polar(cx, cy, r, endDeg);
  const [ex, ey] = polar(cx, cy, r, startDeg);
  const large = endDeg - startDeg <= 180 ? 0 : 1;
  return `M ${sx} ${sy} A ${r} ${r} 0 ${large} 0 ${ex} ${ey}`;
}

function burnoutGauge(score, className, classId) {
  // Semicircle: value 0..100 maps to angle 180deg (left) .. 360deg (right)
  const cx = 150, cy = 150, r = 110;
  const ang = (v) => 180 + (v / 100) * 180;
  const col = [COLORS.healthy, COLORS.mild, COLORS.burnout][classId] || COLORS.text;
  const seg = (a, b, c) =>
    `<path d="${arcPath(cx, cy, r, ang(a), ang(b))}" stroke="${c}" stroke-width="22" fill="none" stroke-linecap="butt"/>`;
  const valArc = `<path d="${arcPath(cx, cy, r, ang(0), ang(score))}" stroke="${COLORS.text}" stroke-width="8" fill="none" stroke-linecap="round" opacity="0.85"/>`;
  const [nx, ny] = polar(cx, cy, r, ang(score));
  return `
  <svg viewBox="0 0 300 200" width="100%" style="max-width:340px">
    ${seg(0, 40, "rgba(74,167,133,0.30)")}
    ${seg(40, 70, "rgba(212,168,67,0.30)")}
    ${seg(70, 100, "rgba(199,80,80,0.30)")}
    ${valArc}
    <circle cx="${nx}" cy="${ny}" r="7" fill="${col}"/>
    <text x="${cx}" y="135" text-anchor="middle" font-size="46" font-weight="600" fill="${COLORS.text}" font-family="Newsreader,serif">${Math.round(score)}</text>
    <text x="${cx}" y="172" text-anchor="middle" font-size="18" font-weight="600" fill="${col}" font-family="Newsreader,serif">${className}</text>
  </svg>`;
}

function gpaRing(gpa, label) {
  const r = 54, c = 2 * Math.PI * r, off = c * (1 - (gpa / 4));
  const col = gpa >= 3.0 ? COLORS.healthy : gpa >= 2.5 ? COLORS.mild : COLORS.burnout;
  return `
  <div style="text-align:center">
    <p class="ring-label">Predicted GPA</p>
    <div style="position:relative;width:150px;height:150px;margin:0 auto">
      <svg width="150" height="150" viewBox="0 0 120 120" style="transform:rotate(-90deg)">
        <circle cx="60" cy="60" r="${r}" fill="none" stroke="rgba(0,0,0,0.07)" stroke-width="6"/>
        <circle cx="60" cy="60" r="${r}" fill="none" stroke="${col}" stroke-width="6" stroke-linecap="round"
          stroke-dasharray="${c}" stroke-dashoffset="${c}">
          <animate attributeName="stroke-dashoffset" from="${c}" to="${off}" dur="1.4s" fill="freeze"
            calcMode="spline" keySplines="0.4 0 0.2 1"/>
        </circle>
      </svg>
      <div class="ring-center">
        <span style="font-size:2rem;font-weight:700;color:${col};display:block">${gpa.toFixed(2)}</span>
        <span style="font-size:0.72rem;color:#9B9B9B">/ 4.0</span>
      </div>
    </div>
    <p style="margin-top:16px;color:#6B6B6B;font-style:italic">${label}</p>
  </div>`;
}

function radarChart(inp) {
  const stressMap = { Low: 2.0, Moderate: 5.0, High: 8.0 };
  const cats = ["Study", "Sleep", "Exercise", "Social", "ECA", "Stress",
    "Exam", "Family", "Finance", "Support", "Anxiety", "Depression"];
  const vals = [
    Math.min(inp.study_hours * 10 / 12, 10), Math.min(inp.sleep_hours * 10 / 12, 10),
    Math.min(inp.physical_hours * 10 / 13, 10), Math.min(inp.social_hours * 10 / 6, 10),
    Math.min(inp.eca_hours * 10 / 4, 10), stressMap[inp.stress_level_category],
    inp.exam_pressure, inp.family_expectation, inp.financial_stress,
    inp.social_support, inp.anxiety_score, inp.depression_score,
  ];
  const cx = 170, cy = 170, R = 120, n = cats.length;
  const grid = [2, 4, 6, 8, 10].map(g => {
    const pts = cats.map((_, i) => polar(cx, cy, R * g / 10, i * 360 / n).join(",")).join(" ");
    return `<polygon points="${pts}" fill="none" stroke="rgba(0,0,0,0.06)"/>`;
  }).join("");
  const axes = cats.map((_, i) => {
    const [x, y] = polar(cx, cy, R, i * 360 / n);
    return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(0,0,0,0.05)"/>`;
  }).join("");
  const dataPts = vals.map((v, i) => polar(cx, cy, R * Math.max(0, v) / 10, i * 360 / n).join(",")).join(" ");
  const labels = cats.map((c, i) => {
    const [x, y] = polar(cx, cy, R + 22, i * 360 / n);
    return `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="#6B6B6B" font-family="Newsreader,serif">${c}</text>`;
  }).join("");
  return `
  <svg viewBox="0 0 340 340" width="100%" style="max-width:380px">
    ${grid}${axes}
    <polygon points="${dataPts}" fill="rgba(217,119,87,0.14)" stroke="${COLORS.accent}" stroke-width="2"/>
    ${vals.map((v, i) => { const [x, y] = polar(cx, cy, R * Math.max(0, v) / 10, i * 360 / n); return `<circle cx="${x}" cy="${y}" r="3" fill="${COLORS.accent}"/>`; }).join("")}
    ${labels}
  </svg>`;
}

function probabilityBars(proba) {
  const names = ["Healthy", "Mildly Burnout", "Burnout"];
  const cols = [COLORS.healthy, COLORS.mild, COLORS.burnout];
  return names.map((nm, i) => `
    <div class="prob-row">
      <span class="prob-name">${nm}</span>
      <div class="prob-track"><div class="prob-fill" style="width:${(proba[i] * 100).toFixed(1)}%;background:${cols[i]}"></div></div>
      <span class="prob-val">${(proba[i] * 100).toFixed(1)}%</span>
    </div>`).join("");
}

// ---------------------------------------------------------------------------
// App state + navigation
// ---------------------------------------------------------------------------
const state = {
  study_hours: 4.0, sleep_hours: 8.0, eca_hours: 1.0, social_hours: 3.0, physical_hours: 2.0,
  stress_level_category: "Moderate",
  exam_pressure_rating: "High", family_expectation_rating: "Moderate",
  financial_stress_rating: "Low", social_support_rating: "High",
  anxiety_score_rating: "Moderate", depression_score_rating: "Very Low",
};

function show(id) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function runResults() {
  const inp = {
    study_hours: state.study_hours, sleep_hours: state.sleep_hours,
    eca_hours: state.eca_hours, social_hours: state.social_hours,
    physical_hours: state.physical_hours,
    stress_level_category: state.stress_level_category,
    exam_pressure: RATING_TO_SCORE[state.exam_pressure_rating],
    family_expectation: RATING_TO_SCORE[state.family_expectation_rating],
    financial_stress: RATING_TO_SCORE[state.financial_stress_rating],
    social_support: RATING_TO_SCORE[state.social_support_rating],
    anxiety_score: RATING_TO_SCORE[state.anxiety_score_rating],
    depression_score: RATING_TO_SCORE[state.depression_score_rating],
  };
  const t0 = performance.now();
  const burnout = predictBurnout(inp);
  const gpa = predictGpa(inp);
  const ms = (performance.now() - t0).toFixed(2);

  document.getElementById("gauge").innerHTML = burnoutGauge(burnout.score, burnout.className, burnout.classId);
  document.getElementById("gpa").innerHTML = gpaRing(gpa.gpa, gpa.label);
  document.getElementById("radar").innerHTML = radarChart(inp);
  document.getElementById("probbars").innerHTML = probabilityBars(burnout.probabilities);
  const col = { 0: COLORS.healthy, 1: COLORS.mild, 2: COLORS.burnout }[burnout.classId];
  const card = document.getElementById("insight");
  card.style.borderLeftColor = col;
  card.textContent = INSIGHTS[burnout.classId];
  document.getElementById("latency").textContent = `Inference ran locally in ${ms} ms — no server round-trip.`;

  // stash for feedback
  state._result = { inp, burnout, gpa };
  show("results");
}

// ---------------------------------------------------------------------------
// Wire up DOM
// ---------------------------------------------------------------------------
function bindSlider(id, key) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "_val");
  el.value = state[key];
  out.textContent = state[key].toFixed(1);
  el.addEventListener("input", () => {
    state[key] = parseFloat(el.value);
    out.textContent = state[key].toFixed(1);
    updateRemaining();
  });
}
function updateRemaining() {
  const r = (24 - state.study_hours - state.sleep_hours - state.eca_hours - state.social_hours - state.physical_hours);
  const el = document.getElementById("remaining");
  el.textContent = `${r.toFixed(1)}h unallocated of 24h`;
  const warn = document.getElementById("over24");
  const next = document.getElementById("toStep2");
  if (r < 0) { warn.style.display = "block"; next.disabled = true; }
  else { warn.style.display = "none"; next.disabled = false; }
}

function buildRatingSelectors() {
  const fields = [
    ["exam_pressure_rating", "Exam pressure level"],
    ["family_expectation_rating", "Family expectation level"],
    ["financial_stress_rating", "Financial stress level"],
    ["social_support_rating", "Social support level"],
    ["anxiety_score_rating", "Anxiety level"],
    ["depression_score_rating", "Depression level"],
  ];
  const host = document.getElementById("ratings");
  host.innerHTML = fields.map(([key, label]) => `
    <div class="field">
      <label>${label}</label>
      <div class="seg" data-key="${key}">
        ${RATING_OPTIONS.map(o => `<button type="button" class="seg-btn${state[key] === o ? " on" : ""}" data-val="${o}">${o}</button>`).join("")}
      </div>
    </div>`).join("");
  host.querySelectorAll(".seg").forEach(seg => {
    seg.addEventListener("click", e => {
      const btn = e.target.closest(".seg-btn"); if (!btn) return;
      state[seg.dataset.key] = btn.dataset.val;
      seg.querySelectorAll(".seg-btn").forEach(b => b.classList.toggle("on", b === btn));
    });
  });
}

async function submitFeedback() {
  const rating = state._fbRating || 0;
  const comment = document.getElementById("fb-comment").value;
  const r = state._result;
  const status = document.getElementById("fb-status");
  if (!rating) { status.textContent = "Please pick a star rating."; status.style.color = COLORS.burnout; return; }
  status.textContent = "Sending…"; status.style.color = COLORS.muted;
  try {
    const res = await fetch("/api/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...r.inp,
        burnout_gauge_score: r.burnout.score, burnout_class: r.burnout.className,
        predicted_gpa: r.gpa.gpa, rating, comment,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    status.textContent = "Thank you for your feedback!"; status.style.color = COLORS.healthy;
  } catch (e) {
    status.textContent = "Could not submit feedback (offline or not deployed)."; status.style.color = COLORS.mild;
  }
}

function init() {
  ["study_hours", "sleep_hours", "eca_hours", "social_hours", "physical_hours"]
    .forEach(k => bindSlider(k, k));
  updateRemaining();

  document.getElementById("stress_select").value = state.stress_level_category;
  document.getElementById("stress_select").addEventListener("change", e => {
    state.stress_level_category = e.target.value;
  });
  buildRatingSelectors();

  document.getElementById("toStep2").addEventListener("click", () => show("step2"));
  document.getElementById("backTo1").addEventListener("click", () => show("step1"));
  document.getElementById("analyze").addEventListener("click", runResults);
  document.getElementById("restart").addEventListener("click", () => show("step1"));

  document.querySelectorAll("#fb-stars button").forEach((b, i) => {
    b.addEventListener("click", () => {
      state._fbRating = i + 1;
      document.querySelectorAll("#fb-stars button").forEach((x, j) => x.classList.toggle("on", j <= i));
    });
  });
  document.getElementById("fb-submit").addEventListener("click", submitFeedback);
}

document.addEventListener("DOMContentLoaded", init);
