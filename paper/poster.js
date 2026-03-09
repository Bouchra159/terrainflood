"use strict";
// ============================================================
//  TerrainFlood-UQ — Research Poster (A0 portrait)
//  Generate: NODE_PATH="C:\Users\BOUCHRA\AppData\Roaming\npm\node_modules" node poster.js
// ============================================================
const pptxgen = require("pptxgenjs");

// ─── FIGURE PATHS ────────────────────────────────────────────────────────────
const FIGS = "C:/Users/BOUCHRA/Desktop/terrainflood/paper/figures";

// ─── COLORS (no # prefix) ────────────────────────────────────────────────────
const TEAL   = "4BAAA5";
const DTEAL  = "2E7873";
const LTEAL  = "D6EEEC";
const GOLD   = "C9A227";
const WHITE  = "FFFFFF";
const DARK   = "1A2B3C";
const MID    = "4B5563";
const LGRAY  = "EEF6F5";
const LGRAY2 = "F4F9F9";

// ─── LAYOUT ──────────────────────────────────────────────────────────────────
const W = 33.11;   // A0 width  (inches)
const H = 46.81;   // A0 height (inches)
const MAR   = 0.40;   // side margin
const GAP   = 0.28;   // column gap
const CPAD  = 0.25;   // padding inside section
const COL_W = (W - 2 * MAR - 2 * GAP) / 3;   // ≈ 10.58"
const C1X   = MAR;
const C2X   = MAR + COL_W + GAP;
const C3X   = MAR + 2 * (COL_W + GAP);
const FIG_W = COL_W - 2 * CPAD;               // figure width inside section

const HDR_H    = 3.10;
const CSTART   = HDR_H + 0.20;   // y where columns begin
const FOOTER_Y = H - 0.68;
const STITLE_H = 0.66;            // section title bar height

// ─── SHADOW FACTORY (never reuse) ────────────────────────────────────────────
const mkShadow = () => ({ type:"outer", blur:7, offset:2, angle:135, color:"000000", opacity:0.09 });

// ─── PRESENTATION SETUP ──────────────────────────────────────────────────────
const pres = new pptxgen();
pres.defineLayout({ name:"A0P", width:W, height:H });
pres.layout = "A0P";

const slide = pres.addSlide();
slide.background = { color: LGRAY };

// ─── HELPERS ─────────────────────────────────────────────────────────────────

/**
 * Draw a section box with a teal title bar + gold accent.
 * Returns y-coordinate where body content should start.
 */
function section(x, y, w, h, title) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: WHITE }, line: { color: "DAEAE9", width: 0.5 },
    shadow: mkShadow(),
  });
  // title bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: STITLE_H,
    fill: { color: TEAL }, line: { color: TEAL },
  });
  // gold left accent
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.14, h: STITLE_H,
    fill: { color: GOLD }, line: { color: GOLD },
  });
  slide.addText(title.toUpperCase(), {
    x: x + 0.22, y, w: w - 0.30, h: STITLE_H,
    fontSize: 20, fontFace:"Calibri", bold:true,
    color:WHITE, align:"left", valign:"middle",
    margin:0, charSpacing:1.3,
  });
  return y + STITLE_H + 0.20;   // body starts here
}

/**  Small bold teal sub-heading inside a section. */
function subhead(x, y, w, text) {
  slide.addText(text, {
    x, y, w, h: 0.44,
    fontSize:18, fontFace:"Calibri", bold:true,
    color: DTEAL, align:"left",
  });
  return y + 0.46;
}

/**  Multi-bullet text block. Returns bottom y of the block. */
function bullets(x, y, w, h, items, fsz) {
  const runs = items.map((t, i) => ({
    text: t,
    options: { bullet:true, breakLine: i < items.length - 1 },
  }));
  slide.addText(runs, {
    x, y, w, h,
    fontSize: fsz || 16, fontFace:"Calibri", color:DARK,
    paraSpaceAfter:4, valign:"top",
  });
  return y + h;
}

/**  Italic caption below a figure. */
function cap(x, y, w, text) {
  slide.addText(text, {
    x, y, w, h:0.38,
    fontSize:13, fontFace:"Calibri", italic:true,
    color:MID, align:"center",
  });
  return y + 0.40;
}

/**  Image with sizing: contain. Returns bottom y. */
function fig(path, x, y, w, h) {
  slide.addImage({
    path, x, y, w, h,
    sizing: { type:"contain", w, h },
  });
  return y + h;
}

// ════════════════════════════════════════════════════════════════════════════
//  HEADER
// ════════════════════════════════════════════════════════════════════════════
slide.addShape(pres.shapes.RECTANGLE, {
  x:0, y:0, w:W, h:HDR_H, fill:{ color:DTEAL }, line:{ color:DTEAL },
});
// gold stripes
slide.addShape(pres.shapes.RECTANGLE, {
  x:0, y:0, w:W, h:0.18, fill:{ color:GOLD }, line:{ color:GOLD },
});
slide.addShape(pres.shapes.RECTANGLE, {
  x:0, y:HDR_H - 0.18, w:W, h:0.18, fill:{ color:GOLD }, line:{ color:GOLD },
});

slide.addText(
  "TerrainFlood-UQ: Physics-Informed SAR Flood Mapping\nwith HAND-Guided Attention Gating and Uncertainty Quantification",
  {
    x:0.7, y:0.22, w:W - 1.4, h:1.70,
    fontSize:46, fontFace:"Calibri", bold:true,
    color:WHITE, align:"center", valign:"middle",
  }
);
slide.addText(
  "Bouchra Daddaoui  ·  Computation and Design  ·  Duke Kunshan University",
  {
    x:0.7, y:1.96, w:W - 1.4, h:0.56,
    fontSize:25, fontFace:"Calibri", color:LTEAL, align:"center",
  }
);
slide.addText(
  "Supervisor: Prof. Dongmian Zou, Ph.D.  ·  Class of 2026",
  {
    x:0.7, y:2.55, w:W - 1.4, h:0.44,
    fontSize:21, fontFace:"Calibri", color:GOLD, align:"center",
  }
);

// ════════════════════════════════════════════════════════════════════════════
//  COLUMN 1
// ════════════════════════════════════════════════════════════════════════════

let y1 = CSTART;

// ── Section 1: Introduction & Motivation ────────────────────────────────────
const S1H = 13.20;
let cy = section(C1X, y1, COL_W, S1H, "Introduction & Motivation");

bullets(C1X+CPAD, cy, FIG_W, 2.55, [
  "Floods are the most destructive natural hazard: >250 million people affected annually, >$50B in economic losses.",
  "Optical satellites fail during active storms due to cloud cover. Sentinel-1 SAR provides all-weather, day/night flood mapping.",
  "Binary flood/no-flood maps offer no confidence measure — hindering emergency triage and humanitarian resource allocation.",
  "This work: Siamese ResNet-34 encoder with physics-guided HAND attention gate, combined with TTA, MC Dropout, and temperature scaling for full uncertainty quantification.",
], 15);

let fy = cy + 2.70;
fy = fig(`${FIGS}/study_area.png`, C1X+CPAD, fy, FIG_W, 9.70);
cap(C1X+CPAD, fy, FIG_W,
  "Fig. 1 — Bolivia OOD test region (Amazonian floodplain). Mean HAND = 1.15 m, 15 chips, 12% flood fraction.");

y1 += S1H + 0.20;

// ── Section 2: Materials & Methods ──────────────────────────────────────────
const S2H = FOOTER_Y - y1;
cy = section(C1X, y1, COL_W, S2H, "Materials & Methods");

// Dataset
cy = subhead(C1X+CPAD, cy, FIG_W, "Dataset — Sen1Floods11");
bullets(C1X+CPAD, cy, FIG_W, 2.20, [
  "446 hand-labelled Sentinel-1 tiles (512 × 512 px, 10 m GSD) across 6 flood events (USA, Spain, Paraguay, Ghana, Sri Lanka, Bolivia).",
  "6-band tensor per tile: VV/VH pre-flood, VV/VH post-flood, VV−VH ratio (derived), HAND (z-normalised). Encoder uses 4 SAR bands (Ch. 0–3); HAND (Ch. 5) routes to attention gate after denormalisation.",
  "Bolivia (15 chips, 2.87M valid pixels) is a held-out OOD test set — never included in training or validation.",
], 15);
fy = cy + 2.30;
fy = fig(`${FIGS}/dataset.jpeg`, C1X+CPAD, fy, FIG_W, 5.00);
cap(C1X+CPAD, fy, FIG_W, "Fig. 2 — SAR VV-band diversity across flood events and geographies.");
cy = fy + 0.42 + 0.15;

// Architecture
cy = subhead(C1X+CPAD, cy, FIG_W, "Architecture — Siamese ResNet-34 + HAND Gate");
bullets(C1X+CPAD, cy, FIG_W, 2.00, [
  "Dual-branch weight-shared ResNet-34 encoder: pre-flood and post-flood VV/VH processed in parallel, features fused at each decoder level.",
  "HAND Attention Gate: α = exp(−h / 50), h in metres — physically suppresses flood probability in elevated terrain where flooding is unlikely.",
  "Decoder with 4-stage upsampling and skip connections reconstructs prediction at full 10 m resolution.",
], 15);
fy = cy + 2.10;
fy = fig(`${FIGS}/arch.jpeg`, C1X+CPAD, fy, FIG_W, 5.10);
cap(C1X+CPAD, fy, FIG_W, "Fig. 3 — Siamese ResNet-34 with HAND attention gate and U-Net style decoder.");
cy = fy + 0.42 + 0.15;

// HAND Gate
cy = subhead(C1X+CPAD, cy, FIG_W, "HAND Gate Physics");
fy = cy + 0.05;
fy = fig(`${FIGS}/hand_gate.jpeg`, C1X+CPAD, fy, FIG_W, 4.80);
cap(C1X+CPAD, fy, FIG_W,
  "Fig. 7 — Gate weight α = exp(−h/50): near-zero suppression at Bolivia's mean HAND of 1.15 m; full suppression > 150 m.");
cy = fy + 0.42 + 0.15;

// Training
cy = subhead(C1X+CPAD, cy, FIG_W, "Training & Uncertainty Methods");
bullets(C1X+CPAD, cy, FIG_W, S2H - (cy - (y1 + STITLE_H + 0.20)) - 0.40, [
  "Loss: Tversky (α=0.3, β=0.7) + Focal (γ=2) — addresses severe 88%/12% class imbalance.",
  "Optimiser: AdamW, lr=1×10⁻⁴, weight decay=1×10⁻². Batch size 8.",
  "Variants A/B/C/D: 60 epochs, patience 15. C_full / D_full: 120 epochs, patience 25.",
  "MC Dropout: T=20 forward passes with model.enable_dropout() after model.eval(). Variance σ² is the uncertainty proxy.",
  "TTA: D4 symmetry group — 8 augmentations (H-flip, V-flip × 4 rotations). Mean prediction + variance across passes.",
  "Temperature scaling: post-hoc calibration on validation set → optimal T = 0.100.",
], 15);

// ════════════════════════════════════════════════════════════════════════════
//  COLUMN 2
// ════════════════════════════════════════════════════════════════════════════

let y2 = CSTART;

// ── Section 3: Results ───────────────────────────────────────────────────────
const S3H = 15.80;
cy = section(C2X, y2, COL_W, S3H, "Results");

// Key metrics highlight box
slide.addShape(pres.shapes.RECTANGLE, {
  x: C2X+CPAD, y: cy, w: FIG_W, h: 1.18,
  fill:{ color: LTEAL }, line:{ color: TEAL, width:0.8 },
  shadow: mkShadow(),
});
slide.addText(
  "Best model — D_full:   IoU = 0.724  ·  F₁ = 0.840  ·  ECE = 0.063",
  {
    x: C2X+CPAD, y: cy, w: FIG_W, h: 1.18,
    fontSize:19, fontFace:"Calibri", bold:true,
    color: DTEAL, align:"center", valign:"middle",
  }
);
cy += 1.28;

fy = fig(`${FIGS}/chip_analysis.png`, C2X+CPAD, cy, FIG_W, 9.30);
cap(C2X+CPAD, fy, FIG_W,
  "Fig. 4 — Bolivia test chip: SAR VV, ground truth, D_full prediction, TTA uncertainty (light = uncertain).");
cy = fy + 0.42;

bullets(C2X+CPAD, cy, FIG_W, S3H - (cy - y2) - 0.15, [
  "D_full (HAND gate + MC Dropout, 120 ep): IoU = 0.724 — best overall performance.",
  "Otsu classical threshold: IoU = 0.582. Plain U-Net: IoU = 0.421 — HAND gate adds +11.2pp vs. U-Net.",
  "TTA uncertainty aligns with prediction error: r = +0.614 (useful). MC Dropout: r = −0.815 (inverted — gate suppression failure).",
], 15);

y2 += S3H + 0.20;

// ── Section 4: Ablation Study ─────────────────────────────────────────────────
const S4H = 11.00;
cy = section(C2X, y2, COL_W, S4H, "Ablation Study — Bolivia OOD Test Set");

const hdrOpts = { bold:true, fill:{ color:DTEAL }, color:WHITE, fontSize:15 };
const tRows = [
  [
    { text:"Variant",     options: hdrOpts },
    { text:"Configuration",   options: hdrOpts },
    { text:"IoU ↑",       options: hdrOpts },
    { text:"ΔIoU vs A",   options: hdrOpts },
    { text:"ECE ↓",       options: hdrOpts },
  ],
  ["A",       "SAR-only (no HAND)",           "0.612",  "—",       "—"],
  ["B",       "HAND concat to encoder",        "0.641",  "+0.029",  "—"],
  ["C",       "HAND attention gate",           "0.690",  "+0.078",  "—"],
  ["D",       "HAND gate + MC Dropout",        "0.690",  "+0.078",  "0.078"],
  [
    { text:"C_full", options:{ color:TEAL, bold:true } },
    "Gate — 120 ep, patience 25",
    { text:"0.706", options:{ color:TEAL } },
    { text:"+0.094", options:{ color:TEAL } },
    "—",
  ],
  [
    { text:"D_full ★", options:{ bold:true, color:GOLD } },
    "Gate + Dropout — 120 ep",
    { text:"0.724", options:{ bold:true, color:DTEAL } },
    { text:"+0.112", options:{ bold:true, color:DTEAL } },
    { text:"0.063", options:{ color:DTEAL } },
  ],
  ["Otsu",    "Classical threshold (baseline)", "0.582",  "−0.030",  "—"],
  ["U-Net",   "Vanilla encoder-decoder",        "0.421",  "−0.191",  "—"],
];

slide.addTable(tRows, {
  x: C2X+CPAD, y: cy, w: FIG_W, h: 7.60,
  colW: [1.45, 3.90, 1.45, 1.60, 1.60],
  border:{ pt:0.5, color:"DAEAE9" },
  fontSize:14, fontFace:"Calibri", color:DARK,
  align:"center", autoPage:false, rowH:0.82,
});

slide.addText(
  "HAND gate alone: +7.8pp  ·  Extended training: +3.4pp  ·  Total gain over U-Net: +30.3pp",
  {
    x: C2X+CPAD, y: cy+7.72, w: FIG_W, h: 0.55,
    fontSize:13, fontFace:"Calibri", italic:true,
    color:MID, align:"center",
  }
);

// Uncertainty benchmarks mini-table
slide.addText("Uncertainty Quality:", {
  x: C2X+CPAD, y: cy+8.50, w: FIG_W, h: 0.40,
  fontSize:15, fontFace:"Calibri", bold:true, color:DTEAL,
});
slide.addTable([
  [
    { text:"Method", options:{ bold:true, fill:{ color:DTEAL }, color:WHITE } },
    { text:"σ² (mean)", options:{ bold:true, fill:{ color:DTEAL }, color:WHITE } },
    { text:"r (error corr.)", options:{ bold:true, fill:{ color:DTEAL }, color:WHITE } },
    { text:"Status", options:{ bold:true, fill:{ color:DTEAL }, color:WHITE } },
  ],
  ["TTA (D4)",    "8.3 × 10⁻³",  "+0.614",  { text:"✓ Useful", options:{ color:"2E7873" } }],
  ["MC Dropout",  "4.0 × 10⁻⁴",  "−0.815",  { text:"✗ Inverted", options:{ color:"B91C1C" } }],
], {
  x: C2X+CPAD, y: cy+9.00, w: FIG_W, h: 1.60,
  colW: [2.50, 2.20, 2.20, 2.10],
  border:{ pt:0.5, color:"DAEAE9" },
  fontSize:14, fontFace:"Calibri", color:DARK,
  align:"center", autoPage:false, rowH:0.50,
});

y2 += S4H + 0.20;

// ── Section 5: Calibration ───────────────────────────────────────────────────
const S5H = FOOTER_Y - y2;
cy = section(C2X, y2, COL_W, S5H, "Calibration & Threshold Optimisation");

fy = fig(`${FIGS}/calibration.jpeg`, C2X+CPAD, cy, FIG_W, 9.00);
cap(C2X+CPAD, fy, FIG_W,
  "Fig. 5 — Reliability diagrams for D_full: before (ECE=0.363, overconfident) and after (ECE=0.063) temperature scaling.");
cy = fy + 0.42 + 0.15;

fy = fig(`${FIGS}/threshold.png`, C2X+CPAD, cy, FIG_W, 4.00);
cap(C2X+CPAD, fy, FIG_W,
  "Fig. 6 — IoU vs. decision threshold τ. Optimal threshold τ = 0.50 across all variants on Bolivia OOD test.");
cy = fy + 0.42;

bullets(C2X+CPAD, cy, FIG_W, S5H - (cy - y2) - 0.25, [
  "Temperature scaling (T = 0.100): ECE 0.363 → 0.063 (78.6% reduction) with zero accuracy loss — parameter-free post-hoc recalibration.",
  "TTA adds epistemic uncertainty orthogonal to calibration; the two methods are complementary.",
], 15);

// ════════════════════════════════════════════════════════════════════════════
//  COLUMN 3
// ════════════════════════════════════════════════════════════════════════════

let y3 = CSTART;

// ── Section 6: Training Dynamics ─────────────────────────────────────────────
const S6H = 10.00;
cy = section(C3X, y3, COL_W, S6H, "Training Dynamics — All Variants");

fy = fig(`${FIGS}/training.jpeg`, C3X+CPAD, cy, FIG_W, 8.85);
cap(C3X+CPAD, fy, FIG_W,
  "Fig. 8 — Val IoU over epochs: D_full converges to 0.724 at epoch 96; Variant A plateaus at 0.612.");

y3 += S6H + 0.20;

// ── Section 7: Population Exposure ───────────────────────────────────────────
const S7H = 9.80;
cy = section(C3X, y3, COL_W, S7H, "Population Exposure (WorldPop Post-Processing)");

fy = fig(`${FIGS}/exposure.png`, C3X+CPAD, cy, FIG_W, 8.30);
cap(C3X+CPAD, fy, FIG_W,
  "Fig. 9 — D_full TTA exposure map: 7.84M at risk in Bolivia OOD region; 1.21M (15.4%) flagged uncertain under τ = 0.01.");

y3 += S7H + 0.20;

// ── Section 8: Discussion ─────────────────────────────────────────────────────
const S8H = 10.00;
cy = section(C3X, y3, COL_W, S8H, "Discussion");

// Key numbers highlight row
const KN_W = (FIG_W - 0.20) / 3;
const KNY = cy;
const knBoxes = [
  { val:"0.724", lbl:"Best IoU\n(D_full)" },
  { val:"78.6%", lbl:"ECE reduced\n(temp. scaling)" },
  { val:"7.84M", lbl:"People at risk\n(Bolivia OOD)" },
];
knBoxes.forEach((k, i) => {
  const kx = C3X + CPAD + i * (KN_W + 0.10);
  slide.addShape(pres.shapes.RECTANGLE, {
    x:kx, y:KNY, w:KN_W, h:1.30,
    fill:{ color:LTEAL }, line:{ color:TEAL, width:0.5 },
  });
  slide.addText(k.val, { x:kx, y:KNY+0.02, w:KN_W, h:0.72,
    fontSize:28, fontFace:"Calibri", bold:true, color:DTEAL, align:"center" });
  slide.addText(k.lbl, { x:kx, y:KNY+0.74, w:KN_W, h:0.52,
    fontSize:12, fontFace:"Calibri", color:MID, align:"center" });
});
cy = KNY + 1.40;

bullets(C3X+CPAD, cy, FIG_W, 4.60, [
  "I.  HAND gate drives the dominant gain (+7.8pp IoU vs. SAR-only): topographic priors encode physically meaningful flood boundary constraints. Bolivia's flat Amazonian floodplain (mean HAND = 1.15 m) keeps gate weight near 1, allowing full model capacity where terrain is hydrologically relevant.",
  "II.  TTA captures geometric and boundary uncertainty (r = +0.614): test-time augmentation with the D4 symmetry group exposes spatial inconsistencies at flood/no-flood boundaries, directly indicating model confidence.",
  "III.  MC Dropout fails in gated architectures (r = −0.815 inverted): the HAND gate suppresses activation variance uniformly, collapsing dropout uncertainty — a critical finding for operational deployment in physics-gated models.",
  "IV.  Temperature scaling (T = 0.100) reduces ECE by 78.6% without retraining. Combined with TTA, it provides both calibrated marginal probabilities and spatially resolved uncertainty maps.",
  "V.  15.4% of at-risk population (1.21M of 7.84M) flagged as uncertain under TTA — demonstrating actionable uncertainty maps for humanitarian triage and flood response prioritisation.",
], 15);
cy += 4.70;

cy = subhead(C3X+CPAD, cy, FIG_W, "Limitations & Scope");
bullets(C3X+CPAD, cy, FIG_W, S8H - (cy - y3) - 0.25, [
  "Dataset limited to Sen1Floods11 events; performance on other flood types (urban, coastal, flash floods) not yet validated.",
  "HAND gate assumes topographic data availability — areas without MERIT Hydro coverage require the SAR-only (Variant A) fallback.",
], 15);

y3 += S8H + 0.20;

// ── Section 9: Conclusions ────────────────────────────────────────────────────
const S9H = 7.00;
cy = section(C3X, y3, COL_W, S9H, "Conclusions");

bullets(C3X+CPAD, cy, FIG_W, 3.30, [
  "Physics-guided HAND attention gate is the single largest accuracy contributor (+7.8pp IoU) — confirming that terrain priors encode physically meaningful flood boundary constraints in SAR deep learning.",
  "TTA with D4 symmetry is a reliable, calibration-compatible uncertainty proxy (r = +0.614); MC Dropout must be validated in gated models — activation suppression can invert the uncertainty signal.",
  "Temperature scaling achieves 78.6% ECE reduction without retraining — an efficient post-hoc calibration path for operational deployment.",
  "D_full achieves IoU = 0.724 on out-of-distribution Bolivian flood imagery — a +30.3pp gain over a plain U-Net baseline (0.421).",
  "Population exposure maps identify 1.21M uncertain-risk individuals (15.4%) — demonstrating direct humanitarian utility of uncertainty-aware flood predictions.",
], 15);
cy += 3.40;

cy = subhead(C3X+CPAD, cy, FIG_W, "Future Work");
bullets(C3X+CPAD, cy, FIG_W, S9H - (cy - y3) - 0.20, [
  "Multi-hazard extension: apply HAND-gated architecture to coastal storm surge, landslide susceptibility.",
  "Continuous monitoring: integrate with Copernicus Emergency Management Service for real-time flood alerting.",
  "Ensemble UQ: combine TTA with deep ensembles for more robust uncertainty estimates in high-stakes decisions.",
], 15);

y3 += S9H + 0.20;

// ── Section 10: Selected References ──────────────────────────────────────────
const S10H = FOOTER_Y - y3;
cy = section(C3X, y3, COL_W, S10H, "Selected References");

const refs = [
  "[1] D. Bonafilia et al., \"Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms,\" CVPRW, 2020.",
  "[2] A. D. Nobre et al., \"Height Above the Nearest Drainage — a hydrologically relevant terrain index,\" J. Hydrol., 2011.",
  "[3] Y. Gal & Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,\" ICML, 2016.",
  "[4] C. Guo et al., \"On calibration of modern neural networks,\" ICML, 2017.",
  "[5] O. Ronneberger et al., \"U-Net: Convolutional networks for biomedical image segmentation,\" MICCAI, 2015.",
  "[6] K. He et al., \"Deep residual learning for image recognition,\" CVPR, 2016.",
];

slide.addText(
  refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })),
  {
    x: C3X+CPAD, y: cy, w: FIG_W, h: S10H - STITLE_H - 0.40,
    fontSize:13, fontFace:"Calibri", color:DARK,
    paraSpaceAfter:4, valign:"top",
  }
);

// ════════════════════════════════════════════════════════════════════════════
//  FOOTER
// ════════════════════════════════════════════════════════════════════════════
slide.addShape(pres.shapes.RECTANGLE, {
  x:0, y:FOOTER_Y, w:W, h:H - FOOTER_Y,
  fill:{ color:DTEAL }, line:{ color:DTEAL },
});
slide.addShape(pres.shapes.RECTANGLE, {
  x:0, y:FOOTER_Y, w:W, h:0.09,
  fill:{ color:GOLD }, line:{ color:GOLD },
});
slide.addText(
  "Duke Kunshan University  ·  Computation and Design  ·  Signature Work 2026  ·  TerrainFlood-UQ",
  {
    x:0.5, y:FOOTER_Y + 0.09, w:W - 1.0, h:H - FOOTER_Y - 0.09,
    fontSize:16, fontFace:"Calibri", color:LTEAL,
    align:"center", valign:"middle",
  }
);

// ─── SAVE ─────────────────────────────────────────────────────────────────────
(async () => {
  const out = "C:/Users/BOUCHRA/Desktop/terrainflood/paper/poster.pptx";
  await pres.writeFile({ fileName: out });
  console.log("✓ poster.pptx saved to", out);
})();
