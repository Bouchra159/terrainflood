"use strict";
// ============================================================
//  TerrainFlood-UQ — 10-Slide Presentation (16:9)
//  NODE_PATH="C:\Users\BOUCHRA\AppData\Roaming\npm\node_modules" node presentation.js
// ============================================================
const pptxgen = require("pptxgenjs");

const FIGS  = "C:/Users/BOUCHRA/Desktop/terrainflood/paper/figures";
const TEAL  = "4BAAA5";
const DTEAL = "2E7873";
const LTEAL = "D6EEEC";
const GOLD  = "C9A227";
const WHITE = "FFFFFF";
const DARK  = "1A2B3C";
const MID   = "4B5563";
const LGRAY = "EEF6F5";
const RED   = "B91C1C";
const GREEN = "166534";

const W = 13.33;
const H = 7.50;
const LM = 0.45;  // left margin

const pres = new pptxgen();
// Default is LAYOUT_16x9 (13.33 x 7.5)

const mkSh = () => ({ type:"outer", blur:5, offset:2, angle:135, color:"000000", opacity:0.10 });

// ── helpers ──────────────────────────────────────────────────────────────────
function hdr(sl, title) {
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:W, h:0.78, fill:{color:DTEAL}, line:{color:DTEAL} });
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:0.78, w:W, h:0.055, fill:{color:GOLD}, line:{color:GOLD} });
  sl.addText(title, { x:0.30, y:0, w:W-0.6, h:0.78,
    fontSize:22, fontFace:"Calibri", bold:true, color:WHITE, valign:"middle" });
  return 0.90; // body y-start
}

function bul(sl, x, y, w, h, items, fsz, col) {
  sl.addText(
    items.map((t, i) => ({ text:t, options:{ bullet:true, breakLine: i < items.length-1 } })),
    { x, y, w, h, fontSize:fsz||15, fontFace:"Calibri", color:col||DARK, paraSpaceAfter:5, valign:"top" }
  );
}

function card(sl, x, y, w, h, title, body, accent, light) {
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill:{color:light||WHITE}, line:{color:accent||TEAL, width:0.8}, shadow:mkSh() });
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w, h:0.44, fill:{color:accent||TEAL}, line:{color:accent||TEAL} });
  sl.addText(title, { x:x+0.10, y, w:w-0.20, h:0.44, fontSize:13, fontFace:"Calibri", bold:true, color:WHITE, valign:"middle" });
  sl.addText(body,  { x:x+0.10, y:y+0.50, w:w-0.20, h:h-0.58, fontSize:12, fontFace:"Calibri", color:DARK, valign:"top" });
}

function statBox(sl, x, y, w, h, val, lbl, bg, vc, lc) {
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill:{color:bg||DTEAL}, line:{color:bg||DTEAL}, shadow:mkSh() });
  sl.addText(val, { x, y:y+0.06, w, h:h*0.58, fontSize:28, fontFace:"Calibri", bold:true, color:vc||GOLD, align:"center" });
  sl.addText(lbl, { x, y:y+h*0.62, w, h:h*0.35, fontSize:11, fontFace:"Calibri", color:lc||LTEAL, align:"center" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: DTEAL };
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:W, h:0.14, fill:{color:GOLD}, line:{color:GOLD} });
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:H-0.14, w:W, h:0.14, fill:{color:GOLD}, line:{color:GOLD} });
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:0.14, w:0.20, h:H-0.28, fill:{color:TEAL}, line:{color:TEAL} });

  sl.addText("TerrainFlood-UQ", {
    x:0.5, y:1.10, w:W-1.0, h:1.10,
    fontSize:48, fontFace:"Calibri", bold:true, color:WHITE, align:"center"
  });
  sl.addText("Physics-Informed SAR Flood Mapping\nwith HAND-Guided Attention Gating and Uncertainty Quantification", {
    x:0.5, y:2.25, w:W-1.0, h:1.10,
    fontSize:18, fontFace:"Calibri", color:LTEAL, align:"center"
  });
  sl.addShape(pres.shapes.RECTANGLE, { x:3.8, y:3.50, w:5.73, h:0.055, fill:{color:GOLD}, line:{color:GOLD} });
  sl.addText("Bouchra Daddaoui", {
    x:0.5, y:3.65, w:W-1.0, h:0.52, fontSize:22, fontFace:"Calibri", bold:true, color:WHITE, align:"center"
  });
  sl.addText("Computation and Design  ·  Duke Kunshan University", {
    x:0.5, y:4.20, w:W-1.0, h:0.45, fontSize:17, fontFace:"Calibri", color:LTEAL, align:"center"
  });
  sl.addText("Supervisor: Prof. Dongmian Zou, Ph.D.  ·  Spring 2026", {
    x:0.5, y:4.68, w:W-1.0, h:0.40, fontSize:14, fontFace:"Calibri", color:GOLD, align:"center"
  });
  // slide number note
  sl.addText("Slide 1 / 10", { x:W-1.5, y:H-0.35, w:1.20, h:0.28, fontSize:10, fontFace:"Calibri", color:"88BBBB", align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem & Motivation
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "Problem & Motivation");

  const LW = 6.10;
  sl.addText("Why does this matter?", { x:LM, y:cy+0.08, w:LW, h:0.40, fontSize:17, fontFace:"Calibri", bold:true, color:DTEAL });
  bul(sl, LM, cy+0.53, LW, 3.30, [
    "Floods affect >250 million people/year and cause >$50 billion in annual damage — the world's most destructive natural hazard.",
    "Optical satellites cannot see through storm clouds. Sentinel-1 SAR provides all-weather, day-and-night flood imagery.",
    "Current flood models output binary flood/no-flood with no confidence — emergency responders cannot distinguish high-confidence from uncertain predictions.",
    "Water flows downhill: flood extent is physically constrained to low-lying terrain. Most models ignore this entirely.",
  ], 15);

  // Gap box
  sl.addShape(pres.shapes.RECTANGLE, { x:LM, y:cy+3.98, w:LW, h:1.30, fill:{color:LTEAL}, line:{color:TEAL, width:1.0}, shadow:mkSh() });
  sl.addText("Research goal: physics-informed SAR flood segmentation with full uncertainty quantification — WHERE floods are AND how confident we should be.", {
    x:LM+0.15, y:cy+4.06, w:LW-0.30, h:1.14,
    fontSize:14, fontFace:"Calibri", bold:true, color:DTEAL, valign:"middle"
  });

  // Study area figure
  const RX = LM + LW + 0.30;
  const RW = W - RX - LM;
  sl.addImage({ path:`${FIGS}/study_area.png`, x:RX, y:cy+0.08, w:RW, h:5.45, sizing:{type:"contain",w:RW,h:5.45} });
  sl.addText("Bolivia OOD test region — 15 chips, Amazonian floodplain, mean HAND = 1.15 m", {
    x:RX, y:cy+5.58, w:RW, h:0.38, fontSize:11, fontFace:"Calibri", italic:true, color:MID, align:"center"
  });
  sl.addText("2 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Dataset & Setup
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "Dataset & Experimental Setup");

  const LW = 5.80;
  sl.addText("Sen1Floods11 Dataset", { x:LM, y:cy+0.08, w:LW, h:0.40, fontSize:17, fontFace:"Calibri", bold:true, color:DTEAL });
  bul(sl, LM, cy+0.53, LW, 2.35, [
    "446 hand-labelled Sentinel-1 tiles (512×512 px, 10 m) — 6 global flood events.",
    "6-band tensor: VV/VH pre- & post-flood, VV−VH ratio, HAND (z-normalised). Encoder uses 4 SAR bands; HAND routes to attention gate.",
    "Bolivia (15 chips, 2.87M valid pixels, 12% flood fraction) = held-out OOD test set, never seen during training.",
  ], 15);

  sl.addText("Ablation Variants", { x:LM, y:cy+2.98, w:LW, h:0.40, fontSize:17, fontFace:"Calibri", bold:true, color:DTEAL });
  const vdata = [
    { tag:"A", desc:"SAR-only — no HAND (baseline)", hi:false },
    { tag:"B", desc:"HAND concatenated to encoder", hi:false },
    { tag:"C", desc:"HAND attention gate", hi:true },
    { tag:"D", desc:"HAND gate + MC Dropout (T=20)", hi:true },
    { tag:"D_full ★", desc:"D + extended 120-epoch training — best model", hi:true },
  ];
  vdata.forEach((v, i) => {
    const vy = cy + 3.45 + i * 0.55;
    sl.addShape(pres.shapes.RECTANGLE, { x:LM, y:vy, w:0.88, h:0.46, fill:{color:v.hi?TEAL:"94A3B8"}, line:{color:v.hi?TEAL:"94A3B8"} });
    sl.addText(v.tag, { x:LM, y:vy, w:0.88, h:0.46, fontSize:12, fontFace:"Calibri", bold:true, color:WHITE, align:"center", valign:"middle" });
    sl.addText(v.desc, { x:LM+0.95, y:vy+0.03, w:LW-0.98, h:0.40, fontSize:13, fontFace:"Calibri", color:v.hi?DTEAL:MID });
  });

  const RX = LM + LW + 0.30;
  const RW = W - RX - LM;
  sl.addImage({ path:`${FIGS}/dataset.jpeg`, x:RX, y:cy+0.08, w:RW, h:5.60, sizing:{type:"contain",w:RW,h:5.60} });
  sl.addText("SAR VV-band diversity across flood events (USA, Spain, Ghana, Bolivia…)", {
    x:RX, y:cy+5.72, w:RW, h:0.38, fontSize:11, fontFace:"Calibri", italic:true, color:MID, align:"center"
  });
  sl.addText("3 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Architecture
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "Model Architecture — Siamese ResNet-34 with HAND Attention Gate");

  const FW = W - 2*LM;
  sl.addImage({ path:`${FIGS}/arch.jpeg`, x:LM, y:cy+0.08, w:FW, h:4.05, sizing:{type:"contain",w:FW,h:4.05} });

  const BW = (FW - 0.30) / 3;
  [
    { t:"Siamese Encoder", b:"Dual ResNet-34 branches with shared weights. Pre- and post-flood SAR processed in parallel; change features fused at each decoder level." },
    { t:"HAND Attention Gate", b:"α = exp(−h / 50). Suppresses flood probability in elevated terrain; near-zero suppression in Bolivia's flat floodplain (mean HAND = 1.15 m)." },
    { t:"U-Net Decoder", b:"4-stage upsampling with skip connections. Full-resolution (10 m) flood probability map output." },
  ].forEach((b, i) => card(sl, LM + i*(BW+0.15), cy+4.30, BW, 1.88, b.t, b.b));

  sl.addText("4 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — HAND Gate + Training
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "HAND Gate Physics & Training");

  const LW = 5.50;
  sl.addImage({ path:`${FIGS}/hand_gate.jpeg`, x:LM, y:cy+0.08, w:LW, h:3.70, sizing:{type:"contain",w:LW,h:3.70} });
  sl.addText("Gate weight α = exp(−h/50): near 1 in flat floodplains, near 0 in highlands. Bolivia mean HAND = 1.15 m → α ≈ 1.00", {
    x:LM, y:cy+3.82, w:LW, h:0.45, fontSize:11, fontFace:"Calibri", italic:true, color:MID, align:"center"
  });

  const RX = LM + LW + 0.35;
  const RW = W - RX - LM;
  sl.addText("Training Setup", { x:RX, y:cy+0.08, w:RW, h:0.40, fontSize:17, fontFace:"Calibri", bold:true, color:DTEAL });
  bul(sl, RX, cy+0.52, RW, 2.60, [
    "Loss: Tversky (α=0.3, β=0.7) + Focal (γ=2) — addresses 88%/12% class imbalance.",
    "AdamW, lr=1×10⁻⁴, weight decay=1×10⁻², batch 8.",
    "A/B/C/D: 60 epochs, patience 15.  *_full: 120 epochs, patience 25.",
    "MC Dropout: T=20 passes, enable_dropout() after eval().",
    "TTA: D4 symmetry (8 augmentations). Cal: temperature scaling → T=0.100.",
  ], 14);

  sl.addImage({ path:`${FIGS}/training.jpeg`, x:RX, y:cy+3.30, w:RW, h:2.50, sizing:{type:"contain",w:RW,h:2.50} });
  sl.addText("Val IoU — D_full converges to 0.724 at epoch 96", {
    x:RX, y:cy+5.84, w:RW, h:0.35, fontSize:11, fontFace:"Calibri", italic:true, color:MID, align:"center"
  });
  sl.addText("5 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Ablation Results
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "Ablation Results — Bolivia OOD Test Set");

  // 4 stat boxes
  const stats = [
    { v:"0.724",  l:"D_full IoU\n(best model)"   },
    { v:"+30.3pp",l:"Gain over\nU-Net baseline"  },
    { v:"78.6%",  l:"ECE reduction\n(temp. scaling)" },
    { v:"7.84M",  l:"People at\nflood risk"      },
  ];
  const SW = (W - 2*LM - 0.30) / 4;
  stats.forEach((s, i) => statBox(sl, LM + i*(SW+0.10), cy+0.12, SW, 1.18, s.v, s.l));

  // Table
  const ho = { bold:true, fill:{color:DTEAL}, color:WHITE, fontSize:13 };
  sl.addTable([
    [ {text:"Variant",options:ho}, {text:"Description",options:ho}, {text:"IoU ↑",options:ho}, {text:"ΔIoU",options:ho}, {text:"ECE ↓",options:ho} ],
    ["A",      "SAR-only (baseline)",       "0.612", "—",      "—"],
    ["B",      "HAND concatenated",         "0.641", "+0.029", "—"],
    ["C",      "HAND attention gate",       "0.690", "+0.078", "—"],
    ["D",      "HAND gate + MC Dropout",    "0.690", "+0.078", "0.078"],
    [{text:"C_full",options:{color:TEAL,bold:true}}, "120 epochs",
     {text:"0.706",options:{color:TEAL}}, "+0.094", "—"],
    [{text:"D_full ★",options:{bold:true,color:GOLD}}, "120 ep + Dropout",
     {text:"0.724",options:{bold:true,color:DTEAL}}, {text:"+0.112",options:{bold:true,color:DTEAL}}, {text:"0.063",options:{color:DTEAL}}],
    ["Otsu",   "Classical threshold",       "0.582", "−0.030", "—"],
    ["U-Net",  "Vanilla encoder-decoder",   "0.421", "−0.191", "—"],
  ], {
    x:LM, y:cy+1.43, w:W-2*LM, h:4.55,
    colW:[1.35, 4.20, 1.50, 1.55, 1.40],
    border:{pt:0.5,color:"DAEAE9"},
    fontSize:13, fontFace:"Calibri", color:DARK,
    align:"center", autoPage:false, rowH:0.54,
  });
  sl.addText("6 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Visual Results
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "Visual Results — Bolivia OOD Test Chip");

  const FW = W - 2*LM;
  sl.addImage({ path:`${FIGS}/chip_analysis.png`, x:LM, y:cy+0.12, w:FW, h:5.28, sizing:{type:"contain",w:FW,h:5.28} });
  sl.addText(
    "Left → Right: SAR VV (pre & post-flood)  ·  Ground Truth  ·  D_full Prediction  ·  TTA Uncertainty Map (lighter = uncertain)\n" +
    "Uncertainty concentrates at flood/land boundaries — exactly where the model should be less confident.",
    { x:LM, y:cy+5.44, w:FW, h:0.65, fontSize:13, fontFace:"Calibri", italic:true, color:MID, align:"center" }
  );
  sl.addText("7 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Calibration & Uncertainty
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "Calibration & Uncertainty Quantification");

  const LW = 6.20;
  sl.addImage({ path:`${FIGS}/calibration.jpeg`, x:LM, y:cy+0.10, w:LW, h:4.00, sizing:{type:"contain",w:LW,h:4.00} });
  sl.addText("Reliability diagrams — before: ECE=0.363 (severely overconfident) / after temperature scaling: ECE=0.063", {
    x:LM, y:cy+4.14, w:LW, h:0.44, fontSize:11, fontFace:"Calibri", italic:true, color:MID, align:"center"
  });

  const RX = LM + LW + 0.30;
  const RW = W - RX - LM;

  sl.addText("Uncertainty Methods", { x:RX, y:cy+0.10, w:RW, h:0.40, fontSize:17, fontFace:"Calibri", bold:true, color:DTEAL });

  // TTA — works
  sl.addShape(pres.shapes.RECTANGLE, { x:RX, y:cy+0.60, w:RW, h:1.62, fill:{color:"ECFDF5"}, line:{color:GREEN, width:0.8} });
  sl.addText("✓  TTA — D4 Symmetry Group", { x:RX+0.12, y:cy+0.66, w:RW-0.20, h:0.40, fontSize:14, fontFace:"Calibri", bold:true, color:GREEN });
  sl.addText("σ² = 8.3×10⁻³  ·  r = +0.614\nPositive correlation with error — reliable spatial uncertainty proxy.", {
    x:RX+0.12, y:cy+1.08, w:RW-0.20, h:1.06, fontSize:12, fontFace:"Calibri", color:DARK
  });

  // MC Dropout — fails
  sl.addShape(pres.shapes.RECTANGLE, { x:RX, y:cy+2.38, w:RW, h:1.62, fill:{color:"FEF2F2"}, line:{color:RED, width:0.8} });
  sl.addText("✗  MC Dropout (T=20 passes)", { x:RX+0.12, y:cy+2.44, w:RW-0.20, h:0.40, fontSize:14, fontFace:"Calibri", bold:true, color:RED });
  sl.addText("σ² = 4.0×10⁻⁴  ·  r = −0.815 (inverted!)\nHAND gate collapses activation variance — FAILS in gated architectures.", {
    x:RX+0.12, y:cy+2.86, w:RW-0.20, h:1.06, fontSize:12, fontFace:"Calibri", color:DARK
  });

  // Temperature scaling
  sl.addShape(pres.shapes.RECTANGLE, { x:RX, y:cy+4.15, w:RW, h:1.30, fill:{color:LTEAL}, line:{color:TEAL, width:0.8} });
  sl.addText("🌡  Temperature Scaling: T = 0.100\nECE 0.363 → 0.063  (78.6% reduction, zero accuracy loss)", {
    x:RX+0.12, y:cy+4.24, w:RW-0.20, h:1.12, fontSize:13, fontFace:"Calibri", bold:false, color:DTEAL
  });

  sl.addText("8 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Conclusions & Future Work
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "Conclusions & Future Work");

  const LW = 7.20;
  sl.addText("Key Takeaways", { x:LM, y:cy+0.08, w:LW, h:0.40, fontSize:17, fontFace:"Calibri", bold:true, color:DTEAL });
  bul(sl, LM, cy+0.52, LW, 3.20, [
    "HAND attention gate is the dominant accuracy driver: +7.8pp IoU over SAR-only, encoding terrain constraints directly into the model.",
    "TTA (D4) is a reliable uncertainty proxy (r = +0.614). MC Dropout FAILS in gated models — activation suppression inverts the signal (r = −0.815).",
    "Temperature scaling (T=0.100) reduces ECE by 78.6% with zero accuracy cost — efficient post-hoc calibration for deployment.",
    "D_full achieves IoU = 0.724 on OOD Bolivia — +30.3pp over U-Net (0.421), +14.2pp over Otsu (0.582).",
    "Uncertainty maps flag 1.21M people (15.4%) as high-uncertainty risk — directly supporting humanitarian triage.",
  ], 15);

  sl.addText("Future Work", { x:LM, y:cy+3.82, w:LW, h:0.40, fontSize:17, fontFace:"Calibri", bold:true, color:DTEAL });
  bul(sl, LM, cy+4.26, LW, 1.80, [
    "Multi-hazard extension: coastal storm surge, landslide susceptibility mapping using same HAND-gated backbone.",
    "Real-time integration with Copernicus Emergency Management Service (CEMS) for automated flood alerting.",
    "Ensemble UQ: combine TTA + deep ensembles for higher-confidence uncertainty in life-critical decisions.",
  ], 15);

  const RX = LM + LW + 0.30;
  const RW = W - RX - LM;
  sl.addImage({ path:`${FIGS}/exposure.png`, x:RX, y:cy+0.08, w:RW, h:5.40, sizing:{type:"contain",w:RW,h:5.40} });
  sl.addText("7.84M at risk · 1.21M (15.4%) uncertain\nBolivia OOD — TTA τ=0.01", {
    x:RX, y:cy+5.54, w:RW, h:0.50, fontSize:11, fontFace:"Calibri", italic:true, color:MID, align:"center"
  });
  sl.addText("9 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:MID, align:"right" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — References + Thank You
// ════════════════════════════════════════════════════════════════════════════
{
  const sl = pres.addSlide();
  sl.background = { color: LGRAY };
  const cy = hdr(sl, "References");

  sl.addText([
    '[1] D. Bonafilia et al., "Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms," CVPRW, 2020.',
    '[2] A. D. Nobre et al., "Height Above the Nearest Drainage — a hydrologically relevant terrain index," J. Hydrol., 2011.',
    '[3] Y. Gal & Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," ICML, 2016.',
    '[4] C. Guo et al., "On calibration of modern neural networks," ICML, 2017.',
    '[5] O. Ronneberger et al., "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015.',
    '[6] K. He et al., "Deep residual learning for image recognition," CVPR, 2016.',
    '[7] R. T. Lin et al., "Focal loss for dense object detection," ICCV, 2017.',
    '[8] D. P. Kingma & J. Ba, "Adam: A method for stochastic optimization," ICLR, 2015.',
  ].map((r, i, a) => ({ text:r, options:{ breakLine: i < a.length-1 } })), {
    x:LM, y:cy+0.25, w:W-2*LM, h:5.00,
    fontSize:14, fontFace:"Calibri", color:DARK, paraSpaceAfter:7, valign:"top"
  });

  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:H-0.80, w:W, h:0.80, fill:{color:DTEAL}, line:{color:DTEAL} });
  sl.addShape(pres.shapes.RECTANGLE, { x:0, y:H-0.80, w:W, h:0.07, fill:{color:GOLD}, line:{color:GOLD} });
  sl.addText("Thank you! Questions welcome.  ·  TerrainFlood-UQ  ·  Duke Kunshan University 2026", {
    x:0.4, y:H-0.80, w:W-0.8, h:0.80,
    fontSize:15, fontFace:"Calibri", color:LTEAL, align:"center", valign:"middle"
  });
  sl.addText("10 / 10", { x:W-1.0, y:H-0.32, w:0.75, h:0.25, fontSize:10, fontFace:"Calibri", color:LTEAL, align:"right" });
}

// ─── SAVE ─────────────────────────────────────────────────────────────────────
(async () => {
  const out = "C:/Users/BOUCHRA/Desktop/terrainflood/paper/presentation.pptx";
  await pres.writeFile({ fileName: out });
  console.log("✓ presentation.pptx saved →", out);
})();
