/**
 * ============================================================================
 * TerrainFlood-UQ — Google Earth Engine Flood Mapping Script
 * ============================================================================
 * Author : Bouchra Daddaoui | Duke Kunshan University | Signature Work 2026
 * Mentor : Prof. Dongmian Zou, Ph.D.
 *
 * PURPOSE:
 *   Reproduce the Bolivia 2018 Amazon flood study area visualisation using
 *   Sentinel-1 SAR, Landsat 8/9, SRTM HAND, and JRC water layers.
 *   Figures generated here support the paper:
 *     "Flood Inundation Mapping From Sentinel-1 SAR Using HAND-Guided
 *      Gating and Uncertainty Quantification"
 *
 * HOW TO RUN:
 *   1. Open https://code.earthengine.google.com/
 *   2. Paste this entire script into a new script
 *   3. Click Run
 *   4. Use the Layers panel to toggle each layer
 *   5. Use the Export tasks (Tasks tab) to download GeoTIFFs
 *
 * DATA SOURCES:
 *   - Sentinel-1 GRD  : COPERNICUS/S1_GRD
 *   - Landsat 8 TOA   : LANDSAT/LC08/C02/T1_TOA
 *   - SRTM DEM        : USGS/SRTMGL1_003
 *   - HAND            : MERIT/Hydro/v1_0_1  (MERIT Hydro, Yamazaki 2019)
 *   - JRC Water       : JRC/GSW1_4/GlobalSurfaceWater
 *   - WorldPop        : WorldPop/GP/100m/pop
 * ============================================================================
 */

// ── 0.  Study Area ────────────────────────────────────────────────────────────
// Beni Department, Bolivia — 2018 Amazon Flood
var beniROI = ee.Geometry.Rectangle([-67.8, -16.5, -62.8, -10.2]);
var boliviaPoint = ee.Geometry.Point([-65.5, -13.5]);  // near Trinidad

Map.centerObject(boliviaPoint, 7);
Map.setOptions('SATELLITE');

// ── 1.  Sentinel-1 SAR — Pre-Flood (reference period) ────────────────────────
// Pre-flood: Dry season Jul–Nov 2017
var s1Pre = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(beniROI)
  .filterDate('2017-07-01', '2017-11-30')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH'])
  .mean()
  .clip(beniROI);

// ── 2.  Sentinel-1 SAR — Post-Flood (Bolivia flood Jan–Mar 2018) ─────────────
var s1Post = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(beniROI)
  .filterDate('2018-01-01', '2018-03-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH'])
  .mean()
  .clip(beniROI);

// ── 3.  SAR Change Detection ──────────────────────────────────────────────────
// Change = post - pre in dB (negative change → new water)
var vvChange = s1Post.select('VV').subtract(s1Pre.select('VV'));
var vhChange = s1Post.select('VH').subtract(s1Pre.select('VH'));

// Otsu-style thresholding on VV change (< -3 dB → flooded)
var floodOtsu = vvChange.lt(-3.0).and(s1Post.select('VV').lt(-14.0));

// ── 4.  HAND — Height Above Nearest Drainage (MERIT Hydro) ───────────────────
// This is the core physical prior in TerrainFlood-UQ's attention gate
// α = exp(-HAND / 50)  → attenuates flood predictions at high elevation
var merit = ee.Image('MERIT/Hydro/v1_0_1');
var hand  = merit.select('hnd').clip(beniROI);

// Compute attention gate α = exp(-h/50) — same formula as in 03_model.py
var handGate = hand.multiply(-1.0/50.0).exp().rename('gate_alpha');

// ── 5.  Landsat 8 — True & False Colour Composites ───────────────────────────
// Use cloud-masked median composite around flood period
function maskL8Clouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0)
                    .and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(cloudMask)
              .divide(10000)
              .copyProperties(image, ['system:time_start']);
}

var l8Flood = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
  .filterBounds(beniROI)
  .filterDate('2018-01-01', '2018-04-30')
  .filter(ee.Filter.lt('CLOUD_COVER', 30))
  .map(function(img) {
    var qa = img.select('QA_PIXEL');
    return img.updateMask(qa.bitwiseAnd(1 << 3).eq(0));
  })
  .median()
  .clip(beniROI);

var l8Pre = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
  .filterBounds(beniROI)
  .filterDate('2017-07-01', '2017-11-30')
  .filter(ee.Filter.lt('CLOUD_COVER', 30))
  .median()
  .clip(beniROI);

// ── 6.  JRC Permanent Water + Seasonal Water ─────────────────────────────────
var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(beniROI);
var permanentWater  = jrc.select('occurrence').gt(80);
var seasonalWater   = jrc.select('occurrence').gt(10).and(jrc.select('occurrence').lte(80));
var floodedNew      = floodOtsu.and(permanentWater.not());  // new flood only

// ── 7.  WorldPop Population ───────────────────────────────────────────────────
var worldpop = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('country', 'BOL'))
  .filter(ee.Filter.eq('year', 2018))
  .first()
  .clip(beniROI);

// Exposure at flood boundary
var floodedPop = worldpop.updateMask(floodedNew);

// ── 8.  Visualisation Parameters ─────────────────────────────────────────────
var sarVis = {min: -25, max: 0, palette: ['black', 'white']};
var sarRGBvis = {bands: ['VV', 'VH', 'VV'], min: [-20, -25, -20], max: [0, -5, 0]};

var changeVis = {
  min: -8, max: 2,
  palette: ['#08306b', '#2171b5', '#6baed6', '#f7f7f7', '#fd8d3c', '#d94801', '#7f2704']
};

var handVis  = {min: 0, max: 30, palette: ['#08519c','#6baed6','#bdd7e7','#eff3ff',
                '#ffffcc','#fecc5c','#fd8d3c','#f03b20','#bd0026']};
var gateVis  = {min: 0, max: 1, palette: ['#bd0026','#fd8d3c','#fecc5c','#ffffb2',
                '#d9f0a3','#74c476','#238b45']};

var l8TrueVis  = {bands: ['B4','B3','B2'], min: 0.01, max: 0.25, gamma: 1.3};
var l8FalseVis = {bands: ['B5','B4','B3'], min: 0.01, max: 0.30, gamma: 1.3};  // NIR-R-G
var l8SWIRvis  = {bands: ['B6','B5','B4'], min: 0.01, max: 0.40, gamma: 1.3};  // SWIR-NIR-R (water=dark)

var floodVis   = {min: 0, max: 1, palette: ['white', '#08519c']};
var popVis     = {min: 0, max: 50, palette: ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']};

// ── 9.  Add Layers to Map ─────────────────────────────────────────────────────

// --- Landsat 8 basemaps ---
Map.addLayer(l8Pre,   l8TrueVis,  '🛰️ L8 True Colour (Pre-flood, 2017)', true, 0.9);
Map.addLayer(l8Flood, l8FalseVis, '🛰️ L8 False Colour NIR (Flood, 2018)', false);
Map.addLayer(l8Flood, l8SWIRvis,  '🛰️ L8 SWIR (water=dark, 2018)', false);

// --- Sentinel-1 SAR ---
Map.addLayer(s1Pre,   {bands:['VV'], min:-25, max:0, palette:['black','white']},
             '📡 S1 VV Pre-flood (2017)', false);
Map.addLayer(s1Post,  {bands:['VV'], min:-25, max:0, palette:['black','white']},
             '📡 S1 VV Post-flood (2018)', true, 0.9);
Map.addLayer(s1Post,  sarRGBvis, '📡 S1 RGB composite (VV/VH, 2018)', false);

// --- SAR Change Detection ---
Map.addLayer(vvChange, changeVis, '⚡ VV Change (post-pre dB)', false);
Map.addLayer(vhChange, changeVis, '⚡ VH Change (post-pre dB)', false);

// --- Flood masks ---
Map.addLayer(permanentWater.updateMask(permanentWater),
             {palette:['#08519c']}, '💧 JRC Permanent water', false, 0.7);
Map.addLayer(floodOtsu.updateMask(floodOtsu),
             {palette:['#e41a1c']}, '🌊 Otsu flood detection (S1 VV<-14 & ΔVV<-3dB)', true, 0.7);
Map.addLayer(floodedNew.updateMask(floodedNew),
             {palette:['#ff7f00']}, '🌊 New flood (Otsu minus permanent water)', false, 0.8);

// --- HAND and gate ---
Map.addLayer(hand,     handVis, '⛰️ HAND — Height Above Nearest Drainage (m)', false, 0.8);
Map.addLayer(handGate, gateVis, '🔮 HAND Gate α = exp(-h/50) [TerrainFlood model]', false, 0.85);

// --- Population ---
Map.addLayer(worldpop.updateMask(worldpop.gt(0)),
             popVis, '👥 WorldPop 2018 (people/100m pixel)', false, 0.7);
Map.addLayer(floodedPop.updateMask(floodedPop.gt(0)),
             {min:0, max:20, palette:['#ffffb2','#fe9929','#cc4c02']},
             '🚨 Population in flood zone', true, 0.85);

// ── 10.  Sen1Floods11 Chip Footprints ─────────────────────────────────────────
// The 15 Bolivia test chips from the Sen1Floods11 dataset
// These are the chips used to evaluate TerrainFlood-UQ (OOD test set)
// Approximate bounding boxes in Beni department
var chipGeometries = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-64.90, -14.83]), {chip_id: 'Bolivia_129334', flood_px: 169483}),
  ee.Feature(ee.Geometry.Point([-65.30, -13.92]), {chip_id: 'Bolivia_314919', flood_px: 86588}),
  ee.Feature(ee.Geometry.Point([-64.75, -14.20]), {chip_id: 'Bolivia_432776', flood_px: 78263}),
  ee.Feature(ee.Geometry.Point([-65.10, -14.55]), {chip_id: 'Bolivia_242570', flood_px: 51248}),
  ee.Feature(ee.Geometry.Point([-64.50, -13.80]), {chip_id: 'Bolivia_103757', flood_px: 32247}),
  ee.Feature(ee.Geometry.Point([-65.80, -13.60]), {chip_id: 'Bolivia_290290', flood_px: 22356}),
  ee.Feature(ee.Geometry.Point([-64.20, -14.10]), {chip_id: 'Bolivia_294583', flood_px: 14547}),
  ee.Feature(ee.Geometry.Point([-65.50, -14.80]), {chip_id: 'Bolivia_60373',  flood_px: 9180}),
  ee.Feature(ee.Geometry.Point([-63.80, -13.50]), {chip_id: 'Bolivia_23014',  flood_px: 8405}),
  ee.Feature(ee.Geometry.Point([-66.20, -13.10]), {chip_id: 'Bolivia_379434', flood_px: 5951}),
  ee.Feature(ee.Geometry.Point([-65.90, -14.30]), {chip_id: 'Bolivia_312675', flood_px: 3622}),
  ee.Feature(ee.Geometry.Point([-64.60, -13.30]), {chip_id: 'Bolivia_233925', flood_px: 742}),
  ee.Feature(ee.Geometry.Point([-63.50, -14.50]), {chip_id: 'Bolivia_360519', flood_px: 1192}),
  ee.Feature(ee.Geometry.Point([-66.50, -12.80]), {chip_id: 'Bolivia_195474', flood_px: 351}),
  ee.Feature(ee.Geometry.Point([-63.20, -12.50]), {chip_id: 'Bolivia_76104',  flood_px: 0})
]);

// Draw chip footprints as 5.12km × 5.12km squares (512 px × 10m)
var chipFootprints = chipGeometries.map(function(f) {
  var pt = f.geometry();
  var buf = pt.buffer(2560, 1).bounds();  // 5.12km / 2 = 2560m buffer
  return f.setGeometry(buf);
});

Map.addLayer(chipFootprints,
             {color: 'FFFF00', fillColor: '00000000', width: 1.5},
             '📦 Sen1Floods11 Bolivia chips (n=15)', true);

// Add chip labels
var chipLabels = chipGeometries.map(function(f) {
  return f.set('label', ee.String(f.get('chip_id')).slice(8));
});

// ── 11.  Statistics Panel ─────────────────────────────────────────────────────
// Print area statistics to console
print('=== TerrainFlood-UQ Bolivia Analysis ===');
print('Model: TerrainFlood-UQ D_full (ResNet-34 Siamese + HAND gate)');
print('Test set: Bolivia (OOD) — 15 chips, 2,867,815 valid pixels');
print('Best IoU: 0.724 | ECE: 0.063 | TTA uncertainty corr: r=+0.614');
print('');

var floodArea = floodOtsu.multiply(ee.Image.pixelArea())
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: beniROI, scale: 100, maxPixels: 1e10});
print('Estimated flood area (Otsu, Beni region):', floodArea);

var popExposed = worldpop.updateMask(floodedNew)
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: beniROI, scale: 100, maxPixels: 1e10});
print('Estimated population in new flood zone:', popExposed);

// ── 12.  Export Tasks ─────────────────────────────────────────────────────────
// Run these exports from the Tasks tab to download GeoTIFFs

// Export S1 post-flood VV
Export.image.toDrive({
  image: s1Post.select('VV'),
  description: 'Bolivia_S1_VV_postflood_2018',
  folder: 'TerrainFlood_GEE',
  fileNamePrefix: 'Bolivia_S1_VV_postflood_2018',
  region: beniROI,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// Export S1 VV change
Export.image.toDrive({
  image: vvChange,
  description: 'Bolivia_S1_VVchange_2018',
  folder: 'TerrainFlood_GEE',
  fileNamePrefix: 'Bolivia_S1_VVchange_2018',
  region: beniROI,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// Export HAND
Export.image.toDrive({
  image: hand,
  description: 'Bolivia_HAND_MERIT_Hydro',
  folder: 'TerrainFlood_GEE',
  fileNamePrefix: 'Bolivia_HAND_MERIT_Hydro',
  region: beniROI,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// Export HAND gate
Export.image.toDrive({
  image: handGate,
  description: 'Bolivia_HAND_gate_alpha',
  folder: 'TerrainFlood_GEE',
  fileNamePrefix: 'Bolivia_HAND_gate_alpha',
  region: beniROI,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// Export Landsat 8 flood composite
Export.image.toDrive({
  image: l8Flood.select(['B4','B3','B2','B5','B6']),
  description: 'Bolivia_Landsat8_flood_composite_2018',
  folder: 'TerrainFlood_GEE',
  fileNamePrefix: 'Bolivia_Landsat8_flood_2018',
  region: beniROI,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// Export flood extent
Export.image.toDrive({
  image: floodOtsu.toInt(),
  description: 'Bolivia_flood_extent_Otsu_2018',
  folder: 'TerrainFlood_GEE',
  fileNamePrefix: 'Bolivia_flood_extent_Otsu_2018',
  region: beniROI,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// Export population exposed
Export.image.toDrive({
  image: floodedPop,
  description: 'Bolivia_population_in_flood_2018',
  folder: 'TerrainFlood_GEE',
  fileNamePrefix: 'Bolivia_population_flood_2018',
  region: beniROI,
  scale: 100,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// ── 13.  Legend Panel ─────────────────────────────────────────────────────────
// Create a simple HTML legend in the map panel
var legend = ui.Panel({
  style: {position: 'bottom-left', padding: '8px 15px',
          backgroundColor: 'rgba(255,255,255,0.9)'}
});

legend.add(ui.Label('TerrainFlood-UQ — Bolivia 2018', {fontWeight: 'bold', fontSize: '14px'}));
legend.add(ui.Label('Sentinel-1 + HAND + Landsat 8', {fontSize: '11px', color: '#666'}));
legend.add(ui.Label('─────────────────────────', {color: '#ccc'}));

var addLegendRow = function(color, label) {
  var colorBox = ui.Label({style: {backgroundColor: color, padding: '8px', margin: '0 4px 4px 0'}});
  var row = ui.Panel([colorBox, ui.Label(label, {margin: '4px 0'})],
                      ui.Panel.Layout.flow('horizontal'));
  legend.add(row);
};

addLegendRow('#08519c', 'Flood (Otsu detection)');
addLegendRow('#e41a1c', 'New flood extent (2018)');
addLegendRow('#253494', 'Permanent water (JRC)');
addLegendRow('#fd8d3c', 'HAND gate α=exp(-h/50)');
addLegendRow('#FFFF00', 'Sen1Floods11 test chips');

Map.add(legend);

print('');
print('=== Script complete. Check Tasks tab to run exports. ===');
print('Map layers loaded. Toggle visibility in Layers panel.');
