import { createStaticFillLayer } from './_webglfill.js';
import { standardLegend, insertBeforeExtension } from './_legend.js';
import { opacityUniform } from './_opacity.js';

// Both Live (NASA MODIS observed flooding) and Historical (JRC 100yr hazard) are
// rendered as STATIC (non-animated) textures, not per-forecast-hour ones: FloodRiskUpdater.run()
// (tasks/flood_risk.py) already renders both variants every cycle and publishes only
// the currently-configured mode's latest content to the canonical path, mirroring
// greenhouse_gases.js's species/mode toggle exactly -- so switching modes here needs
// no timeline/forecast_state machinery, just a poll-refresh of a different filename.

// A small-integer CATEGORY lookup (nearest-integer, flat colour per bucket -- no
// interpolation between categories), unlike _colormaps.js's buildScaledLUT/
// buildThresholdLUT which are built for continuous physical quantities with
// sub-range gradients. `colors` are [r,g,b,a] floats in 0..1; index 0 is always the
// "no hazard" bucket (fully transparent, so unaffected land reads as plain basemap).
function buildCategoryLUT(colors) {
    const vmax = colors.length - 1;
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
        const cat = Math.min(vmax, Math.max(0, Math.round((i / 255) * vmax)));
        const c = colors[cat];
        const o = i * 4;
        lut[o] = Math.round(c[0] * 255);
        lut[o + 1] = Math.round(c[1] * 255);
        lut[o + 2] = Math.round(c[2] * 255);
        lut[o + 3] = Math.round((c[3] ?? 1) * 255);
    }
    return lut;
}

// Live: NASA MODIS observed flood detection -- binary (0 = no flood, 1 = MODIS
// Flood pixel value 3, the 1-Day cloud-shadow-screened product). Mirrors
// tasks/flood_risk.py's _LIVE_ENCODE_DOMAIN == (0.0, 1.0).
const LIVE_COLORS = [
    [0, 0, 0, 0],
    [0.85, 0.05, 0.05, 0.85],
];
const LIVE_LABELS = ['None', 'Flood'];

// Historical: JRC RP100 hazard-depth reclass category (0-4, JRC's own scale). Mirrors
// tasks/flood_risk.py's _HISTORICAL_ENCODE_DOMAIN == (0.0, 4.0). NOT comparable to
// Live's band above -- a different metric on a coincidentally-similar small-integer
// scale, hence a wholly separate colour/label set rather than a shared one (see
// issue #371's design grill).
const HISTORICAL_COLORS = [
    [0, 0, 0, 0],
    [0.55, 0.8, 1.0, 0.55],
    [0.2, 0.55, 0.95, 0.7],
    [0.05, 0.25, 0.75, 0.85],
    [0.35, 0.0, 0.55, 0.95],
];
const HISTORICAL_LABELS = ['None', '<1m', '1-3m', '3-10m', '>10m'];

const ENCODE_DOMAIN = { live: [0, 1], historical: [0, 4] };
const modeOf = (cfg) => (String(cfg.mode || 'live').toLowerCase() === 'historical' ? 'historical' : 'live');

// Matches FloodRiskUpdater._variant_path's f"{base}_{suffix}{ext}" naming
// (tasks/flood_risk.py) -- "data/flood_risk.png" -> "data/flood_risk_live.png" /
// "data/flood_risk_historical.png".
function modeFilename(outfile, mode) {
    return insertBeforeExtension(outfile, `_${mode}`);
}

function keySpecFor(cfg) {
    const mode = modeOf(cfg);
    if (mode === 'historical') {
        return {
            lut: buildCategoryLUT(HISTORICAL_COLORS),
            vmin: 0, vmax: 4, ticks: [0, 1, 2, 3, 4],
            title: 'Flood Hazard Depth (Historical, 100yr)',
            tickFormat: (v) => HISTORICAL_LABELS[Math.round(v)] ?? '',
        };
    }
    return {
        lut: buildCategoryLUT(LIVE_COLORS),
        vmin: 0, vmax: 1, ticks: [0, 1],
        title: 'Observed Flooding (NASA MODIS, 1-Day)',
        tickFormat: (v) => LIVE_LABELS[Math.round(v)] ?? '',
    };
}

export function loadLayer(map, config) {
    const legend = standardLegend('flood-risk-legend-slot', keySpecFor, 1);

    return createStaticFillLayer(map, {
        sectionKey: 'flood_risk',
        initialConfig: config,
        dataUrl: (cfg) => `${window.MAP_UI}/${modeFilename(cfg.outfile, modeOf(cfg))}`,
        physicalDomain: (cfg) => ENCODE_DOMAIN[modeOf(cfg)],
        fragmentBody: `
            uniform float u_alpha;
            vec4 shade(float value, vec2 uv) {
                float t = clamp((value - u_vmin) / u_span, 0.0, 1.0);
                vec4 c = texture(u_cmap, vec2(t, 0.5));
                return vec4(c.rgb, c.a * u_alpha);
            }`,
        customUniforms: (cfg) => ({ u_alpha: opacityUniform(cfg, 0.6) }),
        colormap: (cfg) => buildCategoryLUT(modeOf(cfg) === 'historical' ? HISTORICAL_COLORS : LIVE_COLORS),
        onMount: (cfg) => legend.addLegend(cfg),
        onRefresh: (cfg) => legend.addLegend(cfg),
        onUnmount: legend.removeLegend,
    });
}
