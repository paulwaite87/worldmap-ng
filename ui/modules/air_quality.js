import { liveLayerSync } from './_refresh.js';
import { standardLegend, insertBeforeExtension } from './_legend.js';

// Insert "_<variable>" before the extension: "data/air_quality.png" ->
// "data/air_quality_pm2_5.png". The backend always keeps all 3 variables fresh on
// disk (AirQualityCollector fetches unconditionally of variable; AirQualityUpdater
// renders all 3 every cycle -- see tasks/air_quality.py), so switching variable in
// the config UI applies on this layer's next poll tick with no render wait, same as
// greenhouse_gases.js's species/mode equivalent.
function variableFilename(outfile, variable) {
    return insertBeforeExtension(outfile, `_${variable || 'pm2_5'}`);
}

// Fixed AQI-recognisable gradient (green -> yellow -> orange -> red -> purple) --
// mirrors tasks/air_quality.py's _AQI_COLORS/_AQI_CMAP.
const AQI_PALETTE = [
    [0.0, 0.894, 0.0], [1.0, 1.0, 0.0], [1.0, 0.494, 0.0], [1.0, 0.0, 0.0], [0.561, 0.247, 0.592],
];
// so2_volcanic's own non-overlapping palette -- mirrors _VOLCANIC_SO2_COLORS/_VOLCANIC_SO2_CMAP.
const VOLCANIC_SO2_PALETTE = [
    [0.0, 0.953, 1.0], [0.161, 0.475, 1.0], [0.769, 0.0, 1.0], [1.0, 0.0, 0.784],
];

function buildLUT(palette) {
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
        const fp = (i / 255) * (palette.length - 1);
        const lo = Math.floor(fp), hi = Math.min(lo + 1, palette.length - 1), f = fp - lo;
        const o = i * 4;
        for (let j = 0; j < 3; j++)
            lut[o + j] = Math.round((palette[lo][j] * (1 - f) + palette[hi][j] * f) * 255);
        lut[o + 3] = 255;
    }
    return lut;
}
const AQI_LUT = buildLUT(AQI_PALETTE);
const VOLCANIC_SO2_LUT = buildLUT(VOLCANIC_SO2_PALETTE);

// Mirrors tasks/air_quality.py's per-variable display/scale metadata.
const DISPLAY_LABEL = {
    pm2_5: 'PM2.5', pm10: 'PM10', aod: 'Smoke (AOD)',
    so2: 'SO2 (Sulphur Dioxide)', so2_volcanic: 'SO2 (Volcanic)',
};
const DISPLAY_UNIT = { pm2_5: 'µg/m³', pm10: 'µg/m³', aod: '', so2: 'DU', so2_volcanic: 'DU' };
const TICK_FORMAT = { pm2_5: '%d', pm10: '%d', aod: '%.2f', so2: '%.1f', so2_volcanic: '%.1f' };
const MIN_SETTING_KEY = {
    pm2_5: 'pm2_5_min', pm10: 'pm10_min', aod: 'aod_min', so2: 'so2_min', so2_volcanic: 'so2_min',
};
const DEFAULT_MIN = { pm2_5: 35, pm10: 150, aod: 0.5, so2: 1.0, so2_volcanic: 1.0 };
const FIXED_CEILING = { pm2_5: 250, pm10: 400, aod: 3, so2: 20, so2_volcanic: 20 };

// so2_volcanic's min setting lives in the "volcanoes" section (Volcano Properties),
// not "air_quality" -- mirrors tasks/air_quality.py's _SETTINGS_SECTION_OVERRIDE.
// volcanoesCfg is threaded through from liveLayerSync's globalKeys (see loadLayer).
function keySpecFor(cfg, volcanoesCfg) {
    const variable = cfg.variable || 'pm2_5';
    const settings = variable === 'so2_volcanic' ? (volcanoesCfg || {}) : cfg;
    const vmin = Number(settings[MIN_SETTING_KEY[variable]] ?? DEFAULT_MIN[variable]);
    const vmax = FIXED_CEILING[variable];
    const unit = DISPLAY_UNIT[variable];
    return {
        lut: variable === 'so2_volcanic' ? VOLCANIC_SO2_LUT : AQI_LUT,
        vmin, vmax, ticks: [0, 1, 2, 3, 4].map((i) => vmin + (i / 4) * (vmax - vmin)),
        title: unit ? `${DISPLAY_LABEL[variable]} (${unit})` : DISPLAY_LABEL[variable],
        tickFormat: TICK_FORMAT[variable],
    };
}

// volcanoes.js's smoke-plume legend needs this exact same so2_volcanic scale --
// exported so both stay in lockstep rather than risking drift from two independent
// copies of the same LUT/vmin/vmax/title.
export function so2VolcanicKeySpec(so2Min) {
    return keySpecFor({ variable: 'so2_volcanic' }, { so2_min: so2Min });
}

export function loadLayer(map, config) {
    const sourceId = 'air-quality-source';
    const layerId  = 'air-quality-layer';
    const coordinates = [
        [-180, 85.051129], [180, 85.051129],
        [180, -85.051129], [-180, -85.051129],
    ];
    const urlFor = (cfg) => `${window.MAP_UI}/${variableFilename(cfg.outfile, cfg.variable)}`;
    const legend = standardLegend('air-quality-legend-slot', (cfg) => keySpecFor(cfg, cfg._volcanoes), 1);

    const mount = (cfg, globals) => {
        if (!map.getSource(sourceId)) {
            map.addSource(sourceId, { type: 'image', url: `${urlFor(cfg)}?t=${Date.now()}`, coordinates });
            map.addLayer({ id: layerId, type: 'raster', source: sourceId,
                           paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0 } });
        }
        legend.addLegend({ ...cfg, _volcanoes: globals?.volcanoes });
    };

    const refresh = (cfg, globals) => {
        const s = map.getSource(sourceId);
        if (s) s.updateImage({ url: `${urlFor(cfg)}?t=${Date.now()}` });
        legend.addLegend({ ...cfg, _volcanoes: globals?.volcanoes });
    };

    const unmount = () => {
        if (map.getLayer(layerId))   map.removeLayer(layerId);
        if (map.getSource(sourceId)) map.removeSource(sourceId);
        legend.removeLegend();
    };

    // A variable switch changes urlFor itself (it IS part of the filename), so that
    // case is already covered by the default imageUrl regen chase. A scale-only
    // change doesn't change the filename, but it DOES force a full server-side
    // re-render of the image -- AirQualityUpdater.run()'s freshness check also
    // compares a persisted settings signature, not just source-data mtime (see
    // Updater._is_render_fresh / AirQualityUpdater._variable_settings_signature in
    // tasks/air_quality.py). globalKeys:['volcanoes'] gives so2_volcanic's
    // Volcano-Properties-owned min setting to mount/refresh above (see keySpecFor).
    return liveLayerSync(map, {
        sectionKey: 'air_quality', initialConfig: config, mount, refresh, unmount,
        imageUrl: urlFor, globalKeys: ['volcanoes'],
    });
}
