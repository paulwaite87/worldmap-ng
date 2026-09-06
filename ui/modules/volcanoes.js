import { liveDataSync } from './_datasync.js';
import { hoverPopup } from './_hoverpopup.js';
import { fetchOrThrow, buildPopupHtml, preloadIcons } from './_feedhelpers.js';
import { standardLegend } from './_legend.js';
import { so2VolcanicKeySpec } from './air_quality.js';

// AirQualityUpdater._output_path_for("so2_volcanic") (tasks/air_quality.py), derived
// from lib/output_files.py's OUTFILES["air_quality"] -- rendered unconditionally every
// cycle regardless of whether Air Quality OR Volcanoes is enabled (issue #254), so
// it's always safe to read directly here. Its opacity/threshold are baked in
// server-side from the "volcanoes" section (Volcano Properties is the sole settings
// owner -- see tasks/air_quality.py's _SETTINGS_SECTION_OVERRIDE), not this file.
// so2_volcanic (NOT so2, air_quality's own general SO2 picker option) is a distinct
// CDS variable isolating ash-plume-associated SO2 specifically -- see
// tasks/air_quality.py's _CAMS_VARS and lib/air_quality.py's module docstring.
const SMOKE_IMAGE = 'data/air_quality_so2_volcanic.png';

export function loadLayer(map, config) {
    const sourceId = 'volcanoes-source';
    const layerId  = 'volcanoes-layer';
    const smokeSourceId = 'volcanoes-smoke-source';
    const smokeLayerId  = 'volcanoes-smoke-layer';
    const smokeSlotId   = 'volcanoes-smoke-legend-slot';
    const smokeCoordinates = [
        [-180, 85.051129], [180, 85.051129],
        [180, -85.051129], [-180, -85.051129],
    ];
    const volcanoIcons = [
        { id: 'volcano-new', url: '/images/volcano_new.png' },
        { id: 'volcano-continuing', url: '/images/volcano_continuing.png' },
    ];
    let stopPopup = null;

    const urlFor = () => `${window.WM_API}/volcanoes/geojson?t=${Date.now()}`;

    const fetchData = () => fetchOrThrow(urlFor());

    const popupHtml = (f) => {
        const p = f.properties;
        const rows = [{ label: 'Country', value: p.country || 'N/A' }];
        if (p.activity_type) rows.push({ label: 'Activity', value: p.activity_type });
        if (p.report_description) rows.push({ label: 'Report', value: p.report_description });
        if (p.hans_alert_level || p.hans_color_code) {
            rows.push({
                label: 'USGS Alert',
                value: `${p.hans_alert_level || 'N/A'} (${p.hans_color_code || 'N/A'})`,
            });
        }
        return buildPopupHtml({
            title: { text: p.name || 'Unknown Volcano', variant: 'callsign' },
            blocks: [{ type: 'divider' }, { type: 'rows', rows }],
            padding: 3,
        });
    };

    // The smoke overlay's own opacity is baked into SMOKE_IMAGE's pixels server-side
    // (Volcano Properties' smoke_opacity setting) -- this raster-opacity is a fixed
    // constant, matching air_quality.js's own hardcoded 0.85, not a second
    // client-side opacity control. Legend keys render entirely client-side (see
    // _legend.js's module docstring) -- so2VolcanicKeySpec reuses air_quality.js's
    // own so2_volcanic LUT/scale so the two stay in lockstep. No opacityFallback:
    // show_smoke_plume (below) is already this overlay's real show/hide gate,
    // smoke_opacity isn't a live per-viewer opacity to hide the key on.
    const smokeLegend = standardLegend(smokeSlotId, (cfg) => so2VolcanicKeySpec(cfg.so2_min));

    const addSmokeLayer = (cfg) => {
        if (map.getSource(smokeSourceId)) return;
        map.addSource(smokeSourceId, {
            type: 'image', url: `${window.MAP_UI}/${SMOKE_IMAGE}?t=${Date.now()}`, coordinates: smokeCoordinates,
        });
        map.addLayer({
            id: smokeLayerId, type: 'raster', source: smokeSourceId,
            paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0 },
        });
        smokeLegend.addLegend(cfg);
    };

    const removeSmokeLayer = () => {
        if (map.getLayer(smokeLayerId))   map.removeLayer(smokeLayerId);
        if (map.getSource(smokeSourceId)) map.removeSource(smokeSourceId);
        smokeLegend.removeLegend();
    };

    const syncSmokeLayer = (cfg) => {
        if (!cfg.show_smoke_plume) { removeSmokeLayer(); return; }
        if (!map.getSource(smokeSourceId)) { addSmokeLayer(cfg); return; }
        map.getSource(smokeSourceId)?.updateImage({ url: `${window.MAP_UI}/${SMOKE_IMAGE}?t=${Date.now()}` });
        smokeLegend.addLegend(cfg);
    };

    const mount = async (cfg) => {
        await preloadIcons(map, volcanoIcons);
        const data = await fetchData();
        if (!map.getSource(sourceId)) {
            map.addSource(sourceId, { type: 'geojson', data });
            map.addLayer({
                id: layerId, type: 'symbol', source: sourceId,
                layout: {
                    'icon-image': ['case', ['get', 'is_new'], 'volcano-new', 'volcano-continuing'],
                    'icon-size': 0.6 * (cfg.icon_zoom ?? 1.0),
                    'icon-allow-overlap': true, 'icon-ignore-placement': true,
                },
            });
            stopPopup = hoverPopup(map, layerId, { html: popupHtml });
        }
        syncSmokeLayer(cfg);
    };

    const refresh = async (cfg) => {
        const data = await fetchData();
        map.getSource(sourceId)?.setData(data);
        syncSmokeLayer(cfg);
    };

    const unmount = () => {
        stopPopup?.();
        if (map.getLayer(layerId))   map.removeLayer(layerId);
        if (map.getSource(sourceId)) map.removeSource(sourceId);
        removeSmokeLayer();
    };

    // GVP's report is weekly; HANS enrichment is the only thing that could change
    // faster, so a long refresh (matching the old static layer's cadence) is still
    // fine -- also plenty frequent enough for the smoke overlay, which re-renders
    // roughly hourly server-side.
    return liveDataSync(map, { sectionKey: 'volcanoes', initialConfig: config, mount, refresh, unmount, refreshMs: 600000 });
}
