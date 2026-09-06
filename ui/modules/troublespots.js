import { liveDataSync } from './_datasync.js';
import { hoverPopup } from './_hoverpopup.js';
import { fetchOrThrow, buildPopupHtml } from './_feedhelpers.js';
import { replaceSlot, removeLegend } from './_legend.js';
import { opacityUniform } from './_opacity.js';

// Escalating hatch density/color per severity band (Elevated/High/Severe -- driven by
// how many of the 4-type roster converge in a cell, see lib/troublespot_contours.py).
// Severe's color deliberately matches world_events.js's "Conflict or War" marker color,
// so a severe troublespot visually rhymes with the conflict marker elsewhere on the map.
const BAND_STYLE = {
    elevated: { label: 'Elevated', color: '#e8b339', spacing: 10, lineWidth: 1.5, crosshatch: false },
    high: { label: 'High', color: '#d9702e', spacing: 6, lineWidth: 2, crosshatch: false },
    severe: { label: 'Severe', color: '#a30000', spacing: 5, lineWidth: 2, crosshatch: true },
};
const BAND_ORDER = ['elevated', 'high', 'severe'];

// Matches _bands_to_geojson's (db/troublespot_adapter.py) per-feature breakdown keys.
const TYPE_LABELS = {
    earthquake: 'Earthquakes',
    fire: 'Fires',
    volcanic_activity: 'Volcanic Activity',
    world_event: 'World Events',
};

const patternImageId = (band) => `troublespots-hatch-${band}`;

// Draws a small tileable diagonal-hatch (or crosshatch, for Severe) pattern -- fill-
// pattern needs a registered image, not a raw color, so hatching (rather than a flat
// fill) requires this instead of just fill-color.
function buildHatchPattern({ color, spacing, lineWidth, crosshatch }) {
    const size = 16;
    const canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    const drawDiagonals = () => {
        for (let x = -size; x < size * 2; x += spacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x + size, size);
            ctx.stroke();
        }
    };
    drawDiagonals();
    if (crosshatch) {
        // Mirror the same diagonals the other way for a crosshatch (Severe only).
        ctx.save();
        ctx.translate(size, 0);
        ctx.scale(-1, 1);
        drawDiagonals();
        ctx.restore();
    }
    return ctx.getImageData(0, 0, size, size);
}

// Registered once per map instance (not per mount): map.addImage throws if an id
// already exists, and a basemap style swap (setStyle wipes ALL images, not just
// layers/sources) means this must be re-checked every mount, not assumed done once.
function ensurePatterns(map) {
    for (const band of BAND_ORDER) {
        const id = patternImageId(band);
        if (!map.hasImage(id)) map.addImage(id, buildHatchPattern(BAND_STYLE[band]));
    }
}

const fillPatternExpr = ['match', ['get', 'band'],
    'elevated', patternImageId('elevated'),
    'high', patternImageId('high'),
    'severe', patternImageId('severe'),
    patternImageId('elevated'),
];
const lineColorExpr = ['match', ['get', 'band'],
    'elevated', BAND_STYLE.elevated.color,
    'high', BAND_STYLE.high.color,
    'severe', BAND_STYLE.severe.color,
    BAND_STYLE.elevated.color,
];

const LEGEND_SLOT_ID = 'troublespots-legend-slot';

// A custom 3-swatch legend, not standardLegend()/drawKey() (_legend.js) -- that
// helper renders a continuous vmin/vmax colorbar, the wrong shape for 3 discrete
// named bands. Only the generic slot lifecycle (replaceSlot/removeLegend) is reused.
function renderLegend(cfg) {
    if (opacityUniform(cfg, 0.7) <= 0) { removeLegend(LEGEND_SLOT_ID); return; }
    replaceSlot(LEGEND_SLOT_ID, (slot) => {
        slot.style.fontSize = '11px';
        slot.style.color = '#ddd';
        const title = document.createElement('div');
        title.textContent = 'Troublespots';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '4px';
        slot.appendChild(title);
        for (const band of BAND_ORDER) {
            const style = BAND_STYLE[band];
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '6px';
            row.style.marginBottom = '2px';
            const swatch = document.createElement('canvas');
            swatch.width = 16; swatch.height = 16;
            swatch.style.width = '16px';
            swatch.style.height = '16px';
            swatch.style.border = `1px solid ${style.color}`;
            swatch.getContext('2d').putImageData(buildHatchPattern(style), 0, 0);
            const label = document.createElement('span');
            label.textContent = style.label;
            row.appendChild(swatch);
            row.appendChild(label);
            slot.appendChild(row);
        }
    });
}

export function loadLayer(map, config) {
    const sourceId = 'troublespots-source';
    const fillLayerId = 'troublespots-fill';
    const lineLayerId = 'troublespots-outline';
    let stopPopup = null;

    const urlFor = (cfg) => `${window.WM_API}/troublespots/geojson`
        + `?cell_size_deg=${cfg.cell_size_deg ?? 2.0}&window_hours=${cfg.window_hours ?? 48}&t=${Date.now()}`;

    const fetchData = (cfg) => fetchOrThrow(urlFor(cfg));

    const popupHtml = (f) => {
        const d = f.properties;
        const style = BAND_STYLE[d.band] || BAND_STYLE.elevated;
        const rows = Object.entries(TYPE_LABELS)
            .map(([type, label]) => ({ label, value: String(d[type] || 0) }));
        return buildPopupHtml({
            title: { text: `${style.label} Troublespot`, variant: 'plain' },
            blocks: [{ type: 'divider' }, { type: 'rows', rows }],
        });
    };

    const mount = async (cfg) => {
        ensurePatterns(map);
        const data = await fetchData(cfg);
        if (map.getSource(sourceId)) return;   // guard against races
        map.addSource(sourceId, { type: 'geojson', data });
        map.addLayer({
            id: fillLayerId, type: 'fill', source: sourceId,
            paint: { 'fill-pattern': fillPatternExpr, 'fill-opacity': opacityUniform(cfg, 0.7) },
        });
        map.addLayer({
            id: lineLayerId, type: 'line', source: sourceId,
            paint: {
                'line-color': lineColorExpr, 'line-width': 1.5,
                'line-opacity': opacityUniform(cfg, 0.7),
            },
        });
        stopPopup = hoverPopup(map, fillLayerId, { html: popupHtml });
        renderLegend(cfg);
    };

    const refresh = async (cfg) => {
        const data = await fetchData(cfg);
        map.getSource(sourceId)?.setData(data);
        const opacity = opacityUniform(cfg, 0.7);
        if (map.getLayer(fillLayerId)) map.setPaintProperty(fillLayerId, 'fill-opacity', opacity);
        if (map.getLayer(lineLayerId)) map.setPaintProperty(lineLayerId, 'line-opacity', opacity);
        renderLegend(cfg);
    };

    const unmount = () => {
        stopPopup?.();
        removeLegend(LEGEND_SLOT_ID);
        if (map.getLayer(lineLayerId)) map.removeLayer(lineLayerId);
        if (map.getLayer(fillLayerId)) map.removeLayer(fillLayerId);
        if (map.getSource(sourceId)) map.removeSource(sourceId);
    };

    // 300s: Troublespots is derived from 4 sources with their own, generally slower
    // cadences (World Events ~15min being the fastest) and each recomputation is a
    // real backend computation, not a cache read -- no need for World Events' own
    // tighter 60s poll.
    return liveDataSync(map, {
        sectionKey: 'troublespots', initialConfig: config, mount, refresh, unmount, refreshMs: 300000,
    });
}
