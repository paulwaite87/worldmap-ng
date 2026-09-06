import { liveDataSync } from './_datasync.js';
import { hoverPopup } from './_hoverpopup.js';
import { fetchOrThrow, escapeHtml, buildPopupHtml } from './_feedhelpers.js';

// No image assets -- world events render as a native circle layer (radius by
// corroboration, color by category), same choice fires.js made for the same reason
// (see that file's own comment): a new layer with no pre-made icon set doesn't need
// one when a color-coded circle already carries the category distinction.
const CATEGORY_LABELS = {
    explosion: 'Explosion',
    warfare: 'Conflict or War',
    targeted_violence: 'Targeted / Mass Violence',
    diplomacy: 'Diplomatic Meeting',
};

// Matches _feedhelpers.js's TITLE_VARIANTS entries of the same names exactly, so a
// popup's title color always matches its marker's circle-color.
const CATEGORY_COLORS = {
    explosion: '#ff8c00',
    warfare: '#a30000',
    targeted_violence: '#3a0d0d',
    diplomacy: '#1f6feb',
};

const CATEGORY_TOGGLE_KEYS = {
    explosion: 'show_explosion',
    warfare: 'show_warfare',
    targeted_violence: 'show_targeted_violence',
    diplomacy: 'show_diplomacy',
};

// Every category defaults to shown when its toggle is unset (matches every other
// layer's boolean-setting convention: absence means "on" unless the config default
// says otherwise -- config/atmos-gl.json.tmpl's world_events section already defaults
// all four to true).
const visibleCategories = (cfg) => Object.keys(CATEGORY_TOGGLE_KEYS)
    .filter((cat) => cfg[CATEGORY_TOGGLE_KEYS[cat]] !== false);

const filterFor = (cfg) => ['in', ['get', 'category'], ['literal', visibleCategories(cfg)]];

export function loadLayer(map, config) {
    const sourceId = 'world-events-source';
    const layerId = 'world-events-layer';
    let stopPopup = null;

    const urlFor = (cfg) => `${window.WM_API}/world_events/geojson`
        + `?expiry_days=${cfg.expiry_days ?? 7}&t=${Date.now()}`;

    const fetchData = (cfg) => fetchOrThrow(urlFor(cfg));

    const radiusExpr = (cfg) => [
        '*',
        ['interpolate', ['linear'], ['get', 'num_mentions'], 10, 4, 50, 6, 500, 10],
        Number(cfg.marker_size) || 1.0,
    ];

    const popupHtml = (f) => {
        const d = f.properties;
        const rows = [];
        if (d.actor1_name && d.actor2_name) {
            rows.push({ label: 'Actors', value: `${escapeHtml(d.actor1_name)} → ${escapeHtml(d.actor2_name)}`, raw: true, width: 60 });
        } else if (d.actor1_name || d.actor2_name) {
            rows.push({ label: 'Actor', value: d.actor1_name || d.actor2_name, width: 60 });
        }
        if (d.place) rows.push({ label: 'Place', value: d.place, width: 60 });
        if (d.event_date) {
            const dateStr = new Date(d.event_date).toLocaleString(undefined,
                { year: 'numeric', month: 'short', day: 'numeric' });
            rows.push({ label: 'Date', value: dateStr, width: 60 });
        }
        if (d.num_sources != null) {
            const plural = d.num_sources === 1 ? '' : 's';
            rows.push({ label: 'Sources', value: `Reported by ${d.num_sources} source${plural}`, width: 60 });
        }

        const blocks = [{ type: 'divider' }];
        if (rows.length) blocks.push({ type: 'rows', rows });
        if (d.source_url) {
            const href = escapeHtml(d.source_url);
            blocks.push({
                type: 'notice',
                raw: true,
                color: '#6c757d',
                text: `<a href="${href}" target="_blank" rel="noopener noreferrer">Read more →</a>`,
            });
        }

        return buildPopupHtml({
            title: { text: CATEGORY_LABELS[d.category] || d.category, variant: d.category },
            blocks,
        });
    };

    const mount = async (cfg) => {
        const data = await fetchData(cfg);
        if (map.getSource(sourceId)) return;          // guard against races
        map.addSource(sourceId, { type: 'geojson', data });
        map.addLayer({
            id: layerId, type: 'circle', source: sourceId,
            filter: filterFor(cfg),
            paint: {
                'circle-radius': radiusExpr(cfg),
                'circle-color': [
                    'match', ['get', 'category'],
                    'explosion', CATEGORY_COLORS.explosion,
                    'warfare', CATEGORY_COLORS.warfare,
                    'targeted_violence', CATEGORY_COLORS.targeted_violence,
                    'diplomacy', CATEGORY_COLORS.diplomacy,
                    '#888888',
                ],
                'circle-opacity': Number(cfg.opacity ?? 80) / 100,
                'circle-stroke-width': 0.5,
                'circle-stroke-color': 'rgba(0,0,0,0.6)',
            },
        });
        stopPopup = hoverPopup(map, layerId, { html: popupHtml });
    };

    const refresh = async (cfg) => {
        const data = await fetchData(cfg);
        map.getSource(sourceId)?.setData(data);
        if (map.getLayer(layerId)) {
            map.setFilter(layerId, filterFor(cfg));
            map.setPaintProperty(layerId, 'circle-radius', radiusExpr(cfg));
            map.setPaintProperty(layerId, 'circle-opacity', Number(cfg.opacity ?? 80) / 100);
        }
    };

    const unmount = () => {
        stopPopup?.();
        if (map.getLayer(layerId))   map.removeLayer(layerId);
        if (map.getSource(sourceId)) map.removeSource(sourceId);
    };

    return liveDataSync(map, {
        sectionKey: 'world_events', initialConfig: config, mount, refresh, unmount, refreshMs: 60000,
    });
}
