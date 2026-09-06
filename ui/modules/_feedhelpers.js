// ui/modules/_feedhelpers.js
/**
 * Shared fetch/icon/popup plumbing behind every popup-bearing layer (quakes.js,
 * volcanoes.js, satellites.js, storms.js, lightning.js, shipping.js, flightradar.js,
 * fires.js, markers.js) -- architecture review candidate #6, "make popup
 * functionality a one-stop-shop for every instance", superseding
 * docs/adr/0002-dont-extend-hoverpopup-for-markers.md (markers.js was the one
 * deliberate holdout; it's since migrated too, alongside hoverPopup.js's own
 * widened event/layer/enabled interface). mount/refresh/unmount stay bespoke per
 * module (layer count and pulse wiring genuinely vary); buildPopupHtml (below) owns
 * popup CONTENT, hoverPopup (_hoverpopup.js) owns the show/hide/positioning
 * mechanic -- together they're the one-stop-shop, not this file alone.
 */

// HTML-entity-escape untrusted text before it's interpolated into a popup template
// string handed to maplibregl.Popup.setHTML() -- which parses that string as real
// HTML/DOM, same trust level as innerHTML. Every popup-bearing layer's data ultimately
// comes from an external feed/API (GVP, USGS, NHC/JTWC, Celestrak, and -- critically --
// AIS ShipStaticData and ADS-B, both of which are self-reported by the vessel/aircraft
// transponder with NO validation, so a ship or aircraft's reported name/callsign/
// destination is fully attacker-controlled free text). This is the actual XSS-blocking
// control: correct regardless of what any given collector does or doesn't strip at
// ingest, and unlike tag-stripping it can't be bypassed by a payload that isn't
// tag-shaped (e.g. one relying on & already being unencoded in the stored value).
export function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

export async function fetchOrThrow(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
}

// Icon-array preloader shared by quakes.js/lightning.js/shipping.js/flightradar.js --
// was byte-for-byte identical in lightning.js/shipping.js; quakes.js's copy was
// missing the !res.ok check, silently fixed by unifying onto this one. volcanoes.js's
// single-icon case has its own post-await hasImage re-check (a race-guard this
// three-icon version doesn't need) and stays bespoke.
//
// An icon entry with `sdf: true` (flightradar.js's aircraft/glider icons -- a white
// silhouette on transparent) is registered as an SDF image, so a layer can tint it at
// render time via the icon-color paint property instead of needing a separately-baked
// PNG per color. Every other caller's icons are pre-colored PNGs and omit `sdf`,
// which must call addImage with exactly its original two-argument signature -- some
// mocked `map`s in tests assert against that exact call shape.
export async function preloadIcons(map, icons) {
    await Promise.all(icons.map(async (ic) => {
        if (map.hasImage(ic.id)) return;
        const res = await fetch(`${window.location.origin}${ic.url}`);
        if (!res.ok) throw new Error(`Could not load ${ic.id}`);
        const bitmap = await createImageBitmap(await res.blob());
        if (ic.sdf) {
            map.addImage(ic.id, bitmap, { sdf: true });
        } else {
            map.addImage(ic.id, bitmap);
        }
    }));
}

// The one-stop-shop popup content model (architecture review candidate #6,
// superseding docs/adr/0002-dont-extend-hoverpopup-for-markers.md) -- replaces
// popupCard AND every hand-rolled popup template (quakes/lightning/flightradar/
// shipping/markers) with one function, a small named set of title variants, and an
// ordered list of typed content blocks. Every current popup shape (a fused title
// line, a <br>-separated field list, a conditional route callout, a live-computed
// row colour, a no-data fallback) is expressible as one of these blocks -- there is
// deliberately no raw-HTML "any content" block; if a future popup needs a shape none
// of these cover, add a new named block type rather than reaching for an escape
// hatch that would quietly undo the unification.
//
// Escaping: title/subtitle/emphasis/rows/line/notice all take PRE-BUILT HTML for any
// field marked `raw`, or auto-escape it via escapeHtml() otherwise (default false --
// safe by default, matching popupCard's own "escaped here, not left to each caller"
// stance, since every field here ultimately traces back to an external feed). `raw`
// exists only for the handful of cases that need it: pre-formatted HTML entities
// (flight radar's `&deg;`/`&#9888;`) or a caller-assembled fragment (title.suffix,
// subtitle, emphasis.html) built from already-escaped pieces.

const TITLE_VARIANTS = {
    default: { color: '#333', size: 13 },   // popupCard's own former default
    callsign: { color: '#007bff', size: 16 }, // flight radar / volcanoes / shipping
    alert: { color: '#ff4a4a', size: 14 },    // storms / lightning / quakes
    plain: { color: '#000', size: 14 },       // markers -- bold, no accent colour
    fire: { color: '#ff5a1f', size: 13 },     // fires -- distinct orange, not alert's red
    // world_events -- one variant per category (not a single shared variant), keyed by
    // the same string the backend stores as `category` (world_events.js passes it
    // straight through as the variant name), so each category's popup title matches its
    // marker color exactly.
    explosion: { color: '#ff8c00', size: 13 },
    warfare: { color: '#a30000', size: 13 },
    targeted_violence: { color: '#3a0d0d', size: 13 },
    diplomacy: { color: '#1f6feb', size: 13 },
};

const LABEL_COLOR = '#666';

function escapeMaybe(value, raw) {
    return raw ? value : escapeHtml(value);
}

function renderTitle({ text, variant = 'default', suffix = '' }) {
    const { color, size } = TITLE_VARIANTS[variant] || TITLE_VARIANTS.default;
    return `<strong style="font-size:${size}px;color:${color};">${escapeHtml(text)}</strong>${suffix}`;
}

function renderSubtitle(subtitle) {
    return subtitle
        ? `<div style="color:#888;font-size:11px;margin-top:-2px;">${subtitle}</div>`
        : '';
}

function renderRows(rows) {
    return rows.map(({ label, value, width = 45, valueColor = LABEL_COLOR, raw = false }) =>
        `<div><strong style="min-width:${width}px;display:inline-block;margin-right:6px;">${escapeHtml(label)}:</strong>` +
        `<span style="color:${valueColor};">${escapeMaybe(value, raw)}</span></div>`
    ).join('');
}

function renderLine(items) {
    // Block-level (a <div>, not a trailing <br>) so a line always starts on its own
    // line regardless of what precedes it -- title, another line, or a divider --
    // the same reason `rows` wraps each row in a <div> rather than relying on <br>.
    const content = items.map(({ label, value, raw = false }) =>
        `<span style="color:${LABEL_COLOR};">${escapeHtml(label)}:</span> ${escapeMaybe(value, raw)}`
    ).join(' | ');
    return `<div>${content}</div>`;
}

function renderBlock(block) {
    switch (block.type) {
        case 'divider':
            return '<hr style="border:0;border-top:1px solid #ccc;margin:4px 0;">';
        case 'rows':
            return renderRows(block.rows);
        case 'line':
            return renderLine(block.items);
        case 'emphasis':
            return `<div style="font-weight:bold;color:#000;font-size:20px;margin-top:2px;">${block.html}</div>`;
        case 'notice':
            return `<div style="color:${block.color || '#c0392b'};font-size:11px;margin-top:4px;">${escapeMaybe(block.text, block.raw)}</div>`;
        case 'fallback':
            return `<div style="color:#888;">${escapeHtml(block.text)}</div>`;
        default:
            throw new Error(`buildPopupHtml: unknown block type "${block.type}"`);
    }
}

export function buildPopupHtml({ title, subtitle, blocks = [], padding = 5, fontSize = 12 }) {
    return `<div style="font-family:sans-serif;font-size:${fontSize}px;color:#000;padding:${padding}px;">` +
        renderTitle(title) +
        renderSubtitle(subtitle) +
        blocks.map(renderBlock).join('') +
        `</div>`;
}
