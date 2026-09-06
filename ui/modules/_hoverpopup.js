// ui/modules/_hoverpopup.js
/**
 * Shared hover-popup wiring behind every popup-bearing layer, including markers.js
 * (architecture review candidate #6, "make popup functionality a one-stop-shop for
 * every instance", superseding docs/adr/0002-dont-extend-hoverpopup-for-markers.md).
 * Originally quakes.js/storms.js/volcanoes.js/satellites.js, which independently
 * rebuilt the same maplibregl.Popup construction, mouseenter/mouseleave cursor+
 * setLngLat+setHTML+addTo/remove dance, and map.on/off teardown; widened to also
 * cover flightradar.js/shipping.js's sticky-hover needs and, per the superseded ADR,
 * markers.js's multi-layer/mousemove/live-enable shape (see `layerId`/`event`/
 * `enabled` below). This owns that mechanics once; each caller supplies only its own
 * layerId(s) and an html(feature) -> string renderer, since the popup CONTENT is
 * genuinely bespoke per layer (different fields, different layout, see
 * buildPopupHtml in _feedhelpers.js for how content itself is now unified too).
 *
 * maxWidth is optional and omitted from the Popup options entirely when not given --
 * passing an explicit `undefined` through to `new maplibregl.Popup({..., maxWidth})`
 * would override MapLibre's own built-in default (240px) with `undefined` via
 * Object.assign's key-presence semantics, widening every caller's popup by accident.
 *
 * "Sticky" while hovered: the popup only closes once the mouse has left BOTH the
 * marker AND the popup's own DOM content, tracked as two independent booleans
 * (overMarker/overPopup) rather than a single flag -- otherwise leaving the marker
 * to move into the popup (Flight Radar's route/tooltip content is large enough to
 * need this) would still close it before the mouse ever reaches the popup. Every
 * caller gets this for free; simpler content (a one-line quake magnitude, say)
 * simply never has a reason to be entered, so the added listeners are inert there.
 *
 * closeDelayMs (default 200): leaving the marker doesn't remove the popup
 * immediately -- it's offset from the marker (see `offset`), so the cursor has to
 * cross a real gap of neither-hovered space to reach it, and an instant remove()
 * never gave it time to arrive. This is a grace period, not a fixed close delay:
 * it's cancelled the moment the mouse reaches the marker or the popup (see
 * cancelClose()), so a genuine move-away still closes promptly once the timer
 * fires. Long content needing to actually scroll (ui/index.html's
 * .maplibregl-popup-content max-height) made this gap-crossing failure visible,
 * but it applies to every caller uniformly, not just scrollable popups.
 *
 * `layerId` accepts a single id or an array -- markers.js binds across its dot AND
 * label layers to the same popup (architecture review candidate #6, superseding
 * docs/adr/0002-dont-extend-hoverpopup-for-markers.md: markers.js originally stayed
 * bespoke specifically because it needed this, plus the two options below, at once).
 *
 * `event` ("enter", the default, or "move") selects mouseenter/mouseleave vs
 * mousemove/mouseleave -- markers.js needs mousemove so the popup tracks
 * continuously and doesn't flicker crossing between its adjacent dot/label layers.
 *
 * `enabled` (optional) is checked live on every enter/move event, same as a
 * feature-less event -- lets a caller flip a setting (markers.js's weather_popup
 * toggle) without rebinding, instead of the on/off dance every other caller here
 * uses (bind once for the layer's whole lifetime).
 */
export function hoverPopup(map, layerId, {
    offset = 15, html, maxWidth, closeDelayMs = 200, event = 'enter', enabled,
}) {
    const popupOpts = { closeButton: false, closeOnClick: false, offset };
    if (maxWidth) popupOpts.maxWidth = maxWidth;
    const popup = new maplibregl.Popup(popupOpts);
    const layerIds = Array.isArray(layerId) ? layerId : [layerId];
    const enterEvent = event === 'move' ? 'mousemove' : 'mouseenter';

    let overMarker = false;
    let overPopup = false;
    let closeTimer = null;

    const cancelClose = () => {
        if (closeTimer) { clearTimeout(closeTimer); closeTimer = null; }
    };

    const closeIfNeitherHovered = () => {
        cancelClose();
        if (overMarker || overPopup) return;
        closeTimer = setTimeout(() => {
            closeTimer = null;
            if (overMarker || overPopup) return;
            map.getCanvas().style.cursor = '';
            popup.remove();
        }, closeDelayMs);
    };

    const onPopupEnter = () => { overPopup = true; cancelClose(); };
    const onPopupLeave = () => { overPopup = false; closeIfNeitherHovered(); };

    const onEnter = (e) => {
        if (enabled && !enabled()) return;
        if (!e.features.length) return;
        overMarker = true;
        cancelClose();
        map.getCanvas().style.cursor = 'pointer';
        // Point features anchor the popup to their own coordinate (stable even as the
        // mouse moves within a small marker's hit area); anything else (e.g. Troublespots'
        // polygons) has no single representative point, so anchor to where the mouse
        // actually is instead -- a Polygon's geometry.coordinates is a nested rings
        // array, not a [lon, lat] pair, and would hand setLngLat garbage. Checked by
        // shape (a flat pair's first entry is a number) rather than geometry.type, so
        // fixtures that only set coordinates still behave like real GeoJSON.
        const coordinates = e.features[0].geometry.coordinates;
        const coords = typeof coordinates[0] === 'number' ? coordinates.slice() : e.lngLat;
        popup.setLngLat(coords).setHTML(html(e.features[0])).addTo(map);
        // Only reachable once addTo() has actually built the DOM -- re-wired on
        // every open since remove() discards the previous element.
        const el = popup.getElement();
        if (el) {
            el.addEventListener('mouseenter', onPopupEnter);
            el.addEventListener('mouseleave', onPopupLeave);
        }
    };
    const onLeave = () => { overMarker = false; closeIfNeitherHovered(); };

    for (const id of layerIds) {
        map.on(enterEvent, id, onEnter);
        map.on('mouseleave', id, onLeave);
    }

    return () => {
        cancelClose();
        for (const id of layerIds) {
            map.off(enterEvent, id, onEnter);
            map.off('mouseleave', id, onLeave);
        }
        const el = popup.getElement();
        if (el) {
            el.removeEventListener('mouseenter', onPopupEnter);
            el.removeEventListener('mouseleave', onPopupLeave);
        }
        popup.remove();
    };
}
