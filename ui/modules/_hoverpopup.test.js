// Tests for the shared hover-popup wiring behind quakes.js/storms.js/volcanoes.js/
// satellites.js (architecture review candidate "a home for copy-pasted legend/
// hover-popup plumbing"). vitest runs in the default "node" environment, so
// maplibregl.Popup and the map object are faked minimally here.
import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';
import { hoverPopup } from './_hoverpopup.js';

function fakePopupElement() {
    const listeners = {};
    return {
        _listeners: listeners,
        addEventListener: vi.fn((evt, fn) => { listeners[evt] = fn; }),
        removeEventListener: vi.fn((evt, fn) => { if (listeners[evt] === fn) delete listeners[evt]; }),
    };
}

function fakePopup() {
    const p = { html: null, lngLat: null, onMap: false };
    const element = fakePopupElement();
    p.setLngLat = vi.fn((c) => { p.lngLat = c; return p; });
    p.setHTML = vi.fn((h) => { p.html = h; return p; });
    p.addTo = vi.fn(() => { p.onMap = true; return p; });
    p.remove = vi.fn(() => { p.onMap = false; return p; });
    p.getElement = vi.fn(() => element);
    return p;
}

function fakeMap() {
    const handlers = {};
    const canvas = { style: { cursor: '' } };
    return {
        _handlers: handlers,
        getCanvas: () => canvas,
        on: vi.fn((evt, layerId, fn) => { handlers[`${evt}:${layerId}`] = fn; }),
        off: vi.fn((evt, layerId, fn) => {
            if (handlers[`${evt}:${layerId}`] === fn) delete handlers[`${evt}:${layerId}`];
        }),
    };
}

beforeEach(() => {
    globalThis.maplibregl = { Popup: vi.fn(fakePopup) };
    vi.useFakeTimers();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('hoverPopup', () => {
    test('registers mouseenter/mouseleave on the given layer', () => {
        const map = fakeMap();
        hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });

        expect(map.on).toHaveBeenCalledWith('mouseenter', 'quakes-layer', expect.any(Function));
        expect(map.on).toHaveBeenCalledWith('mouseleave', 'quakes-layer', expect.any(Function));
    });

    test('mouseenter sets the cursor, positions the popup via html(feature), and adds it to the map', () => {
        const map = fakeMap();
        const html = vi.fn((f) => `<strong>${f.properties.name}</strong>`);
        hoverPopup(map, 'quakes-layer', { html });

        const feature = { properties: { name: 'M 4.2' }, geometry: { coordinates: [1, 2] } };
        map._handlers['mouseenter:quakes-layer']({ features: [feature] });

        expect(map.getCanvas().style.cursor).toBe('pointer');
        expect(html).toHaveBeenCalledWith(feature);
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;
        expect(popup.setLngLat).toHaveBeenCalledWith([1, 2]);
        expect(popup.setHTML).toHaveBeenCalledWith('<strong>M 4.2</strong>');
        expect(popup.addTo).toHaveBeenCalledWith(map);
    });

    test('mouseenter on a non-Point feature (e.g. a Troublespots polygon) anchors the popup to the mouse position, not the geometry', () => {
        const map = fakeMap();
        const html = vi.fn(() => '<div/>');
        hoverPopup(map, 'troublespots-fill', { html });

        const feature = {
            properties: { band: 'severe' },
            geometry: { type: 'Polygon', coordinates: [[[1, 2], [3, 4], [5, 6], [1, 2]]] },
        };
        map._handlers['mouseenter:troublespots-fill']({ features: [feature], lngLat: [9, 10] });

        const popup = globalThis.maplibregl.Popup.mock.results[0].value;
        expect(popup.setLngLat).toHaveBeenCalledWith([9, 10]);
        expect(popup.addTo).toHaveBeenCalledWith(map);
    });

    test('mouseenter with no features is a no-op', () => {
        const map = fakeMap();
        const html = vi.fn();
        hoverPopup(map, 'quakes-layer', { html });

        map._handlers['mouseenter:quakes-layer']({ features: [] });

        expect(html).not.toHaveBeenCalled();
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;
        expect(popup.addTo).not.toHaveBeenCalled();
    });

    test('mouseleave resets the cursor and removes the popup after the close delay', () => {
        const map = fakeMap();
        hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });

        map._handlers['mouseenter:quakes-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        map._handlers['mouseleave:quakes-layer']();
        vi.advanceTimersByTime(200);

        expect(map.getCanvas().style.cursor).toBe('');
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;
        expect(popup.remove).toHaveBeenCalled();
    });

    // ---- grace period: leaving the marker doesn't remove the popup instantly --
    // the cursor needs time to cross the offset gap into the popup itself (e.g. to
    // reach a scrollbar on tall content) -----------------------------------------

    test('mouseleave does not remove the popup immediately -- only after the close delay elapses', () => {
        const map = fakeMap();
        hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:quakes-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        map._handlers['mouseleave:quakes-layer']();

        expect(popup.remove).not.toHaveBeenCalled();
        vi.advanceTimersByTime(199);
        expect(popup.remove).not.toHaveBeenCalled();
        vi.advanceTimersByTime(1);
        expect(popup.remove).toHaveBeenCalled();
    });

    test('re-entering the marker within the close delay cancels the pending close', () => {
        const map = fakeMap();
        hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:quakes-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        map._handlers['mouseleave:quakes-layer']();
        vi.advanceTimersByTime(100);
        map._handlers['mouseenter:quakes-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        vi.advanceTimersByTime(200);

        expect(popup.remove).not.toHaveBeenCalled();
    });

    test('a custom closeDelayMs is honoured', () => {
        const map = fakeMap();
        hoverPopup(map, 'quakes-layer', { html: () => '<div/>', closeDelayMs: 500 });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:quakes-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        map._handlers['mouseleave:quakes-layer']();
        vi.advanceTimersByTime(200);

        expect(popup.remove).not.toHaveBeenCalled();
        vi.advanceTimersByTime(300);
        expect(popup.remove).toHaveBeenCalled();
    });

    // ---- "sticky" while hovered: stays open until the mouse has left BOTH the
    // marker and the popup's own DOM content -----------------------------------

    test('leaving the marker does NOT close the popup while the mouse is over the popup itself', () => {
        const map = fakeMap();
        hoverPopup(map, 'flightradar-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:flightradar-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        popup.getElement()._listeners.mouseenter();   // mouse moves onto the popup
        map._handlers['mouseleave:flightradar-layer'](); // ...then off the marker
        vi.advanceTimersByTime(200);

        expect(popup.remove).not.toHaveBeenCalled();
    });

    test('entering the popup within the close delay (after leaving the marker) cancels the pending close', () => {
        const map = fakeMap();
        hoverPopup(map, 'flightradar-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:flightradar-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        map._handlers['mouseleave:flightradar-layer']();  // gap-crossing moment
        vi.advanceTimersByTime(100);                       // still mid-flight
        popup.getElement()._listeners.mouseenter();         // ...arrives at the popup
        vi.advanceTimersByTime(200);

        expect(popup.remove).not.toHaveBeenCalled();
    });

    test('leaving the popup after leaving the marker finally closes it', () => {
        const map = fakeMap();
        hoverPopup(map, 'flightradar-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:flightradar-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        popup.getElement()._listeners.mouseenter();
        map._handlers['mouseleave:flightradar-layer']();
        popup.getElement()._listeners.mouseleave();   // mouse now leaves the popup too
        vi.advanceTimersByTime(200);

        expect(popup.remove).toHaveBeenCalled();
        expect(map.getCanvas().style.cursor).toBe('');
    });

    test('leaving the popup while still over the marker does not close it', () => {
        const map = fakeMap();
        hoverPopup(map, 'flightradar-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:flightradar-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        popup.getElement()._listeners.mouseenter();
        popup.getElement()._listeners.mouseleave();   // back onto the marker, never left it
        vi.advanceTimersByTime(200);

        expect(popup.remove).not.toHaveBeenCalled();
    });

    test('re-wires the popup element listeners on every re-open (a fresh element each addTo)', () => {
        const map = fakeMap();
        hoverPopup(map, 'flightradar-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:flightradar-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        expect(popup.getElement().addEventListener).toHaveBeenCalledWith('mouseenter', expect.any(Function));
        expect(popup.getElement().addEventListener).toHaveBeenCalledWith('mouseleave', expect.any(Function));
    });

    test('passes offset through to the Popup constructor, defaulting to 15', () => {
        const map = fakeMap();
        hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });
        expect(globalThis.maplibregl.Popup).toHaveBeenCalledWith(
            expect.objectContaining({ offset: 15 }));

        hoverPopup(map, 'storms-points', { offset: 10, html: () => '<div/>' });
        expect(globalThis.maplibregl.Popup).toHaveBeenLastCalledWith(
            expect.objectContaining({ offset: 10 }));
    });

    test('maxWidth is omitted from the Popup constructor call when not given', () => {
        const map = fakeMap();
        hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });

        const opts = globalThis.maplibregl.Popup.mock.calls[0][0];
        expect('maxWidth' in opts).toBe(false);
    });

    test('an explicit maxWidth is passed through to the Popup constructor', () => {
        const map = fakeMap();
        hoverPopup(map, 'storms-points', { html: () => '<div/>', maxWidth: '360px' });

        expect(globalThis.maplibregl.Popup).toHaveBeenCalledWith(
            expect.objectContaining({ maxWidth: '360px' }));
    });

    test('the returned stop() unregisters both handlers and removes the popup', () => {
        const map = fakeMap();
        const stop = hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        stop();

        expect(map._handlers['mouseenter:quakes-layer']).toBeUndefined();
        expect(map._handlers['mouseleave:quakes-layer']).toBeUndefined();
        expect(popup.remove).toHaveBeenCalled();
    });

    test('stop() also cleans up the popup element listeners it wired on open', () => {
        const map = fakeMap();
        const stop = hoverPopup(map, 'flightradar-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:flightradar-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        stop();

        expect(popup.getElement().removeEventListener).toHaveBeenCalledWith('mouseenter', expect.any(Function));
        expect(popup.getElement().removeEventListener).toHaveBeenCalledWith('mouseleave', expect.any(Function));
    });

    test('stop() cancels a pending close timer rather than letting it fire later', () => {
        const map = fakeMap();
        const stop = hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });
        const popup = globalThis.maplibregl.Popup.mock.results[0].value;

        map._handlers['mouseenter:quakes-layer']({
            features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
        });
        map._handlers['mouseleave:quakes-layer'](); // schedules a delayed close
        stop();                                      // torn down before it fires
        popup.remove.mockClear();

        vi.advanceTimersByTime(200);

        expect(popup.remove).not.toHaveBeenCalled();
    });

    // ---- markers.js parity (architecture review candidate #6 -- unify ALL popup
    // consumers, superseding ADR-0002): multi-layer binding, a configurable
    // event pair, and a live enabled predicate --------------------------------

    describe('multi-layer binding (layerId as an array)', () => {
        test('registers the enter/leave pair on every id in the array', () => {
            const map = fakeMap();
            hoverPopup(map, ['markers-dots', 'markers-labels'], { html: () => '<div/>' });

            expect(map.on).toHaveBeenCalledWith('mouseenter', 'markers-dots', expect.any(Function));
            expect(map.on).toHaveBeenCalledWith('mouseleave', 'markers-dots', expect.any(Function));
            expect(map.on).toHaveBeenCalledWith('mouseenter', 'markers-labels', expect.any(Function));
            expect(map.on).toHaveBeenCalledWith('mouseleave', 'markers-labels', expect.any(Function));
        });

        test('entering via either layer in the array opens the same popup', () => {
            const map = fakeMap();
            hoverPopup(map, ['markers-dots', 'markers-labels'], { html: () => '<div/>' });
            const popup = globalThis.maplibregl.Popup.mock.results[0].value;

            map._handlers['mouseenter:markers-labels']({
                features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
            });

            expect(popup.addTo).toHaveBeenCalledWith(map);
        });

        test('stop() unbinds every id in the array', () => {
            const map = fakeMap();
            const stop = hoverPopup(map, ['markers-dots', 'markers-labels'], { html: () => '<div/>' });

            stop();

            expect(map._handlers['mouseenter:markers-dots']).toBeUndefined();
            expect(map._handlers['mouseenter:markers-labels']).toBeUndefined();
            expect(map._handlers['mouseleave:markers-dots']).toBeUndefined();
            expect(map._handlers['mouseleave:markers-labels']).toBeUndefined();
        });

        test('a bare string layerId still works exactly as before (regression guard)', () => {
            const map = fakeMap();
            hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });

            expect(map.on).toHaveBeenCalledWith('mouseenter', 'quakes-layer', expect.any(Function));
        });
    });

    describe('configurable event pair (event: "enter" | "move")', () => {
        test('defaults to mouseenter/mouseleave when event is omitted (regression guard)', () => {
            const map = fakeMap();
            hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });

            expect(map.on).toHaveBeenCalledWith('mouseenter', 'quakes-layer', expect.any(Function));
            expect(map.on).not.toHaveBeenCalledWith('mousemove', 'quakes-layer', expect.any(Function));
        });

        test('event: "move" registers mousemove/mouseleave instead of mouseenter/mouseleave', () => {
            const map = fakeMap();
            hoverPopup(map, 'markers-dots', { html: () => '<div/>', event: 'move' });

            expect(map.on).toHaveBeenCalledWith('mousemove', 'markers-dots', expect.any(Function));
            expect(map.on).toHaveBeenCalledWith('mouseleave', 'markers-dots', expect.any(Function));
            expect(map.on).not.toHaveBeenCalledWith('mouseenter', 'markers-dots', expect.any(Function));
        });

        test('mousemove positions and shows the popup the same way mouseenter does', () => {
            const map = fakeMap();
            const html = vi.fn(() => '<div/>');
            hoverPopup(map, 'markers-dots', { html, event: 'move' });
            const popup = globalThis.maplibregl.Popup.mock.results[0].value;

            map._handlers['mousemove:markers-dots']({
                features: [{ properties: {}, geometry: { coordinates: [5, 6] } }],
            });

            expect(popup.setLngLat).toHaveBeenCalledWith([5, 6]);
            expect(popup.addTo).toHaveBeenCalledWith(map);
        });
    });

    describe('live enabled predicate', () => {
        test('entering does nothing while enabled() returns false', () => {
            const map = fakeMap();
            hoverPopup(map, 'markers-dots', { html: () => '<div/>', enabled: () => false });
            const popup = globalThis.maplibregl.Popup.mock.results[0].value;

            map._handlers['mouseenter:markers-dots']({
                features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
            });

            expect(popup.addTo).not.toHaveBeenCalled();
            expect(map.getCanvas().style.cursor).toBe('');
        });

        test('entering opens the popup normally once enabled() returns true', () => {
            const map = fakeMap();
            let live = false;
            hoverPopup(map, 'markers-dots', { html: () => '<div/>', enabled: () => live });
            const popup = globalThis.maplibregl.Popup.mock.results[0].value;

            live = true;
            map._handlers['mouseenter:markers-dots']({
                features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
            });

            expect(popup.addTo).toHaveBeenCalledWith(map);
        });

        test('omitting enabled entirely behaves as always-enabled (regression guard)', () => {
            const map = fakeMap();
            hoverPopup(map, 'quakes-layer', { html: () => '<div/>' });
            const popup = globalThis.maplibregl.Popup.mock.results[0].value;

            map._handlers['mouseenter:quakes-layer']({
                features: [{ properties: {}, geometry: { coordinates: [0, 0] } }],
            });

            expect(popup.addTo).toHaveBeenCalledWith(map);
        });
    });
});
