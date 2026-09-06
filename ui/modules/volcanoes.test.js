// Regression guard for a real bug (confirmed live): volcanoes.js imported
// `keyFilename`/`showLegend` from ./_legend.js, which stopped exporting either once
// legend keys moved fully client-side (issue #302) -- a leftover from before that
// refactor. A broken static import throws a SyntaxError at module-evaluation time in
// an ES module, which silently killed the WHOLE module (markers, popups, the smoke
// overlay -- everything), not just the legend: `enabled: true` in Settings, real data
// in the DB, and a working geojson/icon endpoints all did nothing, because
// loadLayer() itself never ran. This test simply importing the module (and its
// so2_volcanic key spec) would have caught this immediately.
import { describe, test, expect } from 'vitest';
import { loadLayer } from './volcanoes.js';
import { so2VolcanicKeySpec } from './air_quality.js';

describe('volcanoes.js module shape', () => {
    test('loadLayer imports successfully and is a function', () => {
        expect(typeof loadLayer).toBe('function');
    });
});

describe('so2VolcanicKeySpec', () => {
    test('returns a drawKey-ready spec using the given so2_min as vmin', () => {
        const spec = so2VolcanicKeySpec(2.5);
        expect(spec.vmin).toBe(2.5);
        expect(spec.vmax).toBe(20);
        expect(spec.title).toBe('SO2 (Volcanic) (DU)');
        expect(spec.lut).toBeInstanceOf(Uint8Array);
        expect(spec.lut.length).toBe(256 * 4);
        expect(spec.ticks).toHaveLength(5);
    });

    test('falls back to the default so2_min (1.0) when none is given', () => {
        const spec = so2VolcanicKeySpec(undefined);
        expect(spec.vmin).toBe(1.0);
    });
});
