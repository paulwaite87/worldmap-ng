// Regression guard for the two pieces of createCurrentParticleGLLayer's JS
// orchestration that could be pulled out of the closure as pure functions -- candidate
// #7 follow-up ("particle engine consolidation", architecture review candidate D).
// Everything else in the engine (GL resource lifecycle, mount/refresh/unmount,
// drawTrails/advect's bar-vs-streamline dispatch) still needs a real or heavily-mocked
// GL context and stays untested here; see tests/gl-shaders/ for the shader-level
// coverage of the GLSL these functions' outputs eventually feed.
import { describe, test, expect } from 'vitest';
import { computeParams, viewBox } from './_streamparticles_gl.js';

// A mapper bundle where every function just returns a fixed, easily-distinguished
// value -- computeParams's job is dispatch/fallback/clamp, not the mapper's own
// tuning (that's covered per-consumer in wind.test.js/waves.test.js/etc.), so the
// mappers themselves stay trivial here.
function fixedMappers(overrides = {}) {
    return {
        speedFromConfig: () => 0.1,
        thicknessFromConfig: () => 2.0,
        maxSpeedColor: () => 5.0,
        landReset: () => 1.0,
        hFromConfig: () => 8.0e-4,
        lenFromConfig: () => 7,
        calmDrop: () => 0.06,
        calmFade: () => 0.6,
        minValue: () => 0.0,
        coherenceRadius: () => 0,
        ageStep: () => 1 / 180,
        vmax: 2.5,
        ...overrides,
    };
}

describe('computeParams', () => {
    test('dispatches each mapper against cfg and returns its value', () => {
        const p = computeParams({}, fixedMappers(), 0);
        expect(p.curSpeed).toBe(0.1);
        expect(p.curThick).toBe(2.0);
        expect(p.curMaxSpeed).toBe(5.0);
        expect(p.curH).toBe(8.0e-4);
        expect(p.curHalfLen).toBe(7);
        expect(p.curCalmDrop).toBe(0.06);
        expect(p.curCalmFade).toBe(0.6);
        expect(p.curAgeStep).toBeCloseTo(1 / 180);
    });

    test('curAlpha reads cfg.particle_opacity directly (not via a mapper), falling back to 0.9', () => {
        expect(computeParams({ particle_opacity: 50 }, fixedMappers(), 0).curAlpha).toBeCloseTo(0.5);
        expect(computeParams({}, fixedMappers(), 0).curAlpha).toBeCloseTo(0.9);
        expect(computeParams({ particle_opacity: 0 }, fixedMappers(), 0).curAlpha).toBeCloseTo(0.9);
    });

    test('curMaxSpeed falls back to vmax when maxSpeedColor returns falsy', () => {
        const p = computeParams({}, fixedMappers({ maxSpeedColor: () => 0 }), 0);
        expect(p.curMaxSpeed).toBe(2.5); // vmax from fixedMappers()
    });

    test('curLandReset thresholds landReset(cfg) at 0.5', () => {
        expect(computeParams({}, fixedMappers({ landReset: () => 0.6 }), 0).curLandReset).toBe(1.0);
        expect(computeParams({}, fixedMappers({ landReset: () => 0.5 }), 0).curLandReset).toBe(0.0);
        expect(computeParams({}, fixedMappers({ landReset: () => 0 }), 0).curLandReset).toBe(0.0);
    });

    test('curMinValue coerces non-finite/falsy minValue output to 0', () => {
        expect(computeParams({}, fixedMappers({ minValue: () => NaN }), 0).curMinValue).toBe(0.0);
        expect(computeParams({}, fixedMappers({ minValue: () => 1.5 }), 0).curMinValue).toBe(1.5);
    });

    test('curAgeStep falls back to 1/90 when ageStep output is invalid', () => {
        expect(computeParams({}, fixedMappers({ ageStep: () => 0 }), 0).curAgeStep).toBeCloseTo(1 / 90);
        expect(computeParams({}, fixedMappers({ ageStep: () => -1 }), 0).curAgeStep).toBeCloseTo(1 / 90);
        expect(computeParams({}, fixedMappers({ ageStep: () => NaN }), 0).curAgeStep).toBeCloseTo(1 / 90);
    });

    test('curCohRadius always reflects the mapper output regardless of prevCohRadius', () => {
        expect(computeParams({}, fixedMappers({ coherenceRadius: () => 6 }), 0).curCohRadius).toBe(6);
        expect(computeParams({}, fixedMappers({ coherenceRadius: () => 6 }), 6).curCohRadius).toBe(6);
    });

    test('cohChanged is true only when coherenceRadius differs from prevCohRadius', () => {
        expect(computeParams({}, fixedMappers({ coherenceRadius: () => 6 }), 6).cohChanged).toBe(false);
        expect(computeParams({}, fixedMappers({ coherenceRadius: () => 6 }), 0).cohChanged).toBe(true);
        expect(computeParams({}, fixedMappers({ coherenceRadius: () => 0 }), 6).cohChanged).toBe(true);
    });

    test('coherenceRadius output falls back to 0 when invalid', () => {
        const p = computeParams({}, fixedMappers({ coherenceRadius: () => NaN }), 0);
        expect(p.curCohRadius).toBe(0);
        expect(p.cohChanged).toBe(false);
    });
});

function mockMap({ north, south, lng = 0, zoom = 3, clientWidth = 1024 } = {}) {
    return {
        getBounds: () => ({ getNorth: () => north, getSouth: () => south }),
        getCenter: () => ({ lng }),
        getCanvas: () => ({ clientWidth }),
        getZoom: () => zoom,
    };
}

describe('viewBox', () => {
    test('pads latitude and derives longitude span from viewport pixel width', () => {
        const [lonMin, yN, lonMax, yS] = viewBox(mockMap({ north: 10, south: -10, lng: 0, zoom: 3 }));
        // yN/yS are equirect-normalized (0=north pole, 1=south pole); padded outward from 10/-10.
        expect(yN).toBeLessThan((90 - 10) / 180);
        expect(yS).toBeGreaterThan((90 - -10) / 180);
        expect(lonMin).toBeLessThan(0.5);
        expect(lonMax).toBeGreaterThan(0.5);
    });

    test('enforces a minimum box height when getBounds() N/S span collapses to near-zero', () => {
        const [, yN, , yS] = viewBox(mockMap({ north: 5.0001, south: 5.0, zoom: 12 }));
        expect(yS - yN).toBeGreaterThanOrEqual(1e-6 - 1e-9);
    });

    test('shrinks the respawn box to track a normal viewport all the way through high zoom, instead of freezing around zoom 10', () => {
        // Regression for the "hardly any particles active past zoom ~10" bug: the old
        // floors (MIN_H=0.006 UV / spanLon>=1deg) were sized as a "reasonable minimum
        // useful size" rather than a true degenerate-input guard, so a normal ~1024px
        // viewport's genuinely-shrinking span got clamped back UP to that size well
        // before MapLibre's own ~22 zoom ceiling -- freezing the respawn box's size
        // while the actual visible viewport kept shrinking underneath it, so an
        // ever-growing fraction of respawns landed off-screen. getBounds()'s own N/S
        // report already shrinks correctly with zoom (mirrors real MapLibre behaviour;
        // unlike longitude, this function doesn't re-derive it from pixel width), so a
        // realistic tiny span here must NOT get pulled back up to the old ~1deg floor.
        const zoom10 = viewBox(mockMap({ north: 0.05, south: -0.05, lng: 0, zoom: 10 }));
        const zoom18 = viewBox(mockMap({ north: 0.0002, south: -0.0002, lng: 0, zoom: 18 }));
        const lonSpan = ([lonMin, , lonMax]) => lonMax - lonMin;
        const latSpan = ([, yN, , yS]) => yS - yN;
        expect(lonSpan(zoom18)).toBeLessThan(lonSpan(zoom10));
        expect(latSpan(zoom18)).toBeLessThan(latSpan(zoom10));
        // Both must be a tiny sliver of the globe, nowhere near the old ~1deg (~0.0028
        // of the 0..1 longitude fraction) floor that used to cap them.
        expect(lonSpan(zoom10)).toBeLessThan(0.0028);
        expect(lonSpan(zoom18)).toBeLessThan(0.0001);
    });

    test('falls back to the whole world when the derived longitude span is degenerate/huge', () => {
        // A very small worldPx (low zoom) blows spanLon past the 350 cutoff.
        const [lonMin, , lonMax] = viewBox(mockMap({ north: 10, south: -10, zoom: -3 }));
        expect(lonMin).toBe(0);
        expect(lonMax).toBe(1);
    });

    test('returns the whole-world fallback box when getBounds() yields non-finite N/S', () => {
        expect(viewBox(mockMap({ north: NaN, south: -10 }))).toEqual([0, 0, 1, 1]);
    });

    test('returns the whole-world fallback box if map access throws', () => {
        const throwingMap = { getBounds: () => { throw new Error('no style loaded'); } };
        expect(viewBox(throwingMap)).toEqual([0, 0, 1, 1]);
    });

    test('wraps longitude across the antimeridian when centered near +/-180', () => {
        const [lonMin, , lonMax] = viewBox(mockMap({ north: 10, south: -10, lng: 179.9, zoom: 5 }));
        expect(lonMin).toBeGreaterThan(0.9);
        expect(lonMax).toBeLessThan(0.1);
    });
});
