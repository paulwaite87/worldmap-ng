// Tests for flightradar.js's pure dead-reckoning + freeze-clamp helpers (issue #203,
// docs/adr/0009). Expected values are worked by hand from first principles (60 knots
// for 1 hour = 60 nautical miles = exactly 1 degree of latitude), not recomputed the
// way the code does, so a broken formula can actually disagree with the test.
import { describe, test, expect } from 'vitest';
import { interpolatedPosition, smoothedPosition, smoothedScalar, smoothedAngle, isBackwardCorrection, bearingDeg, recordFromFeature, extrapolatedAltitude, boundedElapsedSeconds, isFrozen, flightStatus, targetAltitudeLabel, aircraftClass, aircraftGroup, aircraftGroupColor, airlineForFlight, stopCode, routePathHtml, plausibleWarningHtml, parseRouteStops, deriveDisplayState, buildFeatureCollection } from './flightradar.js';

describe('interpolatedPosition', () => {
    test('due-north flight for 1 hour at 60kts moves exactly 1 degree of latitude', () => {
        const pos = interpolatedPosition({ lat: 10, lon: 20, gs: 60, track: 0 }, 3600);
        expect(pos.lat).toBeCloseTo(11.0, 6);
        expect(pos.lon).toBeCloseTo(20.0, 6);
    });

    test('due-south flight moves latitude negative', () => {
        const pos = interpolatedPosition({ lat: 10, lon: 20, gs: 60, track: 180 }, 3600);
        expect(pos.lat).toBeCloseTo(9.0, 6);
        expect(pos.lon).toBeCloseTo(20.0, 6);
    });

    test('due-east flight at the equator moves exactly 1 degree of longitude', () => {
        const pos = interpolatedPosition({ lat: 0, lon: 20, gs: 60, track: 90 }, 3600);
        expect(pos.lat).toBeCloseTo(0.0, 6);
        expect(pos.lon).toBeCloseTo(21.0, 6);
    });

    test('due-east flight at 60deg latitude moves 2 degrees of longitude (convergence)', () => {
        const pos = interpolatedPosition({ lat: 60, lon: 20, gs: 60, track: 90 }, 3600);
        expect(pos.lat).toBeCloseTo(60.0, 6);
        expect(pos.lon).toBeCloseTo(22.0, 6);
    });

    test('zero elapsed time means no movement', () => {
        const pos = interpolatedPosition({ lat: 10, lon: 20, gs: 400, track: 45 }, 0);
        expect(pos).toEqual({ lat: 10, lon: 20 });
    });

    test('zero ground speed means no movement', () => {
        const pos = interpolatedPosition({ lat: 10, lon: 20, gs: 0, track: 45 }, 100);
        expect(pos).toEqual({ lat: 10, lon: 20 });
    });

    test('missing track (no heading data) means no movement', () => {
        const pos = interpolatedPosition({ lat: 10, lon: 20, gs: 400, track: undefined }, 100);
        expect(pos).toEqual({ lat: 10, lon: 20 });
    });
});

describe('smoothedPosition', () => {
    test('zero elapsed time leaves the displayed position unchanged', () => {
        const display = { lat: 5, lon: 5 };
        expect(smoothedPosition(display, { lat: 10, lon: 10 }, 0, 0.6)).toEqual(display);
    });

    test('negative elapsed time (defensive; should not occur) leaves the position unchanged', () => {
        const display = { lat: 5, lon: 5 };
        expect(smoothedPosition(display, { lat: 10, lon: 10 }, -1, 0.6)).toEqual(display);
    });

    test('a small correction (well under the speed cap) eases by the exponential-smoothing fraction for the given dt/tau', () => {
        // A tiny (~0.6m) target gap keeps the whole step far under MAX_CORRECTION_KTS's
        // per-frame budget, exercising the plain exponential formula:
        // alpha = 1 - exp(-dtS/tauS) = 1 - exp(-1) = 0.6321206.
        const pos = smoothedPosition({ lat: 0, lon: 0 }, { lat: 0.00001, lon: 0 }, 0.6, 0.6);
        expect(pos.lat).toBeCloseTo(0.00001 * 0.6321206, 9);
        expect(pos.lon).toBeCloseTo(0.0, 9);
    });

    test('a much longer elapsed time than tau converges close to a small, uncapped target', () => {
        const pos = smoothedPosition({ lat: 0, lon: 0 }, { lat: 0.00001, lon: 0 }, 6.0, 0.6);
        expect(pos.lat).toBeCloseTo(0.00001, 8);
    });

    test('never overshoots past a small, uncapped target', () => {
        const pos = smoothedPosition({ lat: 0, lon: 0 }, { lat: 0.00001, lon: 0 }, 100.0, 0.6);
        expect(pos.lat).toBeLessThanOrEqual(0.00001);
    });

    // The actual bug this closes (caught live on final approach into a busy airport):
    // a landing aircraft's real deceleration outruns constant-velocity dead reckoning
    // between samples badly enough that the resulting position error can be
    // kilometers, not meters. Plain exponential smoothing closes that just as fast in
    // wall-clock time as a tiny error (same alpha), which reads as the icon sliding at
    // an obviously-impossible speed. These prove the per-frame distance cap actually
    // engages for a large error instead.
    test('a large correction is capped to MAX_CORRECTION_KTS worth of distance for this frame, not snapped in one exponential step', () => {
        // 10 degrees of latitude (~600nm) away -- pure exponential (alpha=0.6321206)
        // would close ~379nm in this single 0.6s step. The 800kt cap instead limits
        // real movement to 800kt * (0.6/3600)h = 0.13333nm = 0.0022222deg.
        const pos = smoothedPosition({ lat: 0, lon: 0 }, { lat: 10, lon: 0 }, 0.6, 0.6);
        expect(pos.lat).toBeCloseTo(0.0022222, 6);
    });

    test('a capped correction still moves toward the target, never past it', () => {
        const pos = smoothedPosition({ lat: 0, lon: 0 }, { lat: 10, lon: 0 }, 0.6, 0.6);
        expect(pos.lat).toBeGreaterThan(0);
        expect(pos.lat).toBeLessThan(10);
    });
});

describe('isBackwardCorrection', () => {
    test('no track reference means never backward (nothing to project against)', () => {
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: -1, lon: 0 }, null)).toBe(false);
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: -1, lon: 0 }, undefined)).toBe(false);
    });

    test('a target ahead along a due-north track is not backward', () => {
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: 0.01, lon: 0 }, 0)).toBe(false);
    });

    // The actual bug this closes: a landing aircraft's real position, once a new real
    // sample lands, is consistently BEHIND where constant-velocity extrapolation had
    // already advanced the icon to (see interpolatedPosition's docstring) -- this must
    // be detected so buildFeatureCollection can hold position instead of visibly
    // sliding the icon backward.
    test('a target behind along a due-north track is backward', () => {
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: -0.01, lon: 0 }, 0)).toBe(true);
    });

    test('a target directly to the side (perpendicular to track) is not backward', () => {
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: 0, lon: 0.01 }, 0)).toBe(false);
    });

    test('works for a non-north track too (due-east heading)', () => {
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: 0, lon: 0.01 }, 90)).toBe(false);
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: 0, lon: -0.01 }, 90)).toBe(true);
    });

    test('the exact same position as display is not backward (zero progress, boundary case)', () => {
        expect(isBackwardCorrection({ lat: 0, lon: 0 }, { lat: 0, lon: 0 }, 0)).toBe(false);
    });
});

describe('bearingDeg', () => {
    test('due north is 0', () => {
        expect(bearingDeg({ lat: 0, lon: 0 }, { lat: 1, lon: 0 })).toBeCloseTo(0, 6);
    });

    test('due east is 90', () => {
        expect(bearingDeg({ lat: 0, lon: 0 }, { lat: 0, lon: 1 })).toBeCloseTo(90, 6);
    });

    test('due south is 180', () => {
        expect(bearingDeg({ lat: 0, lon: 0 }, { lat: -1, lon: 0 })).toBeCloseTo(180, 6);
    });

    test('due west is 270, not negative (normalized to 0-360)', () => {
        expect(bearingDeg({ lat: 0, lon: 0 }, { lat: 0, lon: -1 })).toBeCloseTo(270, 6);
    });

    test('a diagonal move applies the same longitude-convergence correction interpolatedPosition uses', () => {
        // At 60deg latitude, 1deg of longitude covers half the ground distance 1deg of
        // latitude does (cos(60)=0.5) -- dLat=1, dLon-corrected=1*cos(60)=0.5 ->
        // atan2(0.5, 1) = 26.565 degrees, not the naive atan2(1,1)=45.
        expect(bearingDeg({ lat: 60, lon: 0 }, { lat: 61, lon: 1 })).toBeCloseTo(26.565, 2);
    });

    test('coincident points (no movement) return null rather than an arbitrary angle', () => {
        expect(bearingDeg({ lat: 10, lon: 20 }, { lat: 10, lon: 20 })).toBeNull();
    });

    test('is the inverse of interpolatedPosition -- round-trips a track through position and back', () => {
        const from = { lat: 10, lon: 20 };
        const to = interpolatedPosition({ lat: from.lat, lon: from.lon, gs: 300, track: 137 }, 600);
        expect(bearingDeg(from, to)).toBeCloseTo(137, 6);
    });
});

// The actual root cause diagnosed live: real ADS-B position (lat/lon + altitude) and
// velocity (gs/track/baro_rate) are broadcast as INDEPENDENT message types. Close to
// the ground -- final approach, rollout, taxi -- position updates can stall for
// several consecutive real samples (identical lat/lon/alt_baro_ft across 3-5 samples,
// 40+ seconds, confirmed against the real deployment's database) while velocity keeps
// reporting normally throughout. Dead reckoning from a stale position anchor using a
// genuinely-current speed/rate double-counts distance/altitude the aircraft already
// covered during the stall -- this is why landing traffic overshot even though real
// speed barely changes ("dead reckoning should be ~100% accurate here").
describe('recordFromFeature', () => {
    function featureFor({ lat = -41.3, lon = 174.8, ...propOverrides } = {}) {
        return {
            properties: {
                hex: 'a1b2c3', flight: 'ANZ423', registration: 'ZK-TST', aircraft_type: 'B738',
                category: 'A3', alt_baro_ft: 5000, on_ground: false, gs: 200, track: 90,
                baro_rate: 0, nav_altitude_mcp: null,
                last_seen: '2026-07-29T00:00:10.000000+00:00',
                route_stops: null, route_plausible: null,
                ...propOverrides,
            },
            geometry: { coordinates: [lon, lat] },
        };
    }

    test('first sighting (no prior record) uses the raw gs/baro_rate directly -- nothing to compare against yet', () => {
        const rec = recordFromFeature(featureFor(), 1000);
        expect(rec.deadReckonGs).toBe(200);
        expect(rec.deadReckonBaroRate).toBe(0);
    });

    test('the same real sample re-fetched (identical last_seen) is never treated as a stall, even with identical position', () => {
        const feature = featureFor({ lat: 10, lon: 20 });
        const prevRec = recordFromFeature(feature, 1000);
        const rec = recordFromFeature(feature, 2000, prevRec);   // same last_seen -- just a faster poll
        expect(rec.deadReckonGs).toBe(200);
    });

    test('a genuinely new real sample with an unchanged position suppresses dead-reckoning speed', () => {
        const prevRec = recordFromFeature(
            featureFor({ lat: 10, lon: 20, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ lat: 10, lon: 20, gs: 210, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        expect(rec.deadReckonGs).toBe(0);
        // The real reported speed is still preserved for the popup/informational display.
        expect(rec.gs).toBe(210);
    });

    test('a genuinely new real sample with a changed position resumes normal dead reckoning', () => {
        const prevRec = recordFromFeature(
            featureFor({ lat: 10, lon: 20, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ lat: 10.01, lon: 20, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        expect(rec.deadReckonGs).toBe(200);
    });

    test('a stalled altitude suppresses the baro_rate projection the same way', () => {
        const prevRec = recordFromFeature(
            featureFor({ alt_baro_ft: 1000, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ alt_baro_ft: 1000, baro_rate: -500, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        expect(rec.deadReckonBaroRate).toBe(0);
        expect(rec.baro_rate).toBe(-500);   // real reported rate preserved for the popup
    });

    test('a changed altitude resumes normal baro_rate extrapolation', () => {
        const prevRec = recordFromFeature(
            featureFor({ alt_baro_ft: 1000, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ alt_baro_ft: 900, baro_rate: -500, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        expect(rec.deadReckonBaroRate).toBe(-500);
    });

    test('on-ground (non-numeric altitude) is never treated as an altitude stall', () => {
        const prevRec = recordFromFeature(
            featureFor({ on_ground: true, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ on_ground: true, baro_rate: 5, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        expect(rec.deadReckonBaroRate).toBe(5);
        expect(rec.alt_baro).toBe('ground');
    });

    // flightStatus's fallback vertical rate: adsb.lol's baro_rate can drop out for a
    // poll or two (observed live, especially on approach/climb-out); derivedBaroRateFpm
    // is the change in alt_baro across the last two DISTINCT real samples, divided by
    // the real elapsed time between them.
    test('derives a fallback vertical rate from consecutive real altitude samples', () => {
        const prevRec = recordFromFeature(
            featureFor({ alt_baro_ft: 5000, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ alt_baro_ft: 4700, baro_rate: null, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        // 300ft dropped over 12s (0.2min) = -1500ft/min.
        expect(rec.derivedBaroRateFpm).toBeCloseTo(-1500, 6);
    });

    test('first sighting has no prior sample to derive a rate from', () => {
        const rec = recordFromFeature(featureFor(), 1000);
        expect(rec.derivedBaroRateFpm).toBeNull();
    });

    test('a repeated poll of the same real sample carries the last derived rate forward rather than resetting to null', () => {
        const prevRec = recordFromFeature(
            featureFor({ alt_baro_ft: 5000, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ alt_baro_ft: 4700, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        const recAgain = recordFromFeature(
            featureFor({ alt_baro_ft: 4700, last_seen: '2026-07-29T00:00:22.000000+00:00' }),   // same sample, re-fetched
            3000, rec,
        );
        expect(recAgain.derivedBaroRateFpm).toBe(rec.derivedBaroRateFpm);
    });

    test('on-ground (non-numeric altitude) never derives a vertical rate', () => {
        const prevRec = recordFromFeature(
            featureFor({ on_ground: true, last_seen: '2026-07-29T00:00:10.000000+00:00' }), 1000,
        );
        const rec = recordFromFeature(
            featureFor({ on_ground: true, last_seen: '2026-07-29T00:00:22.000000+00:00' }),
            2000, prevRec,
        );
        expect(rec.derivedBaroRateFpm).toBeNull();
    });
});

describe('smoothedScalar', () => {
    test('zero elapsed time leaves the displayed value unchanged', () => {
        expect(smoothedScalar(5, 10, 0, 0.6, 100)).toBe(5);
    });

    test('negative elapsed time (defensive; should not occur) leaves the value unchanged', () => {
        expect(smoothedScalar(5, 10, -1, 0.6, 100)).toBe(5);
    });

    test('a small correction (well under the rate cap) eases by the exponential-smoothing fraction', () => {
        // alpha = 1 - exp(-1) = 0.6321206; 10ft gap needs nowhere near a 100ft/s cap.
        expect(smoothedScalar(10000, 10010, 0.6, 0.6, 100)).toBeCloseTo(10000 + 10 * 0.6321206, 4);
    });

    // Mirrors smoothedPosition's cap tests -- extrapolatedAltitude's linear baro_rate
    // projection has the identical overshoot failure mode as interpolatedPosition's
    // constant-velocity guess, for the same landing-approach reason.
    test('a large correction is capped to maxRatePerSecond*dtS worth of change for this frame', () => {
        // 5000ft away; uncapped alpha would close ~3160ft in one 0.6s step; a 100ft/s
        // cap instead limits real movement to 100*0.6=60ft.
        expect(smoothedScalar(10000, 15000, 0.6, 0.6, 100)).toBeCloseTo(10060, 6);
    });

    test('capping applies symmetrically to a decreasing (descending) correction', () => {
        expect(smoothedScalar(15000, 10000, 0.6, 0.6, 100)).toBeCloseTo(14940, 6);
    });

    test('a capped correction still moves toward the target, never past it', () => {
        const value = smoothedScalar(10000, 15000, 0.6, 0.6, 100);
        expect(value).toBeGreaterThan(10000);
        expect(value).toBeLessThan(15000);
    });
});

describe('smoothedAngle', () => {
    test('zero elapsed time leaves the displayed value unchanged', () => {
        expect(smoothedAngle(10, 20, 0, 0.6, 10)).toBe(10);
    });

    test('negative elapsed time (defensive; should not occur) leaves the value unchanged', () => {
        expect(smoothedAngle(10, 20, -1, 0.6, 10)).toBe(10);
    });

    test('no prior display (first known track) returns the target directly, unsmoothed', () => {
        expect(smoothedAngle(null, 137, 0.6, 0.6, 10)).toBe(137);
    });

    test('no target (defensive; should not occur) returns the display unchanged', () => {
        expect(smoothedAngle(90, null, 0.6, 0.6, 10)).toBe(90);
    });

    test('a small correction (well under the rate cap) eases by the exponential-smoothing fraction', () => {
        // alpha = 1 - exp(-1) = 0.6321206; a 5deg gap needs nowhere near a 10deg/s*0.6s=6deg cap.
        expect(smoothedAngle(10, 15, 0.6, 0.6, 10)).toBeCloseTo(10 + 5 * 0.6321206, 4);
    });

    test('a large correction is capped to maxRateDegPerSecond*dtS worth of change for this frame', () => {
        // 90deg away; uncapped alpha would close ~56.9deg in one 0.6s step; a
        // 10deg/s cap instead limits real movement to 10*0.6=6deg.
        expect(smoothedAngle(0, 90, 0.6, 0.6, 10)).toBeCloseTo(6, 6);
    });

    test('capping applies symmetrically to a decreasing correction', () => {
        expect(smoothedAngle(90, 0, 0.6, 0.6, 10)).toBeCloseTo(84, 6);
    });

    // The actual bug this closes: bearingDeg-derived rotation can briefly point far
    // from the true heading right after a real update lands (see
    // MAX_ICON_TURN_RATE_DEG_S's own comment) -- without wraparound-aware easing, a
    // correction crossing north (e.g. 350 -> 10, actually a 20deg turn) would instead
    // be read as a 340deg gap and eased the LONG way round through 180.
    test('eases through the north wraparound the short way, not the long way round through 180', () => {
        const value = smoothedAngle(350, 10, 0.6, 0.6, 100);   // generous cap -- isolates the wraparound math itself
        // Moving from 350 toward 10 the short way passes through 360/0, landing
        // somewhere in (350, 360] union [0, 10) -- never near 180.
        const distanceFrom0 = Math.min(value, 360 - value);
        expect(distanceFrom0).toBeLessThan(10);
    });

    test('the wraparound case still respects the rate cap', () => {
        const value = smoothedAngle(350, 10, 0.6, 0.6, 10);   // 10deg/s*0.6s = 6deg max
        // 350 + 6 = 356 -- short way round, capped.
        expect(value).toBeCloseTo(356, 6);
    });
});

describe('extrapolatedAltitude', () => {
    test('climbing at 1000ft/min for 30s gains 500ft', () => {
        expect(extrapolatedAltitude(10000, 1000, null, 30)).toBeCloseTo(10500, 6);
    });

    test('descending at 1000ft/min for 30s loses 500ft', () => {
        expect(extrapolatedAltitude(10000, -1000, null, 30)).toBeCloseTo(9500, 6);
    });

    test('zero elapsed time means no altitude change', () => {
        expect(extrapolatedAltitude(10000, 1000, null, 0)).toBe(10000);
    });

    test('no baro_rate reading means no altitude change', () => {
        expect(extrapolatedAltitude(10000, null, null, 30)).toBe(10000);
    });

    test('altitude not a number (on ground / unknown) passes through unchanged', () => {
        expect(extrapolatedAltitude('ground', 1000, null, 30)).toBe('ground');
        expect(extrapolatedAltitude(undefined, 1000, null, 30)).toBeUndefined();
    });

    test('climbing toward a higher target clamps at the target instead of overshooting', () => {
        // Unclamped this would project to 10600; the MCP target is only 300ft away.
        expect(extrapolatedAltitude(10000, 1200, 10300, 30)).toBe(10300);
    });

    test('descending toward a lower target clamps at the target instead of undershooting', () => {
        expect(extrapolatedAltitude(10000, -1200, 9700, 30)).toBe(9700);
    });

    test('climbing well short of a higher target is not clamped', () => {
        expect(extrapolatedAltitude(10000, 1000, 20000, 30)).toBeCloseTo(10500, 6);
    });

    test('a stale target behind the direction of travel is not clamped', () => {
        // Climbing, but the MCP target is below current altitude (contradictory/stale) --
        // don't snap the projection back down to it.
        expect(extrapolatedAltitude(10000, 1000, 9000, 30)).toBeCloseTo(10500, 6);
    });

    test('no target given means the projection is unclamped', () => {
        expect(extrapolatedAltitude(10000, 1000, undefined, 60)).toBeCloseTo(11000, 6);
    });
});

describe('boundedElapsedSeconds', () => {
    test('returns real elapsed seconds when under the cap', () => {
        expect(boundedElapsedSeconds(0, 3000, 10)).toBeCloseTo(3.0, 6);
    });

    test('clamps to the cap once elapsed time exceeds it', () => {
        expect(boundedElapsedSeconds(0, 60000, 10)).toBe(10);
    });

    test('never returns negative (a lastSeen in the future, e.g. clock skew)', () => {
        expect(boundedElapsedSeconds(5000, 0, 10)).toBe(0);
    });

    test('defaults the cap to MAX_EXTRAPOLATION_S when omitted', () => {
        expect(boundedElapsedSeconds(0, 3000)).toBeCloseTo(3.0, 6);
    });
});

describe('isFrozen', () => {
    test('not frozen while under the cap', () => {
        expect(isFrozen(0, 3000, 10)).toBe(false);
    });

    test('frozen once elapsed time reaches the cap', () => {
        expect(isFrozen(0, 10000, 10)).toBe(true);
    });

    test('frozen once elapsed time exceeds the cap', () => {
        expect(isFrozen(0, 60000, 10)).toBe(true);
    });

    test('defaults the cap to MAX_EXTRAPOLATION_S when omitted', () => {
        expect(isFrozen(0, 3000)).toBe(false);
        expect(isFrozen(0, 60000)).toBe(true);
    });
});

describe('stopCode', () => {
    test('prefers IATA over ICAO', () => {
        expect(stopCode({ iata: 'WLG', icao: 'NZWN' })).toBe('WLG');
    });

    test('falls back to ICAO when IATA is missing', () => {
        expect(stopCode({ iata: null, icao: 'NZWN' })).toBe('NZWN');
    });

    test('falls back to an empty string when both are missing', () => {
        expect(stopCode({})).toBe('');
    });
});

describe('routePathHtml', () => {
    test('returns an empty string for a missing or empty stop list', () => {
        expect(routePathHtml(null)).toBe('');
        expect(routePathHtml(undefined)).toBe('');
        expect(routePathHtml([])).toBe('');
    });

    test('joins two stops with an arrow', () => {
        const html = routePathHtml([
            { iata: 'WLG', icao: 'NZWN', name: 'Wellington International Airport' },
            { iata: 'AKL', icao: 'NZAA', name: 'Auckland International Airport' },
        ]);
        expect(html).toContain('WLG');
        expect(html).toContain('&rarr;');
        expect(html).toContain('AKL');
        expect(html.indexOf('WLG')).toBeLessThan(html.indexOf('AKL'));
    });

    test('preserves the full path including an intermediate technical stop, in order', () => {
        const html = routePathHtml([
            { iata: 'WLG', icao: 'NZWN', name: 'Wellington' },
            { iata: 'CHC', icao: 'NZCH', name: 'Christchurch' },
            { iata: 'AKL', icao: 'NZAA', name: 'Auckland' },
        ]);
        expect(html.indexOf('WLG')).toBeLessThan(html.indexOf('CHC'));
        expect(html.indexOf('CHC')).toBeLessThan(html.indexOf('AKL'));
    });

    test('carries each stop\'s full name as a title tooltip', () => {
        const html = routePathHtml([{ iata: 'WLG', icao: 'NZWN', name: 'Wellington International Airport' }]);
        expect(html).toContain('title="Wellington International Airport"');
    });

    test('escapes a double quote in an airport name so it cannot break the title attribute', () => {
        const html = routePathHtml([{ iata: 'WLG', name: 'Wellington "Intl"' }]);
        expect(html).toContain('Wellington &quot;Intl&quot;');
        expect(html).not.toContain('"Wellington "Intl""');
    });

    test('falls back to ICAO for a stop with no IATA code', () => {
        const html = routePathHtml([{ iata: null, icao: 'NZWN', name: 'Wellington' }]);
        expect(html).toContain('NZWN');
    });
});

describe('plausibleWarningHtml', () => {
    test('shows a warning only when plausible is exactly false', () => {
        expect(plausibleWarningHtml(false)).toContain('&#9888;');
    });

    test('shows nothing when the route is plausible', () => {
        expect(plausibleWarningHtml(true)).toBe('');
    });

    test('shows nothing when there is no route to judge', () => {
        expect(plausibleWarningHtml(null)).toBe('');
        expect(plausibleWarningHtml(undefined)).toBe('');
    });
});

describe('parseRouteStops', () => {
    test('round-trips a JSON-stringified stop list back into an array', () => {
        const stops = [{ iata: 'WLG', icao: 'NZWN', name: 'Wellington' }];
        expect(parseRouteStops(JSON.stringify(stops))).toEqual(stops);
    });

    test('returns null for a missing/empty value', () => {
        expect(parseRouteStops(null)).toBeNull();
        expect(parseRouteStops(undefined)).toBeNull();
        expect(parseRouteStops('')).toBeNull();
    });

    test('returns null (not a throw) for malformed JSON', () => {
        expect(parseRouteStops('{not valid json')).toBeNull();
    });
});

// Regression guard (the actual live bug this feature shipped with): MapLibre tiles
// every GeoJSON source internally for rendering, and that pipeline doesn't safely
// round-trip non-primitive property values -- a bare array/object in route_stops
// broke hover/mouseenter feature-querying for the WHOLE flightradar layer, not just
// aircraft carrying a route. Every property buildFeatureCollection hands to
// setData() must stay a primitive (string/number/boolean/null).
describe('buildFeatureCollection route_stops shape', () => {
    function recFor(overrides = {}) {
        return {
            hex: 'a1b2c3', flight: 'ANZ423', category: '', receivedAt: 1000,
            lat: -41.3, lon: 174.8, gs: 200, track: 0, alt_baro: 5000,
            ...overrides,
        };
    }

    test('a matched route is JSON-stringified, never a bare array, in the built feature properties', () => {
        const stops = [
            { iata: 'WLG', icao: 'NZWN', name: 'Wellington' },
            { iata: 'AKL', icao: 'NZAA', name: 'Auckland' },
        ];
        const aircraftByHex = new Map([['a1b2c3', recFor({ route_stops: stops, route_plausible: true })]]);
        const fc = buildFeatureCollection(aircraftByHex, 1000);
        const props = fc.features[0].properties;

        expect(typeof props.route_stops).toBe('string');
        expect(JSON.parse(props.route_stops)).toEqual(stops);
    });

    test('no route stays null, not an empty string or array', () => {
        const aircraftByHex = new Map([['a1b2c3', recFor({ route_stops: null, route_plausible: null })]]);
        const fc = buildFeatureCollection(aircraftByHex, 1000);
        expect(fc.features[0].properties.route_stops).toBeNull();
    });

    test('route_plausible stays a plain boolean (already primitive, no encoding needed)', () => {
        const aircraftByHex = new Map([['a1b2c3', recFor({ route_stops: null, route_plausible: false })]]);
        const fc = buildFeatureCollection(aircraftByHex, 1000);
        expect(fc.features[0].properties.route_plausible).toBe(false);
    });
});

describe('buildFeatureCollection baro_rate_fpm fallback', () => {
    function recFor(overrides = {}) {
        return {
            hex: 'a1b2c3', flight: 'ANZ423', category: '', receivedAt: 1000,
            lat: -41.3, lon: 174.8, gs: 200, track: 0, alt_baro: 5000,
            ...overrides,
        };
    }

    test('the real reported baro_rate wins when present, even with a derived fallback available', () => {
        const aircraftByHex = new Map([['a1b2c3', recFor({ baro_rate: -800, derivedBaroRateFpm: -1500 })]]);
        const fc = buildFeatureCollection(aircraftByHex, 1000);
        expect(fc.features[0].properties.baro_rate_fpm).toBe(-800);
    });

    test('falls back to the derived rate when baro_rate is missing', () => {
        const aircraftByHex = new Map([['a1b2c3', recFor({ baro_rate: null, derivedBaroRateFpm: -1500 })]]);
        const fc = buildFeatureCollection(aircraftByHex, 1000);
        expect(fc.features[0].properties.baro_rate_fpm).toBe(-1500);
    });

    test('null when neither the real nor the derived rate is available', () => {
        const aircraftByHex = new Map([['a1b2c3', recFor({ baro_rate: null, derivedBaroRateFpm: null })]]);
        const fc = buildFeatureCollection(aircraftByHex, 1000);
        expect(fc.features[0].properties.baro_rate_fpm).toBeNull();
    });
});

// Correction smoothing (issue: aircraft icons visibly snapping backward when a new
// real sample's position falls behind where constant-velocity dead reckoning had
// already extrapolated to -- see interpolatedPosition/smoothedPosition).
// prevDisplay=undefined (no prior displayed state) skips easing entirely -- this is
// what buildFeatureCollection's displayByHex=null default (see the wiring tests below)
// reduces to for every record.
describe('deriveDisplayState', () => {
    // gs=3600kt is unrealistic but keeps the numbers clean the same way the existing
    // interpolatedPosition tests do (60kt for 1hr = 1 degree) while staying under
    // MAX_EXTRAPOLATION_S's 60s cap: 3600kt for 60s covers the same 60nm distance.
    // alt_baro is 'ground' (non-numeric) by default so these position-focused cases
    // don't also pull altitude into the smoothed state -- see the dedicated altitude
    // smoothing tests below for that.
    function movingRec(overrides = {}) {
        return {
            hex: 'a1b2c3', flight: 'ANZ423', category: '', receivedAt: 1000,
            lat: 0, lon: 0, gs: 3600, track: 0, alt_baro: 'ground',
            ...overrides,
        };
    }

    // A 3600kt due-north flight covers exactly 1 degree of latitude in 60s.
    const ELAPSED_60S = 60;

    test('no prior display state (a genuine first sighting) renders the raw dead-reckoned target directly', () => {
        const state = deriveDisplayState(movingRec(), ELAPSED_60S, undefined, 1.0, 0.6);
        expect(state.pos).toEqual({ lat: 1, lon: 0 });
        // No prior display to compute a rendered bearing from -- falls back to the
        // raw reported track (movingRec's default, 0).
        expect(state.iconTrack).toBe(0);
    });

    test('an existing display position eases toward the new target rather than snapping to it', () => {
        const dtS = 0.6, tauS = 0.6;
        const target = interpolatedPosition({ lat: 0, lon: 0, gs: 3600, track: 0 }, 60);
        const expectedPos = smoothedPosition({ lat: 0.2, lon: 0 }, target, dtS, tauS);

        const state = deriveDisplayState(movingRec(), ELAPSED_60S, { lat: 0.2, lon: 0 }, dtS, tauS);

        expect(state.pos).toEqual(expectedPos);
        // The eased position must land strictly between the old display and the raw
        // target -- neither still at the old spot nor snapped straight to the target.
        expect(state.pos.lat).toBeGreaterThan(0.2);
        expect(state.pos.lat).toBeLessThan(1.0);
        // The rendered movement was due north the whole time (lon stays 0), so the
        // derived icon_track is 0 -- matches raw track here since nothing diverged,
        // but derived independently via bearingDeg, not copied.
        expect(state.iconTrack).toBe(0);
    });

    test('altitude eases toward its new target the same way position does, when a prior display altitude exists', () => {
        // baro_rate=6000ft/min for 60s (elapsed, capped at MAX_EXTRAPOLATION_S) ->
        // extrapolatedAltitude(10000, 6000, null, 60) = 10000 + 6000*(60/60) = 16000.
        // Computed via smoothedScalar directly (rather than hand-derived alpha math)
        // so this stays correct regardless of whether the 5000ft gap trips the rate
        // cap -- 6000/60 (MAX_ALT_CORRECTION_FPM/60) matches the internal constant.
        const dtS = 0.6, tauS = 0.6;
        const expectedAlt = smoothedScalar(11000, 16000, dtS, tauS, 6000 / 60);
        const rec = movingRec({ gs: 0, alt_baro: 10000, baro_rate: 6000 });

        const state = deriveDisplayState(rec, ELAPSED_60S, { lat: 0, lon: 0, alt: 11000 }, dtS, tauS);

        expect(state.altBaroFt).toBeCloseTo(expectedAlt, 6);
        // The actual bug this closes: a 5000ft gap must NOT resolve in one 0.6s step
        // (that's the "visible reversal" -- an implied ~500,000ft/min climb rate).
        expect(state.altBaroFt).toBeLessThan(11100);
    });

    test('altitude newly becoming known this cycle (no prior display altitude) is not eased -- renders directly', () => {
        // Prior display exists (position was already being tracked) but never had an
        // altitude yet -- e.g. the aircraft was previously 'ground'/unknown.
        const rec = movingRec({ gs: 0, alt_baro: 10000, baro_rate: 6000 });
        const state = deriveDisplayState(rec, ELAPSED_60S, { lat: 0, lon: 0 }, 0.6, 0.6);
        expect(state.altBaroFt).toBeCloseTo(16000, 6);
    });

    test('a non-numeric altitude (ground/unknown) is never smoothed -- stays non-numeric, same as unsmoothed', () => {
        const rec = movingRec({ gs: 0, alt_baro: 'ground' });
        const state = deriveDisplayState(rec, ELAPSED_60S, { lat: 0, lon: 0, alt: 500 }, 0.6, 0.6);
        expect(state.altBaroFt).toBe('ground');
    });

    // The actual bug this closes (caught live, repeatedly, on final approach): even
    // with the speed cap, a target landing behind the display still visibly slides
    // the icon backward for a moment -- real aircraft never do that, so it must be
    // held instead. See isBackwardCorrection's own docstring.
    test('a target landing behind the display (relative to current track) holds position instead of visibly moving backward', () => {
        // target = interpolatedPosition({lat:0,lon:0,gs:3600,track:0}, 60) = {lat:1,lon:0}
        const rec = movingRec({ lat: 0, lon: 0, gs: 3600, track: 0 });
        const prevDisplay = { lat: 2, lon: 0 };   // already further along than the new target

        const state = deriveDisplayState(rec, ELAPSED_60S, prevDisplay, 0.6, 0.6);

        expect(state.pos).toEqual({ lat: 2, lon: 0 });
        // No movement this frame (held) -- no bearing to derive, so icon_track falls
        // back to the raw reported track (0) rather than snapping.
        expect(state.iconTrack).toBe(0);
    });

    test('a target landing ahead of the display (relative to current track) eases normally, not held', () => {
        const rec = movingRec({ lat: 0, lon: 0, gs: 3600, track: 0 });
        const prevDisplay = { lat: 0.5, lon: 0 };   // behind the new target (1)

        const state = deriveDisplayState(rec, ELAPSED_60S, prevDisplay, 0.6, 0.6);

        expect(state.pos.lat).toBeGreaterThan(0.5);
    });

    // The actual feature this closes: the icon should point where it's actually
    // moving on screen, not blindly follow the raw reported track -- which can
    // diverge from the icon's real rendered path (a crosswind crab angle, or our own
    // extrapolation/smoothing mid-correction).
    test('icon_track is derived from the rendered movement direction, which can differ from the raw reported track', () => {
        // Raw track says due east (90), but the prior display sits well south of the
        // new target, so the icon's real eased movement has a genuine northward
        // component too -- it should point somewhere between north and east, not
        // snap to the raw 90.
        const rec = movingRec({ lat: 0, lon: 0, gs: 3600, track: 90 });
        const prevDisplay = { lat: -0.5, lon: 0, track: 90 };

        const state = deriveDisplayState(rec, ELAPSED_60S, prevDisplay, 0.6, 0.6);

        expect(state.iconTrack).toBeGreaterThan(0);
        expect(state.iconTrack).toBeLessThan(90);
    });

    test('icon_track falls back to the raw track on first sighting (nothing to derive a bearing from yet)', () => {
        const rec = movingRec({ lat: 0, lon: 0, gs: 3600, track: 137 });
        const state = deriveDisplayState(rec, ELAPSED_60S, undefined, 0.6, 0.6);
        expect(state.iconTrack).toBe(137);
    });

    // The actual bug this closes (found by replaying real captured adsb.lol data:
    // ICAO 781a53/c81e2c, live-reported as "the aircraft rendered sideways through its
    // takeoff roll, straightening only once airborne"). adsb.lol's `track` is null for
    // essentially an entire ground phase in real data (ADS-B ground track is
    // undefined/unbroadcast while near-stationary) -- with the old rule ("skip
    // smoothing when there's no prior NUMBER to ease from"), an aircraft could sit
    // with a null stored track for the whole taxi phase (nothing to derive a bearing
    // from either, no movement yet), then the FIRST real bearing reading once it
    // finally starts moving -- exactly the noisy, first-movement reading the earlier
    // twitch fix targeted -- rendered instantly, fully unsmoothed, because that "first
    // known" branch treated a null-so-far track as indistinguishable from a genuine
    // first sighting.
    test('a null raw track on first sighting stores a real number (0), not null, so a later bearing eases in rather than snapping', () => {
        // First sighting: no raw track, no movement to derive a bearing from either --
        // must still return a real number so a later frame has something to ease from
        // (the actual fix -- previously this stored track: null).
        const rec1 = movingRec({ lat: 0, lon: 0, gs: 0, track: null });
        const state1 = deriveDisplayState(rec1, 0, undefined, 0.6, 0.6);
        expect(state1.iconTrack).toBe(0);

        // A later real sample, easing from state1's returned display (exactly what
        // buildFeatureCollection would have persisted into displayByHex): the aircraft
        // has moved (track is STILL null, matching real ground-phase data) --
        // icon_track can only come from bearingDeg's position-delta, due east (90deg)
        // here.
        const rec2 = movingRec({ lat: 0, lon: 0.001, gs: 3600, track: null });
        const prevDisplay2 = { ...state1.pos, track: state1.iconTrack };
        const state2 = deriveDisplayState(rec2, 1.6, prevDisplay2, 0.6, 0.6);

        // Rate-capped from the held 0, same as any other correction -- at most
        // 10deg/s * 0.6s = 6deg this frame, nowhere near the raw 90deg bearing. The
        // bug this replaces: icon_track jumping straight to ~90 in one frame.
        expect(state2.iconTrack).toBeCloseTo(6, 6);
    });

    test('icon_track holds its previous value when there is no movement this frame (e.g. a hold, or a stationary aircraft)', () => {
        const rec = movingRec({ lat: 0, lon: 0, gs: 0, track: 45 });
        // Already displayed exactly at the target (gs=0 means the target never
        // advances) -- no movement this frame, so the previously-rendered track (200,
        // deliberately different from the raw 45) must be kept, not recomputed.
        const prevDisplay = { lat: 0, lon: 0, track: 200 };

        const state = deriveDisplayState(rec, ELAPSED_60S, prevDisplay, 0.6, 0.6);

        expect(state.iconTrack).toBe(200);
    });

    // End-to-end wiring check for recordFromFeature's stalled-position fix (see that
    // describe block) -- deriveDisplayState must actually use deadReckonGs, not
    // silently fall back to the real gs, once a record carries it.
    test('a stalled record (deadReckonGs=0) does not dead-reckon forward, even though the real reported gs is nonzero', () => {
        const rec = movingRec({ lat: 5, lon: 5, gs: 200, deadReckonGs: 0, track: 90 });
        const state = deriveDisplayState(rec, ELAPSED_60S, undefined, 0, 0.6);
        expect(state.pos).toEqual({ lat: 5, lon: 5 });   // unchanged from the raw reported position
    });

    // The actual bug this closes (reported live: the icon "twitches", snapping to a
    // direction 45+ degrees off before resetting back, around a real update landing).
    // Root cause: icon_track is derived from the tiny per-FRAME displayed movement
    // vector (~16ms worth of real motion at 60fps), but right after a real update
    // lands, that vector is dominated by the position-CORRECTION component instead --
    // a gap that accumulated over the whole ~11-13s poll interval, squeezed into one
    // frame's worth of easing. Even a modest, realistic position discrepancy (well
    // within normal ADS-B/GPS noise -- nowhere near a data-quality outlier) makes the
    // correction's direction swamp the aircraft's true heading in the computed
    // bearing for over a second. Simulated here at real requestAnimationFrame cadence
    // (dtS ~= 1/60s per frame), not the rest of this file's exaggerated 0.6s steps.
    test('icon_track does not swing wildly off the true heading in the seconds after a real update lands', () => {
        const trueTrack = 0;   // due north
        const gs = 450;   // typical jet cruise speed, knots
        const startLat = 40.0, startLon = -74.0;
        // 0.05nm (~90m) lateral discrepancy between the prior smoothed display and the
        // freshly-landed real sample -- realistic ADS-B/extrapolation drift, not a
        // contrived outlier.
        const lateralOffsetNm = 0.05;
        const cosLat = Math.cos((startLat * Math.PI) / 180);
        const lonOffset = (lateralOffsetNm / 60.0) / cosLat;

        const rec = movingRec({ lat: startLat + 0.001, lon: startLon + lonOffset, gs, track: trueTrack });
        let prevDisplay = { lat: startLat, lon: startLon, track: trueTrack };

        const dtS = 1 / 60;
        let peakDeviation = 0;
        for (let frame = 1; frame <= 180; frame++) {   // 3s of real animation frames
            const state = deriveDisplayState(rec, frame * dtS, prevDisplay, dtS, 0.6);
            const deviation = Math.min(
                Math.abs(state.iconTrack - trueTrack), 360 - Math.abs(state.iconTrack - trueTrack),
            );
            peakDeviation = Math.max(peakDeviation, deviation);
            prevDisplay = { ...state.pos, track: state.iconTrack };
        }

        // Unfixed (raw bearingDeg with no rate cap), this scenario peaks at ~38.8deg
        // on the very first frame. MAX_ICON_TURN_RATE_DEG_S keeps the same scenario
        // well clear of anything reading as a "wrong direction" snap.
        expect(peakDeviation).toBeLessThan(20);
    });
});

// buildFeatureCollection delegates all dead-reckoning/easing to deriveDisplayState
// (see that describe block above) -- these tests only check the wiring: that omitting
// displayByHex reduces to the raw target, and that a real displayByHex Map gets
// deriveDisplayState's result written back into it correctly (including the
// alt-key-inclusion rule) so the next call eases from here.
describe('buildFeatureCollection display-state wiring', () => {
    function movingRec(overrides = {}) {
        return {
            hex: 'a1b2c3', flight: 'ANZ423', category: '', receivedAt: 1000,
            lat: 0, lon: 0, gs: 3600, track: 0, alt_baro: 'ground',
            ...overrides,
        };
    }

    // receivedAt=1000ms; now = 1000ms + 60s (in ms) -> elapsed = exactly 60s (right at,
    // not past, the cap), so a 3600kt due-north flight moves exactly 1 degree of
    // latitude.
    const NOW_AFTER_60S = 1000 + 60 * 1000;

    test('omitting displayByHex renders the raw dead-reckoned target, unsmoothed (backward compatible)', () => {
        const aircraftByHex = new Map([['a1b2c3', movingRec()]]);
        const fc = buildFeatureCollection(aircraftByHex, NOW_AFTER_60S);
        expect(fc.features[0].geometry.coordinates).toEqual([0, 1]);
    });

    test('a real displayByHex Map is populated on first sighting, keyed by hex', () => {
        const aircraftByHex = new Map([['a1b2c3', movingRec()]]);
        const displayByHex = new Map();
        buildFeatureCollection(aircraftByHex, NOW_AFTER_60S, displayByHex, 1.0, 0.6);
        expect(displayByHex.get('a1b2c3')).toEqual({ lat: 1, lon: 0, track: 0 });
    });

    test('the eased position and derived track are persisted into displayByHex, not the raw target', () => {
        const aircraftByHex = new Map([['a1b2c3', movingRec()]]);
        const displayByHex = new Map([['a1b2c3', { lat: 0.2, lon: 0 }]]);
        const dtS = 0.6, tauS = 0.6;
        const target = interpolatedPosition({ lat: 0, lon: 0, gs: 3600, track: 0 }, 60);
        const expectedPos = smoothedPosition({ lat: 0.2, lon: 0 }, target, dtS, tauS);

        buildFeatureCollection(aircraftByHex, NOW_AFTER_60S, displayByHex, dtS, tauS);

        // displayByHex is updated in place so the next frame eases from here, not
        // from the stale 0.2 starting point.
        expect(displayByHex.get('a1b2c3')).toEqual({ ...expectedPos, track: 0 });
    });

    test('a numeric eased altitude is persisted into displayByHex under `alt`', () => {
        const aircraftByHex = new Map([['a1b2c3', movingRec({ gs: 0, alt_baro: 10000, baro_rate: 6000 })]]);
        const displayByHex = new Map([['a1b2c3', { lat: 0, lon: 0, alt: 11000 }]]);

        buildFeatureCollection(aircraftByHex, NOW_AFTER_60S, displayByHex, 0.6, 0.6);

        expect(displayByHex.get('a1b2c3').alt).toBeCloseTo(smoothedScalar(11000, 16000, 0.6, 0.6, 6000 / 60), 6);
    });

    test('a non-numeric altitude (ground/unknown) is omitted from displayByHex, not carried forward as stale, and falls back to 0 in the built feature', () => {
        const aircraftByHex = new Map([['a1b2c3', movingRec({ gs: 0, alt_baro: 'ground' })]]);
        const displayByHex = new Map([['a1b2c3', { lat: 0, lon: 0, alt: 500 }]]);

        const fc = buildFeatureCollection(aircraftByHex, NOW_AFTER_60S, displayByHex, 0.6, 0.6);

        expect(fc.features[0].properties.alt_baro_ft).toBe(0);
        // The stale alt from a previous (now-ended) climb/descent must not linger.
        expect(displayByHex.get('a1b2c3').alt).toBeUndefined();
    });
});

describe('flightStatus', () => {
    test('a clearly positive climb rate is Climbing', () => {
        expect(flightStatus(false, 250, 1000)).toBe('Climbing');
    });

    test('a clearly negative rate is Descending', () => {
        expect(flightStatus(false, 250, -1000)).toBe('Descending');
    });

    test('zero rate is Level flight', () => {
        expect(flightStatus(false, 250, 0)).toBe('Level flight');
    });

    test('small positive noise within the deadband stays Level flight', () => {
        expect(flightStatus(false, 250, 100, 150)).toBe('Level flight');
    });

    test('small negative noise within the deadband stays Level flight', () => {
        expect(flightStatus(false, 250, -100, 150)).toBe('Level flight');
    });

    test('just past the deadband on either side switches state', () => {
        expect(flightStatus(false, 250, 151, 150)).toBe('Climbing');
        expect(flightStatus(false, 250, -151, 150)).toBe('Descending');
    });

    test('missing rate data defaults to Level flight', () => {
        expect(flightStatus(false, 250, null)).toBe('Level flight');
        expect(flightStatus(false, 250, undefined)).toBe('Level flight');
    });

    test('on the ground and stationary reads as Landed, regardless of vertical rate', () => {
        expect(flightStatus(true, 0, null)).toBe('Landed');
    });

    test('on the ground but still moving is ambiguous (taxiing, rollout, or takeoff roll) and renders nothing', () => {
        expect(flightStatus(true, 12, null)).toBe('');
        expect(flightStatus(true, 12, 1000)).toBe('');
    });
});

describe('targetAltitudeLabel', () => {
    test('an exact match reads as Reached', () => {
        expect(targetAltitudeLabel(37000, 37000)).toBe('Reached');
    });

    test('within tolerance (real-world sensor/MCP noise) also reads as Reached', () => {
        // A real adsb.lol record at cruise: alt_baro=37000, nav_altitude_mcp=36992.
        expect(targetAltitudeLabel(36992, 37000)).toBe('Reached');
    });

    test('a target well away from current altitude renders the formatted number', () => {
        expect(targetAltitudeLabel(38000, 35000)).toBe('38,000 ft');
    });

    test('no target altitude data available renders nothing', () => {
        expect(targetAltitudeLabel(null, 35000)).toBe(null);
        expect(targetAltitudeLabel(undefined, 35000)).toBe(null);
    });

    test('current altitude unknown still shows the raw target', () => {
        expect(targetAltitudeLabel(37000, null)).toBe('37,000 ft');
    });

    test('ambiguous ground state (taxiing/rollout/takeoff roll) suppresses the target entirely, even with a valid MCP value', () => {
        expect(targetAltitudeLabel(3000, 0, true)).toBe(null);
    });

    test('once no longer ambiguous, target altitude resumes as normal', () => {
        expect(targetAltitudeLabel(38000, 35000, false)).toBe('38,000 ft');
    });
});

describe('aircraftClass', () => {
    test('a real live-captured widebody type designator resolves correctly', () => {
        // B77W = Boeing 777-300ER, captured from real adsb.lol traffic (see #203's popup work).
        expect(aircraftClass('B77W')).toBe('Widebody Jet');
    });

    test('resolves one designator from each register category', () => {
        expect(aircraftClass('B738')).toBe('Narrowbody Jet');   // 737-800
        expect(aircraftClass('E190')).toBe('Regional Jet');     // Embraer E190
        expect(aircraftClass('AT76')).toBe('Turboprop');        // ATR72-600
        expect(aircraftClass('GLF6')).toBe('Business Jet');     // Gulfstream G650
        expect(aircraftClass('C172')).toBe('Light Aircraft');   // Cessna 172
        expect(aircraftClass('R44')).toBe('Helicopter');        // Robinson R44
        expect(aircraftClass('F16')).toBe('Military Aircraft'); // F-16
    });

    test('is case-insensitive (adsb.lol always sends uppercase, but do not depend on it)', () => {
        expect(aircraftClass('b77w')).toBe('Widebody Jet');
    });

    test('an unregistered designator falls back to a vague default', () => {
        expect(aircraftClass('ZZZZ')).toBe('Aircraft (unclassified)');
    });

    test('missing type data falls back to the same vague default', () => {
        expect(aircraftClass(null)).toBe('Aircraft (unclassified)');
        expect(aircraftClass(undefined)).toBe('Aircraft (unclassified)');
        expect(aircraftClass('')).toBe('Aircraft (unclassified)');
    });
});

describe('aircraftGroup', () => {
    test('widebody jets are their own group', () => {
        expect(aircraftGroup('B77W')).toBe('widebody');
    });

    test('narrowbody and regional jets both fold into the airliner group', () => {
        expect(aircraftGroup('B738')).toBe('airliner');
        expect(aircraftGroup('E190')).toBe('airliner');
    });

    test('turboprops, business jets, and light aircraft all fold into the light group', () => {
        expect(aircraftGroup('AT76')).toBe('light');
        expect(aircraftGroup('GLF6')).toBe('light');
        expect(aircraftGroup('C172')).toBe('light');
    });

    test('helicopters, military, and unclassified types all fall into the other group', () => {
        expect(aircraftGroup('R44')).toBe('other');
        expect(aircraftGroup('F16')).toBe('other');
        expect(aircraftGroup('ZZZZ')).toBe('other');
        expect(aircraftGroup(null)).toBe('other');
    });

    test('there are fewer than 5 groups in total', () => {
        const allGroups = new Set(
            ['B77W', 'B738', 'E190', 'AT76', 'GLF6', 'C172', 'R44', 'F16', 'ZZZZ'].map(aircraftGroup)
        );
        expect(allGroups.size).toBeLessThan(5);
    });
});

describe('aircraftGroupColor', () => {
    test('each group resolves to a distinct color', () => {
        const widebody = aircraftGroupColor('B77W');
        const airliner = aircraftGroupColor('B738');
        const light = aircraftGroupColor('C172');
        const other = aircraftGroupColor('R44');
        expect(new Set([widebody, airliner, light, other]).size).toBe(4);
    });

    test('an unregistered type still resolves to a valid color (the other group)', () => {
        expect(aircraftGroupColor('ZZZZ')).toBe(aircraftGroupColor('R44'));
    });
});

describe('airlineForFlight', () => {
    test('a real live-captured callsign resolves to its airline', () => {
        // ANZ583L, captured from real adsb.lol traffic (see #203's popup work).
        expect(airlineForFlight('ANZ583L')).toBe('Air New Zealand');
    });

    test('resolves regardless of trailing flight-number digits/suffix', () => {
        expect(airlineForFlight('QFA938')).toBe('Qantas');
        expect(airlineForFlight('UAL1')).toBe('United Airlines');
    });

    test('is case-insensitive (adsb.lol always sends uppercase, but do not depend on it)', () => {
        expect(airlineForFlight('anz583l')).toBe('Air New Zealand');
    });

    test('ignores surrounding whitespace (adsb.lol pads flight to a fixed width)', () => {
        expect(airlineForFlight('ANZ583L ')).toBe('Air New Zealand');
    });

    test('a callsign with no recognized 3-letter designator renders no airline', () => {
        expect(airlineForFlight('ZZZ123')).toBe(null);
    });

    test('a GA aircraft broadcasting its own registration as callsign renders no airline', () => {
        expect(airlineForFlight('N12345')).toBe(null);
    });

    test('missing flight data renders no airline', () => {
        expect(airlineForFlight(null)).toBe(null);
        expect(airlineForFlight(undefined)).toBe(null);
        expect(airlineForFlight('')).toBe(null);
    });
});
