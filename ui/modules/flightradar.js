// ui/modules/flightradar.js
// Client-side dead-reckoning + bounded-freeze helpers behind the Flight Radar layer
// (issue #203, docs/adr/0009). The backend pushes aircraft state on its own
// hot/gentle cadence (routes/flightradar.py, 2-20s); between pushes, the frontend
// extrapolates each aircraft's position from its last-known lat/lon + ground speed +
// track via requestAnimationFrame, so movement reads as smooth rather than snapping
// every few seconds. If updates stop arriving (disconnect, region churn),
// boundedElapsedSeconds caps how far that extrapolation is allowed to run --
// past MAX_EXTRAPOLATION_S the aircraft freezes in place rather than flying off along
// a stale heading indefinitely.

const NM_PER_DEGREE_LAT = 60.0;
// 60s, not a tighter cap: AircraftCollector's GlobalSampleScheduler round-robins up
// to MAX_HOT_CELLS_PER_VIEWPORT (12) hot cells through one shared
// requests_per_minute budget, so any single cell's worst-case refresh cycle is
// ~(hot cell count * tick interval) -- comfortably under 60s at the collector's
// default 12rpm (5s/tick * 12 cells = 60s), but a much shorter cap here would flag
// "signal lost" on effectively every aircraft in view, every cycle.
const MAX_EXTRAPOLATION_S = 60.0;

export function boundedElapsedSeconds(lastSeenMs, nowMs, maxExtrapolationS = MAX_EXTRAPOLATION_S) {
    const elapsed = (nowMs - lastSeenMs) / 1000.0;
    return Math.max(0, Math.min(elapsed, maxExtrapolationS));
}

// Kept as its own predicate (not derived from boundedElapsedSeconds' clamped output)
// so the render loop can show a "signal lost" cue exactly when extrapolation has
// been capped, without the two concerns folded into one function -- see issue #203's
// Testing Decisions ("should we freeze" kept separate from the interpolation itself).
export function isFrozen(lastSeenMs, nowMs, maxExtrapolationS = MAX_EXTRAPOLATION_S) {
    return (nowMs - lastSeenMs) / 1000.0 >= maxExtrapolationS;
}

// onGround comes straight from adsb.lol's own squat/weight-on-wheels transponder field
// (alt_baro === "ground") -- never computed by comparing altitude to field elevation,
// so it's correct even at a high-elevation airport. gs still matters once on the
// ground: gs=0 is unambiguously Landed, but gs>0 covers taxiing, the last few seconds
// of rollout after touchdown, and the first few seconds of the takeoff roll all at
// once -- ADS-B alone can't tell those apart, so this deliberately renders nothing
// rather than guessing (see targetAltitudeLabel's matching ambiguity handling below).
//
// adsb.lol's baro_rate (ft/min, barometric vertical rate) is noisy even in level
// flight -- a deadband avoids the popup flickering between Climbing/Descending/Level
// on sensor jitter alone. 150ft/min is a conservative starting guess (real level-flight
// noise is usually well under 100ft/min), tunable like every other numeric constant in
// this feature.
export function flightStatus(onGround, gs, baroRateFpm, deadbandFpm = 150) {
    if (onGround) return gs === 0 ? 'Landed' : '';
    if (typeof baroRateFpm !== 'number') return 'Level flight';
    if (baroRateFpm > deadbandFpm) return 'Climbing';
    if (baroRateFpm < -deadbandFpm) return 'Descending';
    return 'Level flight';
}

// nav_altitude_mcp (the autopilot/MCP-selected target altitude) rarely matches
// alt_baro (the actual sensed altitude) exactly even once an aircraft has settled at
// cruise -- a real adsb.lol record: alt_baro=37000, nav_altitude_mcp=36992. A small
// tolerance (rather than bit-exact equality) is what makes "Reached" actually fire in
// practice.
//
// altitudeAmbiguous is true exactly when flightStatus's on-the-ground-but-moving case
// applies: the MCP can already hold a pre-programmed climb target while still taxiing,
// but there's no current altitude reading to sensibly compare it against (alt_baro is
// the literal "ground" state, not a number) -- so the target is withheld entirely
// rather than showing a number that looks like live progress-to-target.
export function targetAltitudeLabel(navAltitudeMcpFt, altBaroFt, altitudeAmbiguous, toleranceFt = 50) {
    if (altitudeAmbiguous) return null;
    if (typeof navAltitudeMcpFt !== 'number') return null;
    if (typeof altBaroFt === 'number' && Math.abs(navAltitudeMcpFt - altBaroFt) <= toleranceFt) {
        return 'Reached';
    }
    return `${Math.round(navAltitudeMcpFt).toLocaleString()} ft`;
}

// Flat-earth dead reckoning -- accurate enough over the few-second gaps this bridges
// (see MAX_EXTRAPOLATION_S), not meant for long-range navigation. track is degrees
// true, clockwise from north (adsb.lol's `track`/`true_heading` field); gs is ground
// speed in knots (adsb.lol's `gs`). Longitude degrees shrink toward the poles
// (1 / cos(lat)), same convergence correction as everywhere else in this codebase that
// converts a distance to a longitude delta.
//
// Deliberately constant-velocity, not acceleration-aware: an earlier version of this
// function projected a rate of change of gs (derived from just the two most recent
// real samples) forward via distance = v0*t + 0.5*a*t^2. That made things WORSE, not
// better -- a real aircraft's speed doesn't change at a constant rate for the full
// 10-25s+ gap between hot-cell samples (it happens in bursts: flaps, gear, speed
// brakes), so whatever rate happened to show up between two samples got projected
// forward and reliably overshot, and because the error term is quadratic in elapsed
// time, it compounded badly on final approach specifically -- visible live as smooth
// but "obviously over-cooked" position swings on final approach into a busy airport,
// even with smoothedPosition's correction-easing layered on top (smoothing chases the
// target faithfully; the bug was the target itself). Reverted in favour of this
// simpler, more modest constant-velocity guess plus smoothedPosition below.
export function interpolatedPosition({ lat, lon, gs, track }, elapsedSeconds) {
    if (!gs || !elapsedSeconds || track == null) return { lat, lon };
    const distanceNm = gs * (elapsedSeconds / 3600.0);
    const trackRad = (track * Math.PI) / 180.0;
    const deltaLat = (distanceNm / NM_PER_DEGREE_LAT) * Math.cos(trackRad);
    const cosLat = Math.cos((lat * Math.PI) / 180.0);
    const deltaLon = cosLat !== 0 ? ((distanceNm / NM_PER_DEGREE_LAT) * Math.sin(trackRad)) / cosLat : 0;
    return { lat: lat + deltaLat, lon: lon + deltaLon };
}

// Compass bearing (degrees, clockwise from north) from `from` to `to` -- the inverse
// of interpolatedPosition's deltaLat/deltaLon decomposition (same flat-earth
// approximation and longitude-convergence correction, just run backward). Used to
// derive the ICON'S rotation from its own rendered movement rather than the raw
// reported track/heading -- adsb.lol's `track` is ground track, not heading, but a
// real aircraft's track and heading still diverge in a crosswind (crab angle), and
// separately, our own extrapolation/smoothing/backward-hold pipeline (see
// isBackwardCorrection) means the icon's ACTUAL on-screen path doesn't always match
// raw `track` moment-to-moment either. Deriving rotation from consecutive rendered
// positions instead guarantees the icon always visibly points the way it's actually
// moving, by construction. Returns null when the two points coincide (no movement
// this frame -- e.g. held per isBackwardCorrection, or genuinely stationary) --
// callers should hold the previous rotation rather than snapping to some arbitrary
// angle for a zero-length vector.
export function bearingDeg(from, to) {
    const cosLat = Math.cos((from.lat * Math.PI) / 180.0);
    const dLat = to.lat - from.lat;
    const dLon = (to.lon - from.lon) * cosLat;
    if (dLat === 0 && dLon === 0) return null;
    const deg = (Math.atan2(dLon, dLat) * 180.0) / Math.PI;
    return deg < 0 ? deg + 360 : deg;
}

// How fast the DISPLAYED state (position and/or altitude) catches up to its true
// dead-reckoned target (smoothedPosition/smoothedScalar's default time constant).
// Chosen so a correction settles in roughly 1-2s (tau=0.6s: ~63% closed after 0.6s,
// ~95% after ~1.8s) without reading as laggy -- tunable like every other
// empirically-picked constant in this codebase. Tradeoff: during SUSTAINED constant
// motion (no correction needed) this exponential filter settles into a small constant
// lag behind the target, roughly rate*tauS (at cruise speed ~450kt, ~0.6s of lag is on
// the order of ~140m) -- accepted in exchange for eliminating the visible snap a hard
// reset causes.
const SMOOTH_TAU_S = 0.6;

// Caps on how fast a correction is visually allowed to close (see smoothedPosition/
// smoothedScalar) -- both set well above realistic sustained aircraft performance
// (a strong-tailwind groundspeed outlier for the former; a steep-but-plausible
// descent for the latter) so neither ever limits an ordinary correction, only an
// extreme one caused by our own extrapolation error.
const MAX_CORRECTION_KTS = 800;
const MAX_ALT_CORRECTION_FPM = 6000;

// icon_track's own correction-rate cap (see smoothedAngle below), the rotational
// counterpart to MAX_CORRECTION_KTS/MAX_ALT_CORRECTION_FPM. Set well above a real
// "standard rate turn" (3deg/s, the aviation-standard reference for a normal turn) so
// it never visibly lags a genuine turn (a sustained 3deg/s turn lags well under 2deg
// behind the true heading at this cap) -- it only limits the much faster, spurious
// swings bearingDeg's position-delta-derived rotation can otherwise produce right
// after a real update lands, when a modest, realistic position correction (well
// within normal ADS-B/GPS noise) is squeezed into a single ~16ms animation frame: a
// correction distance utterly dwarfing that frame's own genuine dead-reckoned
// movement dominates the computed bearing, briefly pointing the icon somewhere far
// from its real heading before easing back over the following ~1s (caught live: a
// twitch/snap-then-reset, worse the smaller and more frequent the underlying
// correction). Capping the RATE the icon is allowed to visibly rotate, independent of
// where the raw signal briefly points, suppresses that without needing to first
// classify which bearing readings are trustworthy.
const MAX_ICON_TURN_RATE_DEG_S = 10;

// Exponential smoothing of the DISPLAYED position toward a (possibly moving) target,
// framerate-independent via the elapsed-time-based alpha below, with the per-frame
// movement capped to MAX_CORRECTION_KTS worth of real distance (same flat-earth
// approximation interpolatedPosition uses). Plain exponential smoothing alone closes
// a LARGE error (kilometers -- e.g. a landing aircraft's real deceleration outrunning
// constant-velocity dead reckoning between samples, see interpolatedPosition) just as
// fast in wall-clock time as a small one, since tau alone doesn't know how far it has
// to go -- that reads as the icon sliding at an obviously-impossible speed rather
// than as a gentle correction (caught live: "smooth but totally weird to watch").
// Capping the per-frame distance instead means small errors still resolve almost
// immediately (well under the cap) while large ones take proportionally longer, at a
// speed that looks like real aircraft motion. dtS<=0 (first frame, or a defensive
// guard against clock weirdness) leaves the displayed position unchanged.
export function smoothedPosition(display, target, dtS, tauS = SMOOTH_TAU_S) {
    if (!dtS || dtS <= 0) return display;
    const alpha = 1 - Math.exp(-dtS / tauS);
    let deltaLat = (target.lat - display.lat) * alpha;
    let deltaLon = (target.lon - display.lon) * alpha;

    const cosLat = Math.cos((display.lat * Math.PI) / 180.0);
    const distNm = Math.sqrt(
        (deltaLat * NM_PER_DEGREE_LAT) ** 2 + (deltaLon * NM_PER_DEGREE_LAT * cosLat) ** 2,
    );
    const maxDistNm = MAX_CORRECTION_KTS * (dtS / 3600.0);
    if (distNm > maxDistNm) {
        const scale = maxDistNm / distNm;
        deltaLat *= scale;
        deltaLon *= scale;
    }
    return { lat: display.lat + deltaLat, lon: display.lon + deltaLon };
}

// True if moving from `display` toward `target` would be BACKWARD progress along
// `trackDeg` (degrees true, clockwise from north -- the aircraft's own current
// heading). A real aircraft never flies backward, so a target landing behind the
// currently-displayed position relative to its own heading is always an artifact of
// our extrapolation overshooting, never genuine aircraft motion -- caught live on
// final approach: constant-velocity dead reckoning assumes the pre-touchdown speed
// holds for the whole sample gap, but a landing aircraft decelerates hard within that
// same gap, so the real position is consistently less advanced than the guess.
//
// A capped/eased correction (smoothedPosition alone) still visibly slides the icon
// backward for a moment -- slower than a hard snap, but any backward motion at all
// reads as obviously wrong, since it's something no real aircraft does. Detecting this
// case and holding position instead (see buildFeatureCollection) avoids ever showing
// it, at the cost of the icon briefly pausing until a later real sample shows genuine
// progress past wherever it's held -- a much smaller, self-resolving artifact than a
// visible reversal.
export function isBackwardCorrection(display, target, trackDeg) {
    if (trackDeg == null) return false;
    const deltaLatNm = (target.lat - display.lat) * NM_PER_DEGREE_LAT;
    const cosLat = Math.cos((display.lat * Math.PI) / 180.0);
    const deltaLonNm = (target.lon - display.lon) * NM_PER_DEGREE_LAT * cosLat;
    const trackRad = (trackDeg * Math.PI) / 180.0;
    const progressNm = deltaLatNm * Math.cos(trackRad) + deltaLonNm * Math.sin(trackRad);
    return progressNm < 0;
}

// 1D counterpart of smoothedPosition -- same exponential-smoothing-with-a-rate-cap
// shape, for a plain scalar (altitude) rather than a geographic position, since
// altitude's unit (feet) doesn't need or share smoothedPosition's distance math.
// maxRatePerSecond is in the same units as display/target, per second (pass
// MAX_ALT_CORRECTION_FPM / 60 for altitude in feet).
export function smoothedScalar(display, target, dtS, tauS, maxRatePerSecond) {
    if (!dtS || dtS <= 0) return display;
    const alpha = 1 - Math.exp(-dtS / tauS);
    let delta = (target - display) * alpha;
    const maxDelta = maxRatePerSecond * dtS;
    if (Math.abs(delta) > maxDelta) delta = Math.sign(delta) * maxDelta;
    return display + delta;
}

// Angular counterpart of smoothedScalar -- same exponential-smoothing-with-a-rate-cap
// shape, but closing via the SHORTEST signed angular distance (so e.g. display=350,
// target=10 eases through 360/0, a 20deg turn, never the long way round through 180)
// and wrapped back into [0, 360) rather than left to drift outside that range.
// maxRateDegPerSecond is in degrees per second (pass MAX_ICON_TURN_RATE_DEG_S).
// See buildFeatureCollection's icon_track derivation for why this exists: bearingDeg's
// raw position-delta-derived rotation can briefly swing far from the true heading
// right after a real update lands (see MAX_ICON_TURN_RATE_DEG_S's own comment) --
// capping the icon's visible turn rate suppresses that without ever meaningfully
// lagging a genuine turn, which is much slower.
export function smoothedAngle(display, target, dtS, tauS, maxRateDegPerSecond) {
    if (display == null) return target;
    if (target == null || !dtS || dtS <= 0) return display;
    let delta = ((target - display + 540) % 360) - 180;
    const alpha = 1 - Math.exp(-dtS / tauS);
    delta *= alpha;
    const maxDelta = maxRateDegPerSecond * dtS;
    if (Math.abs(delta) > maxDelta) delta = Math.sign(delta) * maxDelta;
    return ((display + delta) % 360 + 360) % 360;
}

// Vertical counterpart to interpolatedPosition -- projects baro_rate (ft/min) forward
// by elapsedSeconds, same dead-reckoning idea as the lat/lon case. Clamped so the
// projection never overshoots the aircraft's own MCP-selected target
// (nav_altitude_mcp, same field targetAltitudeLabel reads) in the direction of
// travel: a real aircraft levels off at its target between polls rather than
// climbing/descending straight through it, and naively projecting baro_rate forward
// unclamped would show it doing exactly that. Only clamps when the target is
// actually ahead in the current direction of travel (climbing toward a higher
// target, or descending toward a lower one) -- a stale/contradictory MCP reading
// (e.g. still showing a past target behind the aircraft) is left unclamped rather
// than snapping the projection back to it.
export function extrapolatedAltitude(altBaroFt, baroRateFpm, targetAltitudeFt, elapsedSeconds) {
    if (typeof altBaroFt !== 'number') return altBaroFt;
    if (typeof baroRateFpm !== 'number' || !elapsedSeconds) return altBaroFt;
    const projected = altBaroFt + baroRateFpm * (elapsedSeconds / 60.0);
    if (typeof targetAltitudeFt !== 'number') return projected;
    if (baroRateFpm > 0 && targetAltitudeFt > altBaroFt) return Math.min(projected, targetAltitudeFt);
    if (baroRateFpm < 0 && targetAltitudeFt < altBaroFt) return Math.max(projected, targetAltitudeFt);
    return projected;
}

// ICAO aircraft type designator (adsb.lol's `t` field, e.g. "B77W") -> a broad,
// human-friendly class for the hover popup. Locally maintained and deliberately not
// exhaustive -- covers designators likely to actually show up in real ADS-B traffic
// (airliners, regional/business jets, turboprops, light GA, rotorcraft, a handful of
// military types). An unregistered designator (including any adsb.lol variant/typo
// this list doesn't happen to cover) falls back to a vague default rather than
// guessing -- see aircraftClass()'s default.
const AIRCRAFT_CLASS_REGISTER = {
    // Widebody jets
    A332: 'Widebody Jet', A333: 'Widebody Jet', A338: 'Widebody Jet', A339: 'Widebody Jet',
    A342: 'Widebody Jet', A343: 'Widebody Jet', A345: 'Widebody Jet', A346: 'Widebody Jet',
    A359: 'Widebody Jet', A35K: 'Widebody Jet', A388: 'Widebody Jet',
    B744: 'Widebody Jet', B748: 'Widebody Jet', B772: 'Widebody Jet', B773: 'Widebody Jet',
    B77L: 'Widebody Jet', B77W: 'Widebody Jet', B788: 'Widebody Jet', B789: 'Widebody Jet',
    B78X: 'Widebody Jet', MD11: 'Widebody Jet',

    // Narrowbody jets
    A318: 'Narrowbody Jet', A319: 'Narrowbody Jet', A320: 'Narrowbody Jet', A321: 'Narrowbody Jet',
    A19N: 'Narrowbody Jet', A20N: 'Narrowbody Jet', A21N: 'Narrowbody Jet',
    B712: 'Narrowbody Jet', B737: 'Narrowbody Jet', B738: 'Narrowbody Jet', B739: 'Narrowbody Jet',
    B37M: 'Narrowbody Jet', B38M: 'Narrowbody Jet', B39M: 'Narrowbody Jet', B3XM: 'Narrowbody Jet',
    B752: 'Narrowbody Jet', B753: 'Narrowbody Jet',
    MD82: 'Narrowbody Jet', MD83: 'Narrowbody Jet', MD90: 'Narrowbody Jet',

    // Regional jets
    CRJ1: 'Regional Jet', CRJ2: 'Regional Jet', CRJ7: 'Regional Jet', CRJ9: 'Regional Jet',
    CRJX: 'Regional Jet',
    E135: 'Regional Jet', E145: 'Regional Jet', E170: 'Regional Jet', E175: 'Regional Jet',
    E190: 'Regional Jet', E195: 'Regional Jet', E290: 'Regional Jet', E295: 'Regional Jet',
    SU95: 'Regional Jet', ARJ21: 'Regional Jet', F70: 'Regional Jet', F100: 'Regional Jet',

    // Turboprops
    AT43: 'Turboprop', AT45: 'Turboprop', AT72: 'Turboprop', AT75: 'Turboprop', AT76: 'Turboprop',
    DH8A: 'Turboprop', DH8B: 'Turboprop', DH8C: 'Turboprop', DH8D: 'Turboprop',
    SF34: 'Turboprop', B190: 'Turboprop', C208: 'Turboprop', C208B: 'Turboprop',
    PC12: 'Turboprop', TBM7: 'Turboprop', TBM8: 'Turboprop', TBM9: 'Turboprop',
    D328: 'Turboprop', SW4: 'Turboprop', BE20: 'Turboprop',

    // Business jets
    GLF4: 'Business Jet', GLF5: 'Business Jet', GLF6: 'Business Jet',
    CL30: 'Business Jet', CL35: 'Business Jet', CL60: 'Business Jet',
    GLEX: 'Business Jet', GL5T: 'Business Jet', GL6T: 'Business Jet',
    C525: 'Business Jet', C550: 'Business Jet', C560: 'Business Jet', C56X: 'Business Jet',
    C650: 'Business Jet', C680: 'Business Jet', C68A: 'Business Jet', C700: 'Business Jet',
    C750: 'Business Jet',
    FA7X: 'Business Jet', FA8X: 'Business Jet', FA50: 'Business Jet', FA6X: 'Business Jet',
    FA20: 'Business Jet',
    LJ35: 'Business Jet', LJ45: 'Business Jet', LJ60: 'Business Jet', LJ75: 'Business Jet',
    PC24: 'Business Jet', E50P: 'Business Jet', E55P: 'Business Jet', H25B: 'Business Jet',

    // Light aircraft (piston GA)
    C152: 'Light Aircraft', C172: 'Light Aircraft', C182: 'Light Aircraft', C206: 'Light Aircraft',
    P28A: 'Light Aircraft', PA31: 'Light Aircraft', PA34: 'Light Aircraft', PA44: 'Light Aircraft',
    BE33: 'Light Aircraft', BE35: 'Light Aircraft', BE36: 'Light Aircraft', BE58: 'Light Aircraft',
    M20P: 'Light Aircraft', M20T: 'Light Aircraft', SR20: 'Light Aircraft', SR22: 'Light Aircraft',
    DA40: 'Light Aircraft', DA42: 'Light Aircraft',

    // Helicopters
    R22: 'Helicopter', R44: 'Helicopter', R66: 'Helicopter',
    EC30: 'Helicopter', EC35: 'Helicopter', EC45: 'Helicopter', EC55: 'Helicopter',
    AS50: 'Helicopter', AS55: 'Helicopter', AS65: 'Helicopter',
    A109: 'Helicopter', A119: 'Helicopter', A139: 'Helicopter', A169: 'Helicopter', A189: 'Helicopter',
    B06: 'Helicopter', B47: 'Helicopter', B407: 'Helicopter', B412: 'Helicopter',
    B429: 'Helicopter', B430: 'Helicopter',
    S76: 'Helicopter', S92: 'Helicopter',
    H60: 'Helicopter', AH64: 'Helicopter', UH1: 'Helicopter',

    // Military (fixed-wing) -- a small representative set; ADS-B military traffic is
    // rare and often squawks without a populated `t` at all.
    F15: 'Military Aircraft', F16: 'Military Aircraft', F18: 'Military Aircraft',
    F22: 'Military Aircraft', F35: 'Military Aircraft',
    C130: 'Military Aircraft', C17: 'Military Aircraft',
    B52: 'Military Aircraft', A10: 'Military Aircraft',
};

const DEFAULT_AIRCRAFT_CLASS = 'Aircraft (unclassified)';

export function aircraftClass(typeCode) {
    if (!typeCode) return DEFAULT_AIRCRAFT_CLASS;
    return AIRCRAFT_CLASS_REGISTER[typeCode.toUpperCase()] || DEFAULT_AIRCRAFT_CLASS;
}

// Coarser than aircraftClass() -- 4 icon-color groups, roughly by size/prominence
// (a widebody is more globally significant than a light aircraft, same reasoning
// docs/adr/0008 uses for the altitude zoom filter). 'other' is the catch-all for
// anything that isn't a fixed-wing airliner/GA type -- helicopters, military, and
// anything aircraftClass() couldn't identify at all.
export function aircraftGroup(typeCode) {
    const cls = aircraftClass(typeCode);
    if (cls === 'Widebody Jet') return 'widebody';
    if (cls === 'Narrowbody Jet' || cls === 'Regional Jet') return 'airliner';
    if (cls === 'Turboprop' || cls === 'Business Jet' || cls === 'Light Aircraft') return 'light';
    return 'other';
}

// Brightness/warmth roughly tracks group size: near-white for the largest widebodies,
// down to light blue for the smallest/other-shaped craft. Multiplied onto the SDF
// icon at render time (icon-color paint property) rather than baked into the PNG, so
// one shape asset per icon (aircraft/glider) covers every group.
const AIRCRAFT_GROUP_COLORS = {
    widebody: '#f2f2f2',
    airliner: '#ffd166',
    light: '#ff8c42',
    other: '#8ec9ff',
};

export function aircraftGroupColor(typeCode) {
    return AIRCRAFT_GROUP_COLORS[aircraftGroup(typeCode)];
}

// adsb.lol's `flight` field is the ICAO callsign (e.g. "ANZ583L"), whose leading 3
// letters are the aircraft operator's standard ICAO airline designator -- the same
// mechanism real flight trackers use to show an airline name. Locally maintained and
// deliberately not exhaustive (thousands of these exist worldwide), same register/
// fallback shape as AIRCRAFT_CLASS_REGISTER -- covers major/regional carriers likely to
// actually show up in real traffic, skewed toward NZ/Australia/Pacific given that's
// this deployment's home region. A callsign whose leading 3 letters aren't a
// recognized designator (private/GA aircraft usually just broadcast their own
// registration as the callsign, e.g. "N12345") falls through to no airline at all
// rather than guessing -- see airlineForFlight()'s null return, and popupHtml, which
// omits the whole line rather than showing a placeholder for the (common) GA case.
const AIRLINE_REGISTER = {
    // New Zealand / Australia / Pacific
    ANZ: 'Air New Zealand', QFA: 'Qantas', JST: 'Jetstar', VOZ: 'Virgin Australia',

    // North America
    AAL: 'American Airlines', UAL: 'United Airlines', DAL: 'Delta Air Lines',
    SWA: 'Southwest Airlines', JBU: 'JetBlue Airways', ASA: 'Alaska Airlines',
    FFT: 'Frontier Airlines', NKS: 'Spirit Airlines', ACA: 'Air Canada', WJA: 'WestJet',

    // Europe
    BAW: 'British Airways', VIR: 'Virgin Atlantic', EZY: 'easyJet', RYR: 'Ryanair',
    AFR: 'Air France', DLH: 'Lufthansa', KLM: 'KLM Royal Dutch Airlines', IBE: 'Iberia',
    SAS: 'SAS Scandinavian Airlines', FIN: 'Finnair', SWR: 'Swiss International Air Lines',
    THY: 'Turkish Airlines',

    // Middle East
    UAE: 'Emirates', QTR: 'Qatar Airways', ETD: 'Etihad Airways', SVA: 'Saudia',

    // Asia
    SIA: 'Singapore Airlines', CPA: 'Cathay Pacific', ANA: 'All Nippon Airways',
    JAL: 'Japan Airlines', KAL: 'Korean Air', AAR: 'Asiana Airlines', CCA: 'Air China',
    CES: 'China Eastern Airlines', CSN: 'China Southern Airlines', THA: 'Thai Airways',
    MAS: 'Malaysia Airlines', GIA: 'Garuda Indonesia', PAL: 'Philippine Airlines',
    AIC: 'Air India', IGO: 'IndiGo',

    // Africa
    ETH: 'Ethiopian Airlines', SAA: 'South African Airways', MSR: 'EgyptAir',

    // South America
    LAN: 'LATAM Airlines', TAM: 'LATAM Airlines', GLO: 'Gol Linhas Aereas', AVA: 'Avianca',

    // Cargo
    FDX: 'FedEx Express', UPS: 'UPS Airlines', GTI: 'Atlas Air',
};

export function airlineForFlight(flightCallsign) {
    if (!flightCallsign) return null;
    return AIRLINE_REGISTER[flightCallsign.trim().slice(0, 3).toUpperCase()] || null;
}

// ---------------------------------------------------------------------------------
// Layer wiring: periodic REST poll, requestAnimationFrame render loop, MapLibre
// filters/icons, hover popup + hover-track (issue #215, docs/adr/0010 -- supersedes
// the WebSocket-push client this replaced). Not unit-tested (same boundary every
// other layer module in this codebase draws: pure math/config-mapping gets tests --
// buildLUT/speedFromConfig in jetstream.js, interpolatedPosition/boundedElapsedSeconds
// above -- the DOM/network/map glue below is verified live instead, same as
// mount/refresh/unmount in shipping.js/markers.js/satellites.js).
// ---------------------------------------------------------------------------------
import { liveDataSync } from './_datasync.js';
import { hoverPopup } from './_hoverpopup.js';
import { fetchOrThrow, preloadIcons, escapeHtml, buildPopupHtml } from './_feedhelpers.js';

// Deliberately tightened below flightradar_collector.starvation_floor_minutes
// (default 30 min, AircraftCollector's cache-warming sweep's worst-case gap for a
// background/non-hotspot cell) -- a lower prune threshold means an aircraft outside
// wherever the viewport's own hot cells currently are can occasionally get pruned
// and briefly disappear before its next scheduled background resample brings it
// back, rather than sitting frozen on the map for up to 40 minutes. Traded off
// against exactly that decluttering benefit (issue: "way too many aircraft,
// particularly in dense regions like Europe") -- an aircraft actually being looked
// at sits in a hot cell refreshed on a ~60s cycle regardless, so this mostly thins
// out stale aircraft the user isn't currently focused on. isFrozen's much shorter
// MAX_EXTRAPOLATION_S (60s) is what shows the "signal lost" cue in the meantime.
const STALE_PRUNE_MS = 10 * 60 * 1000;

// How often the frontend polls GET /api/flightradar/geojson. Independent of
// AircraftCollector's own adsb.lol request budget -- this is a local DB read, cheap
// regardless of poll rate -- chosen only for how smooth the render loop should feel
// between dead-reckoning interpolation steps.
const POLL_INTERVAL_MS = 3000;

// docs/adr/0008: alt_baro (not category) drives the zoom-density filter -- high-
// altitude traffic is visible zoomed out, low-altitude/ground traffic only reveals
// once zoomed in close, mirroring shipping.js's length-based step filter. Feet.
const ALT_ZOOM_STEP = ['step', ['zoom'], 30000, 4, 20000, 5, 10000, 6, 3000, 7, 500, 8, 0];

// Nose points north (track=0), matching icon-rotate's rotation-from-north semantics.
// docs/adr/0008: B* categories (gliders/balloons/drones) get distinct treatment
// (aircraft_light.png) rather than being folded into the generic aircraft icon.
// Both are plain white silhouettes registered as SDF (sdf: true) so the layer can tint
// them per aircraftGroupColor() at render time -- one shape asset per icon covers
// every color group, rather than needing a separately-baked PNG per group per shape.
const FLIGHTRADAR_ICONS = [
    { id: 'flightradar-aircraft', url: '/images/aircraft_generic.png', sdf: true },
    { id: 'flightradar-glider', url: '/images/aircraft_light.png', sdf: true },
];

// docs/adr/0008: category C* (ground vehicles/obstacles) is filtered out entirely --
// done here at feature-build time rather than as a MapLibre style filter, so the style
// itself only needs the altitude density step.
//
// Orchestrates dead-reckoning (interpolatedPosition/extrapolatedAltitude) and, when
// there's a prior displayed state to ease from, correction-smoothing on top
// (smoothedPosition/bearingDeg/smoothedAngle/smoothedScalar, plus isBackwardCorrection's
// hold-in-place special case) -- everything buildFeatureCollection needs to know about
// ONE aircraft's next {pos, altBaroFt, iconTrack}. Kept separate from GeoJSON assembly
// so this orchestration -- the source of most of this file's real, live-caught bugs
// (the icon twitch, the backward slide on final approach, the sideways takeoff roll) --
// has its own direct test surface instead of being read back out of a Feature's
// geometry/properties.
//
// prevDisplay=null/undefined (a genuine first sighting, or the caller opting out of
// smoothing entirely by never populating a displayByHex map) skips easing: pos/altBaroFt
// come back exactly as dead-reckoned, iconTrack falls back to the raw reported track.
//
// deadReckonGs/deadReckonBaroRate (recordFromFeature) fall back to the raw gs/baro_rate
// for any caller/test that doesn't set them -- only recordFromFeature actually forces
// them to 0 (see its docstring for the stalled-position case).
export function deriveDisplayState(rec, elapsed, prevDisplay, dtS, tauS) {
    const target = interpolatedPosition(
        { lat: rec.lat, lon: rec.lon, gs: rec.deadReckonGs ?? rec.gs, track: rec.track }, elapsed,
    );
    const rawAltBaroFt = extrapolatedAltitude(
        rec.alt_baro, rec.deadReckonBaroRate ?? rec.baro_rate, rec.nav_altitude_mcp, elapsed,
    );

    if (!prevDisplay) {
        // Nothing to ease from yet, render directly. iconTrack defaults to 0 (north),
        // not null/undefined, specifically so every later frame (which WILL have a
        // prevDisplay) always finds a real number in prevDisplay.track below.
        return { pos: target, altBaroFt: rawAltBaroFt, iconTrack: rec.track ?? 0 };
    }

    const pos = isBackwardCorrection(prevDisplay, target, rec.track)
        ? prevDisplay   // hold -- never visibly fly backward (see isBackwardCorrection)
        : smoothedPosition(prevDisplay, target, dtS, tauS);

    const bearing = bearingDeg({ lat: prevDisplay.lat, lon: prevDisplay.lon }, pos);
    // No movement this frame (held, or genuinely stationary) -- keep pointing the way
    // it was already pointing rather than snapping to an arbitrary angle for a
    // zero-length vector.
    const rawIconTrack = bearing != null ? bearing : (prevDisplay.track ?? rec.track ?? 0);
    // Always rate-capped (see MAX_ICON_TURN_RATE_DEG_S) once there's a PRIOR FRAME to
    // ease from at all -- deliberately NOT gated on whether prevDisplay.track happens
    // to already be a number. adsb.lol's `track` is null for essentially an entire
    // ground phase in real captured data (ADS-B ground track is undefined/unbroadcast
    // near-stationary, e.g. at the gate or early taxi) -- with the old "no prior
    // number, skip smoothing" rule, an aircraft could sit at track=null (nothing to
    // derive a bearing from either, no movement yet) for the whole taxi phase, then
    // the FIRST real bearing/track reading once it finally starts moving -- exactly
    // the noisy, unreliable one the original twitch fix targeted -- rendered
    // instantly, fully unsmoothed, because that "first known" branch treated it as
    // indistinguishable from a genuine first sighting. Caught live: an aircraft
    // rendering "sideways" through its takeoff roll, straightening only once
    // airborne. The no-prevDisplay branch above's real-number default (0, not null)
    // guarantees this branch can never be bypassed again after the first frame.
    const iconTrack = smoothedAngle(prevDisplay.track ?? 0, rawIconTrack, dtS, tauS, MAX_ICON_TURN_RATE_DEG_S);

    // Altitude only eases when it's a genuine number -- 'ground'/unknown has nothing
    // meaningful to ease toward or from, and popupHtml's on_ground/alt_baro_known
    // already read the raw rec directly.
    let altBaroFt = rawAltBaroFt;
    if (typeof rawAltBaroFt === 'number') {
        const prevAlt = typeof prevDisplay.alt === 'number'
            ? prevDisplay.alt : rawAltBaroFt;   // no lag if altitude just became known
        altBaroFt = smoothedScalar(prevAlt, rawAltBaroFt, dtS, tauS, MAX_ALT_CORRECTION_FPM / 60);
    }

    return { pos, altBaroFt, iconTrack };
}

// displayByHex/dtS/tauS (all optional) opt into deriveDisplayState's correction
// smoothing: when displayByHex is given, each hex's previous displayed state is looked
// up and handed to deriveDisplayState, then the result is written back so the NEXT call
// eases from here. displayByHex=null (the default) means every prevDisplay is
// undefined, so deriveDisplayState's no-prior-state branch runs for every record --
// every pre-existing caller/test gets the exact raw dead-reckoned target, unchanged.
export function buildFeatureCollection(aircraftByHex, now, displayByHex = null, dtS = 0, tauS = SMOOTH_TAU_S) {
    const features = [];
    for (const rec of aircraftByHex.values()) {
        if (typeof rec.lat !== 'number' || typeof rec.lon !== 'number') continue;
        const category = (rec.category || '').toUpperCase();
        if (category.startsWith('C')) continue;
        const elapsed = boundedElapsedSeconds(rec.receivedAt, now);
        const prevDisplay = displayByHex ? displayByHex.get(rec.hex) : undefined;
        const { pos, altBaroFt, iconTrack } = deriveDisplayState(rec, elapsed, prevDisplay, dtS, tauS);

        if (displayByHex) {
            displayByHex.set(rec.hex, {
                ...pos,
                ...(typeof altBaroFt === 'number' ? { alt: altBaroFt } : {}),
                track: iconTrack,
            });
        }
        features.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [pos.lon, pos.lat] },
            properties: {
                hex: rec.hex,
                flight: (rec.flight || '').trim() || rec.hex,
                // Derived from the raw callsign, before the hex fallback above --
                // airlineForFlight() needs the real callsign (or nothing), not a hex
                // ICAO address standing in for a missing one.
                airline: airlineForFlight(rec.flight),
                registration: rec.r || '',
                aircraft_type: rec.t || '',
                // alt_baro_ft stays numeric (0 fallback) purely for the MapLibre zoom-density
                // filter above, which needs a number to compare against. The real reading is
                // alt_baro_known/on_ground -- adsb.lol's alt_baro is either a number, the
                // literal string "ground" (on-ground transponder state, unrelated to this
                // value), or simply absent (no reading yet) -- three states popupHtml must
                // not conflate (see its "ground" vs "unknown" altitude display). Dead-reckoned
                // (extrapolatedAltitude) when it's a genuine number, same as position.
                alt_baro_ft: typeof altBaroFt === 'number' ? altBaroFt : 0,
                alt_baro_known: typeof rec.alt_baro === 'number',
                on_ground: rec.alt_baro === 'ground',
                // adsb.lol's real reported rate wins when present (more instantaneous
                // than a 2-sample derivative); derivedBaroRateFpm (recordFromFeature)
                // fills in only when it's missing -- see that field's own comment.
                baro_rate_fpm: typeof rec.baro_rate === 'number' ? rec.baro_rate
                    : typeof rec.derivedBaroRateFpm === 'number' ? rec.derivedBaroRateFpm
                    : null,
                nav_altitude_mcp_ft: typeof rec.nav_altitude_mcp === 'number' ? rec.nav_altitude_mcp : null,
                gs: rec.gs ?? 0,
                // track is the raw reported ground track (popupHtml's "Heading" line
                // and flightStatus/etc. want the real value); icon_track is what the
                // layer's icon-rotate paint property actually uses -- derived from
                // the icon's own rendered movement (see bearingDeg), which can
                // legitimately differ from raw track during a crosswind crab angle or
                // while our own extrapolation/smoothing is mid-correction.
                track: rec.track ?? 0,
                icon_track: iconTrack ?? 0,
                icon: category.startsWith('B') ? 'flightradar-glider' : 'flightradar-aircraft',
                color: aircraftGroupColor(rec.t),
                frozen: isFrozen(rec.receivedAt, now),
                // JSON-stringified, not a bare array/object: MapLibre tiles every
                // GeoJSON source internally (geojson-vt) for rendering, and that
                // pipeline doesn't safely round-trip non-primitive property values --
                // a nested array here broke hover/mouseenter feature-querying for the
                // WHOLE layer, not just the aircraft carrying a route, when verified
                // live. Every property handed to setData() must stay a primitive
                // (string/number/boolean/null); popupHtml parses this back out via
                // parseRouteStops(). route_plausible (bool/null) is already primitive.
                route_stops: rec.route_stops ? JSON.stringify(rec.route_stops) : null,
                route_plausible: rec.route_plausible ?? null,
            },
        });
    }
    return { type: 'FeatureCollection', features };
}

function pruneStale(aircraftByHex, now) {
    for (const [hex, rec] of aircraftByHex) {
        if (now - rec.receivedAt > STALE_PRUNE_MS) aircraftByHex.delete(hex);
    }
}

// Delegates to the shared escapeHtml (safe in an attribute context too -- it escapes
// a strict superset of what an attribute value needs). Was previously '"'-only,
// missing '&'/'<'/'>'/"'" -- see escapeHtml's comment in _feedhelpers.js for why
// route/airport names (this app's most attacker-reachable route metadata besides the
// flight/callsign fields below) need full entity escaping, not just the one character
// that happened to break out of the surrounding double-quoted attribute.
function _escapeAttr(s) {
    return escapeHtml(s);
}

// IATA preferred over ICAO (this feature's own worked examples throughout --
// "WLG"/"AKL" -- are IATA-style, the codes people actually recognize), falling back
// to ICAO only when a stop has no IATA code (some smaller airports are ICAO-only).
export function stopCode(stop) {
    return stop.iata || stop.icao || '';
}

// route_stops -> an arrow-joined path covering the FULL route (any technical/
// intermediate stop(s) included, not collapsed to just origin/destination -- that
// detail is genuinely worth keeping visible, not just stored). Each code carries a
// title tooltip with that stop's full airport name. Returns '' for an empty/missing
// list -- popupHtml hides the whole route block (and its separating <hr>) in that
// case, same as every other optional field on this popup.
export function routePathHtml(stops) {
    if (!stops || !stops.length) return '';
    return stops
        .map((s) => `<span title="${_escapeAttr(s.name || '')}">${escapeHtml(stopCode(s))}</span>`)
        .join(' &rarr; ');
}

// A small, separate warning glyph -- not a change to the route line's own color/
// weight, which stays prominent regardless -- shown only when adsb.lol's own
// great-circle sanity check says this match doesn't fit the aircraft's actual
// current position. Nothing shown for a genuine match (true) or when there's no
// route to judge in the first place (null).
export function plausibleWarningHtml(plausible) {
    if (plausible !== false) return '';
    return ' <span title="This route may not match the aircraft&#39;s current position" style="cursor:help;">&#9888;</span>';
}

// buildFeatureCollection JSON-stringifies route_stops before it ever reaches
// MapLibre's setData() (see that function's comment) -- this undoes it for display.
// Malformed/unparseable input (shouldn't happen, but a hover mid-poll-update is
// timing-sensitive) is treated the same as "no route" rather than throwing.
export function parseRouteStops(raw) {
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch (e) {
        return null;
    }
}

function popupHtml(f) {
    const p = f.properties;
    // On the ground but still moving: taxiing, just-landed rollout, and takeoff roll
    // all look identical from ADS-B alone -- see flightStatus/targetAltitudeLabel.
    const groundAmbiguous = p.on_ground && p.gs > 0;
    const alt = p.on_ground ? 'ground'
        : p.alt_baro_known ? `${Math.round(p.alt_baro_ft).toLocaleString()} ft`
        : 'unknown';
    const status = flightStatus(p.on_ground, p.gs, p.baro_rate_fpm);
    const target = targetAltitudeLabel(p.nav_altitude_mcp_ft, p.alt_baro_ft, groundAmbiguous);
    const cls = aircraftClass(p.aircraft_type);
    // route_stops is null both when this callsign has never been enriched yet AND
    // when adsb.lol confirmed it has no route (the API can't distinguish the two --
    // see FlightRoute's own docstring) -- either way, the route emphasis block is
    // simply omitted rather than showing a placeholder.
    const routePath = routePathHtml(parseRouteStops(p.route_stops));

    // flight/aircraft_type/registration are raw ADS-B transponder fields (self-reported,
    // no validation, same attacker-controlled-free-text class as AIS -- see escapeHtml's
    // comment in _feedhelpers.js). airline/cls/hex are derived/lookup values, but escaped
    // too rather than trusting their fallback paths never echo the raw input back out.
    // status/alt/target/speed/heading are computed labels, not feed data -- raw:true
    // preserves their original unescaped rendering (heading's `&deg;` entity would be
    // double-encoded otherwise).
    const blocks = [];
    if (routePath) {
        blocks.push({ type: 'emphasis', html: routePath + plausibleWarningHtml(p.route_plausible) });
        blocks.push({ type: 'divider' });
    }
    if (p.aircraft_type) blocks.push({ type: 'line', items: [{ label: 'Type', value: p.aircraft_type }] });
    blocks.push({ type: 'line', items: [{ label: 'Class', value: cls }] });
    if (p.airline) blocks.push({ type: 'line', items: [{ label: 'Airline', value: p.airline }] });
    if (p.registration) blocks.push({ type: 'line', items: [{ label: 'Registration', value: p.registration }] });
    if (status) blocks.push({ type: 'line', items: [{ label: 'Status', value: status, raw: true }] });
    blocks.push({ type: 'line', items: [{ label: 'Altitude', value: alt, raw: true }] });
    if (target) blocks.push({ type: 'line', items: [{ label: 'Target altitude', value: target, raw: true }] });
    blocks.push({ type: 'line', items: [{ label: 'Speed', value: `${Math.round(p.gs)} kts`, raw: true }] });
    blocks.push({ type: 'line', items: [{ label: 'Heading', value: `${Math.round(p.track)}&deg;`, raw: true }] });
    blocks.push({ type: 'line', items: [{ label: 'ICAO', value: p.hex }] });
    if (p.frozen) {
        blocks.push({ type: 'notice', text: '&#9888; Signal lost -- position frozen', raw: true });
    }

    return buildPopupHtml({
        title: { text: p.flight, variant: 'callsign' },
        blocks,
    });
}

// One id per page load, sent on every poll so AircraftCollector's aircraft_interest
// table can tell this viewer's viewport apart from any other concurrent session
// (issue #215's per-viewer, not single-global, hotspot signal).
const viewerId = (typeof crypto !== 'undefined' && crypto.randomUUID)
    ? crypto.randomUUID()
    : `viewer-${Date.now()}-${Math.random().toString(36).slice(2)}`;

// A GeoJSON feature from GET /api/flightradar/geojson -> the same raw-record shape
// buildFeatureCollection/pruneStale already expect (adsb.lol's own field names),
// so those two pure functions need no changes for the WS->poll transport swap.
// receivedAt is the SERVER's last_seen timestamp, not "when this poll happened to
// include the row" -- the endpoint returns just the current viewport's aircraft (see
// AircraftAdapter.get_fleet_as_geojson's bbox filter), so mere presence in a response
// is no longer evidence of a fresh sighting the way a WS push used to be.
// Real ADS-B position (lat/lon + barometric altitude) and velocity (gs/track/
// baro_rate) are broadcast as INDEPENDENT message types on their own schedules, not
// bundled together. Close to the ground -- final approach, rollout, taxi -- position
// updates can stall for several consecutive real samples (caught live: identical lat/
// lon and alt_baro_ft across 3-5 samples spanning 40+ seconds) while velocity keeps
// reporting normally throughout. Left alone, dead reckoning (interpolatedPosition/
// extrapolatedAltitude) then extrapolates FURTHER forward every frame using a
// genuinely-current speed/rate from a position anchor that hasn't actually moved --
// this is the real root cause of "dead reckoning should be ~100% accurate at
// near-constant landing speed, but it's moving faster than it should be": we're not
// modelling deceleration wrong, we're extrapolating from a stale anchor with a fresh
// velocity, which double-counts distance/altitude change the aircraft already covered
// during the stall.
//
// isNewSample gates this to genuinely DISTINCT upstream reports (by receivedAt) --
// the 3s frontend poll runs faster than the backend's own ~11-13s real refresh, so
// most polls simply re-fetch the same sample, which must not be misread as a stall.
export function recordFromFeature(feature, fallbackNowMs, prevRec) {
    const p = feature.properties;
    const [lon, lat] = feature.geometry.coordinates;
    const receivedAt = Date.parse(p.last_seen);
    const alt_baro = p.on_ground ? 'ground' : p.alt_baro_ft;

    const isNewSample = !!prevRec && prevRec.receivedAt !== receivedAt;
    const positionStalled = isNewSample && prevRec.lat === lat && prevRec.lon === lon;
    const altitudeStalled = isNewSample && typeof alt_baro === 'number' && prevRec.alt_baro === alt_baro;

    // Fallback vertical rate for flightStatus(): adsb.lol's own baro_rate can drop out
    // for a poll or two -- observed live on final approach and climb-out, exactly when
    // a vertical-rate reading matters most -- leaving flightStatus with nothing to
    // reason about and stuck showing "Level flight" even while alt_baro is visibly
    // moving between real samples. Derived the same way positionStalled/altitudeStalled
    // detect a stall: the change in alt_baro across the last two DISTINCT real samples,
    // over the real elapsed time between them (not the dead-reckoned/display elapsed).
    // Only meaningful when both samples are genuine numeric altitudes (airborne on
    // both) -- matches baro_rate's own "n/a on the ground" contract. Carried forward
    // unchanged across repeated polls of the same backend sample (isNewSample false),
    // so the status doesn't flicker back to "Level flight" between real refreshes.
    let derivedBaroRateFpm = prevRec ? prevRec.derivedBaroRateFpm ?? null : null;
    if (isNewSample && typeof alt_baro === 'number' && typeof prevRec.alt_baro === 'number') {
        const elapsedMin = (receivedAt - prevRec.receivedAt) / 60000;
        derivedBaroRateFpm = elapsedMin > 0 ? (alt_baro - prevRec.alt_baro) / elapsedMin : null;
    }

    return {
        hex: p.hex,
        flight: p.flight,
        r: p.registration,
        t: p.aircraft_type,
        category: p.category,
        alt_baro,
        gs: p.gs,
        // Real reported speed (gs, above) stays intact for the popup; deadReckonGs is
        // what interpolatedPosition actually extrapolates with -- forced to 0 (no
        // movement) when the raw position itself hasn't changed since the last
        // distinct real sample. See the module-level comment on positionStalled's
        // reasoning above.
        deadReckonGs: positionStalled ? 0 : p.gs,
        track: p.track,
        baro_rate: p.baro_rate,
        // Same idea vertically: extrapolatedAltitude's baro_rate projection is
        // suppressed when the raw altitude itself hasn't moved since the last
        // distinct real sample (only meaningful once alt_baro is a genuine number --
        // 'ground' has no altitude to compare).
        deadReckonBaroRate: altitudeStalled ? 0 : p.baro_rate,
        derivedBaroRateFpm,
        nav_altitude_mcp: p.nav_altitude_mcp,
        lat, lon,
        receivedAt: Number.isNaN(receivedAt) ? fallbackNowMs : receivedAt,
        // Carried through as-is (no dead-reckoning/transformation needed, unlike
        // position/altitude) -- see AircraftAdapter.get_fleet_as_geojson's route join.
        route_stops: p.route_stops || null,
        route_plausible: p.route_plausible ?? null,
    };
}

export function loadLayer(map, config) {
    const sourceId = 'flightradar-source';
    const layerId = 'flightradar-layer';
    const trackSourceId = 'flightradar-track-source';
    const trackLayerId = 'flightradar-track-layer';

    const aircraftByHex = new Map();   // hex -> {...adsb.lol-shaped record fields, receivedAt}
    // hex -> the last-rendered {lat, lon} -- correction-smoothing state (see
    // smoothedPosition/buildFeatureCollection), separate from aircraftByHex's raw
    // server-reported state so a new real sample's target can be eased into rather
    // than snapped to.
    const displayByHex = new Map();
    let currentCfg = config;
    let pollTimer = null;
    let rafId = null;
    let lastFrameMs = null;
    let stopPopup = null;
    let hoveredHex = null;

    const geojsonUrlFor = () => {
        const b = map.getBounds();
        const params = new URLSearchParams({
            west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth(),
            viewer_id: viewerId, t: Date.now(),
        });
        return `${window.WM_API}/flightradar/geojson?${params}`;
    };

    // The read request IS the interest signal (docs/adr/0010) -- the bbox this fetch
    // sends is what AircraftCollector reads back out of aircraft_interest, no separate
    // "tell the server where I'm looking" call.
    const pollOnce = async () => {
        let geojson;
        try {
            geojson = await fetchOrThrow(geojsonUrlFor());
        } catch (err) {
            console.warn('[flightradar] poll failed', err);
            return;
        }
        const now = Date.now();
        for (const feature of geojson.features || []) {
            const prevRec = feature.properties?.hex ? aircraftByHex.get(feature.properties.hex) : undefined;
            const rec = recordFromFeature(feature, now, prevRec);
            if (!rec.hex) continue;
            // Mirrors shipping.js's own age-filter-on-fetch (max_age_days) -- a row
            // the collector hasn't refreshed recently enough is left out of this
            // merge entirely rather than resetting its receivedAt to "now".
            if (now - rec.receivedAt > STALE_PRUNE_MS) continue;
            aircraftByHex.set(rec.hex, rec);
        }
    };

    const renderFrame = () => {
        const now = Date.now();
        const dtS = lastFrameMs != null ? Math.max(0, (now - lastFrameMs) / 1000) : 0;
        lastFrameMs = now;
        pruneStale(aircraftByHex, now);
        // displayByHex must never carry a smoothing anchor for an aircraft that's no
        // longer tracked -- otherwise a hex pruned and later re-added would ease in
        // from a stale, unrelated position instead of appearing directly at its
        // fresh one (buildFeatureCollection already handles that "first sighting"
        // case correctly, but only if there's truly no leftover entry here).
        for (const hex of displayByHex.keys()) {
            if (!aircraftByHex.has(hex)) displayByHex.delete(hex);
        }
        map.getSource(sourceId)?.setData(buildFeatureCollection(aircraftByHex, now, displayByHex, dtS));
        rafId = requestAnimationFrame(renderFrame);
    };

    // Hover-only track (flightradar.view_tracks/track_limit/track_color): empty until
    // an aircraft is hovered, cleared again on mouseleave -- same shape as shipping.js's
    // hover-track (see docs/adr/0002 for why this stays bespoke rather than widening
    // hoverPopup's contract).
    const emptyTrack = () => ({ type: 'FeatureCollection', features: [] });

    const trackUrlFor = (hex, limit) =>
        `${window.WM_API}/flightradar/${hex}/track?limit=${limit}&t=${Date.now()}`;

    const showTrack = async (hex) => {
        hoveredHex = hex;
        let points;
        try {
            const resp = await fetchOrThrow(trackUrlFor(hex, currentCfg.track_limit || 50));
            points = resp.data || [];
        } catch (err) {
            console.warn(`[flightradar] track fetch failed for ${hex}`, err);
            return;
        }
        // The hover may have moved to a different aircraft (or left entirely) while
        // this request was in flight -- a stale response must not overwrite what's shown.
        if (hoveredHex !== hex) return;
        if (!map.getSource(trackSourceId)) return;

        // newest-first from the API -- reverse so the line is drawn oldest -> newest.
        const coords = points.slice().reverse().map(p => [p.lon, p.lat]);
        map.getSource(trackSourceId).setData(coords.length >= 2
            ? { type: 'FeatureCollection', features: [
                { type: 'Feature', geometry: { type: 'LineString', coordinates: coords }, properties: {} },
            ] }
            : emptyTrack());
    };

    const clearTrack = () => {
        hoveredHex = null;
        map.getSource(trackSourceId)?.setData(emptyTrack());
    };

    const onTrackEnter = (e) => {
        if (!currentCfg.view_tracks || !e.features.length) return;
        const hex = e.features[0].properties.hex;
        if (hex) showTrack(hex);
    };
    const onTrackLeave = () => clearTrack();

    const mount = async (cfg) => {
        currentCfg = cfg;
        if (map.getSource(sourceId)) return;
        await preloadIcons(map, FLIGHTRADAR_ICONS);

        map.addSource(sourceId, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        map.addLayer({
            id: layerId, type: 'symbol', source: sourceId,
            filter: ['>=', ['get', 'alt_baro_ft'], ALT_ZOOM_STEP],
            layout: {
                'icon-image': ['get', 'icon'],
                'icon-size': 0.5 * (cfg.icon_zoom ?? 1.0),
                'icon-rotate': ['get', 'icon_track'],
                'icon-rotation-alignment': 'map',
                // true/true: every aircraft always renders regardless of visual
                // overlap. MapLibre's collision-based decluttering (false/false) was
                // tried to help dense regions like Europe, but produced its own
                // problems worse than the clutter it fixed -- a long-standing
                // MapLibre/Mapbox flicker bug on frequent setData() calls
                // (mapbox-gl-js#6052) that a render throttle + stable sort-key could
                // only partially mitigate, plus an artificial, grid-like spacing
                // pattern from the collision algorithm's own minimum-distance
                // enforcement. Reverted; if regional density needs addressing again,
                // real point-clustering (MapLibre's native cluster:true, a "N
                // aircraft" bubble that expands on zoom) is the next thing to try --
                // it doesn't fight per-aircraft icon rotation/color the way
                // overlap-collision does.
                'icon-allow-overlap': true, 'icon-ignore-placement': true,
            },
            // Tints the SDF icon per aircraftGroupColor() -- see FLIGHTRADAR_ICONS.
            paint: {
                'icon-color': ['get', 'color'],
            },
        });

        // Added BEFORE layerId so the line renders underneath the aircraft icons.
        map.addSource(trackSourceId, { type: 'geojson', data: emptyTrack() });
        map.addLayer({
            id: trackLayerId, type: 'line', source: trackSourceId,
            layout: { 'line-cap': 'round', 'line-join': 'round' },
            paint: { 'line-color': cfg.track_color || 'white', 'line-width': 2 },
        }, layerId);

        stopPopup = hoverPopup(map, layerId, { offset: 10, html: popupHtml });
        map.on('mouseenter', layerId, onTrackEnter);
        map.on('mouseleave', layerId, onTrackLeave);

        await pollOnce();
        pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS);
        rafId = requestAnimationFrame(renderFrame);
    };

    const refresh = async (cfg) => {
        currentCfg = cfg;
        if (map.getLayer(layerId)) {
            map.setLayoutProperty(layerId, 'icon-size', 0.5 * (cfg.icon_zoom ?? 1.0));
        }
        if (map.getLayer(trackLayerId)) {
            map.setPaintProperty(trackLayerId, 'line-color', cfg.track_color || 'white');
        }
    };

    const unmount = () => {
        if (pollTimer != null) clearInterval(pollTimer);
        pollTimer = null;
        if (rafId != null) cancelAnimationFrame(rafId);
        rafId = null;
        lastFrameMs = null;
        stopPopup?.();
        map.off('mouseenter', layerId, onTrackEnter);
        map.off('mouseleave', layerId, onTrackLeave);
        hoveredHex = null;
        aircraftByHex.clear();
        displayByHex.clear();
        if (map.getLayer(trackLayerId))   map.removeLayer(trackLayerId);
        if (map.getSource(trackSourceId)) map.removeSource(trackSourceId);
        if (map.getLayer(layerId)) map.removeLayer(layerId);
        if (map.getSource(sourceId)) map.removeSource(sourceId);
    };

    return liveDataSync(map, {
        sectionKey: 'flightradar', initialConfig: config,
        mount, refresh, unmount,
        refreshMs: 3600000,   // data arrives via POLL_INTERVAL_MS's own timer, not this -- see mount()
    });
}
