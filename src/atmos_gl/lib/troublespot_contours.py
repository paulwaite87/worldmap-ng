#!/usr/bin/env python3
"""Pure numpy/scipy/contourpy math behind the Troublespots layer (issue #366) --
turns a binned integer "how many distinct source types touch this cell" grid into
smooth severity-band polygons. No database, no HTTP: db/troublespot_adapter.py owns
the SQL binning step and calls compute_troublespot_bands() with the result.

MIN_CONVERGENCE_TYPES is the single source of truth for "how many distinct types must
converge to count as a troublespot at all" -- shared with lib/layer_availability.py's
extension (a Troublespots Show toggle can't ever produce a result with fewer than this
many of its source collectors enabled), so the two can never silently drift apart.
"""
import numpy as np
import contourpy
from scipy.ndimage import gaussian_filter

MIN_CONVERGENCE_TYPES = 2
MAX_CONVERGENCE_TYPES = 4  # size of the source roster (World Events, Earthquakes,
                           # Fires, Volcanic Activity) -- the ceiling a cell's type
                           # count can ever reach.

# Nested severity bands, in ascending order: "severe" is also "high" and "elevated" --
# only the highest applicable band needs to render at a given point (see the design's
# rendering decision). Each threshold is an exact >= comparison on the RAW (unsmoothed)
# integer type-count -- band membership itself has zero floating-point ambiguity.
# Public (no leading underscore): db/troublespot_adapter.py reuses this exact mapping
# for its breakdown step, rather than re-deriving a second copy of the same thresholds.
BAND_THRESHOLDS = (
    ("elevated", MIN_CONVERGENCE_TYPES),
    ("high", 3),
    ("severe", MAX_CONVERGENCE_TYPES),
)

# Smoothing is applied to each band's binary inclusion mask (0/1), never to the raw
# type-count values themselves -- smoothing the raw values would dilute a single
# isolated high-count cell below its own band's threshold (e.g. a lone cell where all
# 4 types converge could smooth down to a "high"-looking peak instead of "severe",
# silently breaking the "severe = all 4 types present" guarantee). Smoothing a binary
# mask and contouring at the 0.5 "majority-in" level keeps membership exact while still
# rounding the boundary's shape. 0.5 sigma is the largest value that still guarantees a
# single isolated True cell's smoothed peak clears the 0.5 contour level (verified
# empirically: sigma 0.6 already drops a lone cell's peak below 0.5).
_DEFAULT_MASK_SIGMA = 0.5


def compute_troublespot_bands(type_count_grid, lons, lats, smoothing_sigma=_DEFAULT_MASK_SIGMA):
    """type_count_grid: 2D array, shape (len(lats), len(lons)) -- integer count (0..4)
    of distinct source types present in each binned cell. lons/lats: ascending 1D cell-
    center coordinate arrays for the grid's axes.

    Returns a list of {"band": "elevated"|"high"|"severe", "rings": [[[lon, lat], ...],
    ...]} -- one entry per band with at least one qualifying polygon; a band with none
    is simply absent. Each ring is a polygon's exterior boundary as plain (float, float)
    pairs, ready to embed directly in GeoJSON. Interior holes are not represented --
    an accepted simplification for this feature (see the design's "Further Notes"):
    a troublespot with a hole in its middle is a rare edge case, not worth the added
    polygon-ring bookkeeping.
    """
    grid = np.asarray(type_count_grid, dtype=float)
    if grid.size and (grid.min() < 0 or grid.max() > MAX_CONVERGENCE_TYPES):
        raise ValueError(
            f"type_count_grid values must be within [0, {MAX_CONVERGENCE_TYPES}], "
            f"got range [{grid.min()}, {grid.max()}]"
        )

    lons_arr = np.asarray(lons, dtype=float)
    lats_arr = np.asarray(lats, dtype=float)

    bands = []
    for band_name, threshold in BAND_THRESHOLDS:
        mask = (grid >= threshold).astype(float)
        if not mask.any():
            continue

        smoothed = gaussian_filter(mask, sigma=smoothing_sigma)
        generator = contourpy.contour_generator(
            x=lons_arr, y=lats_arr, z=smoothed, name="serial", fill_type="OuterCode",
        )
        points, _codes = generator.filled(0.5, 2.0)
        rings = [
            [(float(lon), float(lat)) for lon, lat in polygon]
            for polygon in points
            if polygon is not None and len(polygon) >= 3
        ]
        if rings:
            bands.append({"band": band_name, "rings": rings})
    return bands
