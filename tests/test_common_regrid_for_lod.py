#!/usr/bin/env python3
"""Tests for Updater.regrid_for_lod, the shared LOD+interpolate helper absorbed from
the near-identical blocks in wind.py/ozone.py/temperature.py/precipitation.py/
stormwatch.py (see the architecture review's "Absorb the regrid/LOD block" candidate).

No longer clips to a bbox -- renders are always global now (regions are
reporting-only, see docs/adr/0004-render-bbox-clipping-is-dead-code.md), so the
resampled domain is simply the input field's own lat/lon span.
"""
import numpy as np

from atmos_gl.tasks import common
from atmos_gl.tasks.common import Updater, _MAX_LOD_GRID_POINTS


def make_bare_updater(level_of_detail=1):
    updater = Updater.__new__(Updater)
    updater.section = "test"
    updater.level_of_detail = level_of_detail
    updater.lod_desc = None
    return updater


def _linear_field(lats, lons):
    """field[i, j] = lats[i] + lons[j] -- linear, so bilinear interpolation reproduces
    it exactly at any point, making the resampled values easy to assert on."""
    return lats[:, None] + lons[None, :]


def test_regrid_for_lod_low_detail_step_and_desc():
    updater = make_bare_updater(level_of_detail=1)
    lats = np.arange(0.0, 6.0)
    lons = np.arange(0.0, 6.0)
    field = _linear_field(lats, lons)

    new_lats, new_lons, smooth = updater.regrid_for_lod(field, lats, lons)

    assert updater.lod_desc == "low"
    assert np.isclose(np.diff(new_lats)[0], 0.25)
    assert np.isclose(np.diff(new_lons)[0], 0.25)
    # bilinear interpolation of a linear field reproduces it exactly at grid points
    assert np.isclose(smooth[0, 0], new_lats[0] + new_lons[0])
    assert np.isclose(smooth[-1, -1], new_lats[-1] + new_lons[-1])


def test_regrid_for_lod_medium_detail():
    updater = make_bare_updater(level_of_detail=2)
    lats = np.arange(0.0, 6.0)
    lons = np.arange(0.0, 6.0)
    field = _linear_field(lats, lons)

    new_lats, new_lons, _ = updater.regrid_for_lod(field, lats, lons)

    assert updater.lod_desc == "medium"
    assert np.isclose(np.diff(new_lats)[0], 0.20)


def test_regrid_for_lod_high_detail():
    updater = make_bare_updater(level_of_detail=3)
    lats = np.arange(0.0, 6.0)
    lons = np.arange(0.0, 6.0)
    field = _linear_field(lats, lons)

    new_lats, new_lons, _ = updater.regrid_for_lod(field, lats, lons)

    assert updater.lod_desc == "high"
    assert np.isclose(np.diff(new_lats)[0], 0.15)


def test_regrid_for_lod_handles_descending_latitude_input():
    updater = make_bare_updater(level_of_detail=1)
    lats_desc = np.arange(5.0, -1.0, -1.0)  # [5, 4, 3, 2, 1, 0] -- north-first
    lons = np.arange(0.0, 6.0)
    field = _linear_field(lats_desc, lons)  # built consistently with lats_desc's order

    new_lats, new_lons, smooth = updater.regrid_for_lod(field, lats_desc, lons)

    # Regardless of input order, the returned grid is ascending and values line up
    assert new_lats[0] < new_lats[-1]
    assert np.isclose(smooth[0, 0], new_lats[0] + new_lons[0])
    assert np.isclose(smooth[-1, -1], new_lats[-1] + new_lons[-1])


def test_regrid_for_lod_world_view_stays_at_nominal_high_step():
    # The LOD step sizes are tuned for the DOMINANT case here (the frontend always
    # projects onto a MapLibre globe) -- a world-spanning field at "high" must land
    # comfortably under _MAX_LOD_GRID_POINTS on the nominal step, not rely on the cap
    # to bail it out every time.
    updater = make_bare_updater(level_of_detail=3)
    lats = np.arange(-90.0, 91.0, 1.0)  # coarse synthetic "global" input grid
    lons = np.arange(-180.0, 181.0, 1.0)
    field = _linear_field(lats, lons)

    new_lats, new_lons, smooth = updater.regrid_for_lod(field, lats, lons)

    assert updater.lod_desc == "high"
    step = new_lats[1] - new_lats[0]
    assert np.isclose(step, 0.15)  # nominal step, NOT scaled up by the cap
    assert len(new_lats) * len(new_lons) < _MAX_LOD_GRID_POINTS
    assert smooth.shape == (len(new_lats), len(new_lons))


def test_regrid_for_lod_small_region_unaffected_by_the_cap():
    # A small input field stays at the nominal step -- the cap only kicks in once
    # the grid would actually exceed the budget.
    updater = make_bare_updater(level_of_detail=3)
    lats = np.arange(0.0, 6.0)
    lons = np.arange(0.0, 6.0)
    field = _linear_field(lats, lons)

    new_lats, _, _ = updater.regrid_for_lod(field, lats, lons)

    assert np.isclose(new_lats[1] - new_lats[0], 0.15)


def test_regrid_for_lod_cap_is_still_a_backstop_beyond_world_scale(monkeypatch):
    # The cap isn't reachable via any real field this app produces (nothing renders
    # larger than the whole globe), so prove the mechanism itself still works by
    # lowering the budget artificially rather than constructing an impossible field.
    monkeypatch.setattr(common, "_MAX_LOD_GRID_POINTS", 1_000)
    updater = make_bare_updater(level_of_detail=3)
    lats = np.arange(-90.0, 91.0, 1.0)
    lons = np.arange(-180.0, 181.0, 1.0)
    field = _linear_field(lats, lons)

    new_lats, new_lons, _ = updater.regrid_for_lod(field, lats, lons)

    step = new_lats[1] - new_lats[0]
    assert step > 0.15  # scaled coarser than nominal to fit the shrunk budget
    # np.arange's actual point count isn't exactly the formula's estimate, so allow
    # some slack -- the key property is "much smaller than the unscaled grid", not an
    # exact bound.
    assert len(new_lats) * len(new_lons) <= 1_000 * 1.25


def test_regrid_for_lod_step_override_ignores_level_of_detail():
    # step_override takes a fixed step regardless of self.level_of_detail, and leaves
    # lod_desc unset -- none of "high"/"medium"/"low" describe a fixed step.
    updater = make_bare_updater(level_of_detail=1)
    lats = np.arange(0.0, 6.0)
    lons = np.arange(0.0, 6.0)
    field = _linear_field(lats, lons)

    new_lats, new_lons, _ = updater.regrid_for_lod(field, lats, lons, step_override=0.08)

    assert updater.lod_desc is None
    assert np.isclose(np.diff(new_lats)[0], 0.08)


def test_regrid_for_lod_step_override_bypasses_the_cap():
    # step_override is a deliberate, caller-chosen fixed resolution (e.g. SST's
    # coastline-crispness regrid) -- it must NOT be coarsened by _MAX_LOD_GRID_POINTS
    # the way the level_of_detail tiers are, even for a world-spanning field where a
    # fine step_override would vastly exceed the shared budget.
    updater = make_bare_updater(level_of_detail=3)
    lats = np.arange(-90.0, 91.0, 1.0)
    lons = np.arange(-180.0, 181.0, 1.0)
    field = _linear_field(lats, lons)

    new_lats, new_lons, _ = updater.regrid_for_lod(field, lats, lons, step_override=0.08)

    step = new_lats[1] - new_lats[0]
    assert np.isclose(step, 0.08)  # not scaled up despite exceeding _MAX_LOD_GRID_POINTS
    assert len(new_lats) * len(new_lons) > _MAX_LOD_GRID_POINTS


def test_regrid_for_lod_custom_fill_value_outside_domain():
    # step (0.25 at level_of_detail=1) doesn't evenly divide a 2.1-wide span, so
    # np.arange's last grid point genuinely overshoots the data's real max -> that
    # row/col is a true out-of-bounds query for the interpolator, not floating noise.
    updater = make_bare_updater(level_of_detail=1)
    lats = np.array([0.0, 0.7, 1.4, 2.1])
    lons = np.array([0.0, 0.7, 1.4, 2.1])
    field = _linear_field(lats, lons)

    _, _, smooth_default = updater.regrid_for_lod(field, lats, lons)
    assert np.isnan(smooth_default).any()  # default fill_value=np.nan leaves gaps as NaN

    updater2 = make_bare_updater(level_of_detail=1)
    _, _, smooth_zero = updater2.regrid_for_lod(field, lats, lons, fill_value=0)
    assert not np.isnan(smooth_zero).any()  # fill_value=0 -- no NaNs anywhere


# close_lon_seam_for_contour: the fix for the visible antimeridian seam in the
# isobars/precipitation/wind/scalar_field static renders (contour()/contourf() has no
# concept of a periodic domain, so a global grid's last and first lon columns -- though
# geographically adjacent -- are treated as the two dead-end edges of a plain
# rectangular grid unless a duplicate wrap column closes the loop first).
def test_close_lon_seam_for_contour_appends_a_duplicate_wrap_column_for_a_global_grid():
    updater = make_bare_updater()
    lons = np.arange(-180.0, 180.0, 0.25)  # native GFS-style span: -180..179.75
    field = np.tile(lons, (3, 1))  # field[:, j] == lons[j], so the wrap column is easy to check

    new_lons, new_field = updater.close_lon_seam_for_contour(lons, field)

    assert new_lons[-1] == 180.0
    assert new_field.shape == (3, len(lons) + 1)
    # The appended column is a duplicate of the first (lon=-180 and lon=180 are the
    # same meridian) -- this is exactly what closes the seam for contour/contourf.
    assert np.array_equal(new_field[:, -1], new_field[:, 0])


def test_close_lon_seam_for_contour_leaves_a_regional_grid_unchanged():
    updater = make_bare_updater()
    lons = np.arange(170.0, 180.0, 0.25)  # a real regional span, well under a full globe
    field = np.tile(lons, (3, 1))

    new_lons, new_field = updater.close_lon_seam_for_contour(lons, field)

    assert np.array_equal(new_lons, lons)
    assert np.array_equal(new_field, field)
