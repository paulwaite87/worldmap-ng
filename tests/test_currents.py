#!/usr/bin/env python3
"""Tests for CurrentsUpdater. The palette + legend key are entirely client-side now
(issue #302, see ui/modules/currents.js's own PALETTES/buildLUT) -- VectorFieldUpdater
no longer knows about palettes at all, so the coverage here is limited to what's still
actually on the backend: land mask wiring.
"""
from unittest.mock import MagicMock, patch

import numpy as np

from atmos_gl.tasks.currents import CurrentsUpdater


# ---- land mask wiring ------------------------------------------------------------
# The caching/coastline-cut logic itself moved to LandMaskCache (lib/coastline.py,
# shared with WavesUpdater -- see tests/test_coastline.py for its own coverage). This
# just confirms CurrentsUpdater wires one up correctly, not the logic again.

def test_init_wires_a_land_mask_cache_labelled_currents():
    from atmos_gl.lib.coastline import LandMaskCache

    def fake_updater_init(self, config, section, map_data):
        self.section = section.lower()
        self.settings = {}

    with patch("atmos_gl.tasks.common.Updater.__init__", fake_updater_init):
        u = CurrentsUpdater(config=MagicMock(), map_data=MagicMock())

    assert isinstance(u._land_mask, LandMaskCache)
    assert u._land_mask._label == "Currents"


# ---- land mask is applied as-is (dilation, if any, is LandMaskCache/ ------------
# coastline_land_mask's own responsibility -- see tests/test_coastline.py's
# "coastal-bleed fix" tests) -- this just confirms plot() cuts land straight from
# whatever self._land_mask.get() returns, with no further processing of its own.

def test_plot_masks_land_cells_as_nan_from_the_land_mask_as_is():
    land = np.array([[False, False, True], [False, False, True], [False, False, True]])
    u_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    v_arr = u_arr.copy()
    new_lats = np.array([10.0, 5.0, 0.0])
    new_lons = np.array([0.0, 5.0, 10.0])

    u = CurrentsUpdater.__new__(CurrentsUpdater)
    u.settings = {}
    u.regrid_for_lod = MagicMock()
    u._land_mask = MagicMock()
    u._land_mask.get.return_value = land
    u.get_output_path_for_hour = MagicMock(return_value="/data/currents_f003.png")

    with patch(
        "atmos_gl.tasks.currents.nearest_fill_and_regrid_uv",
        return_value=(new_lats, new_lons, u_arr.copy(), v_arr.copy()),
    ), patch("atmos_gl.tasks.currents.encode_uv") as m_encode:
        u.plot(field0={"u": u_arr, "v": v_arr, "lat": None, "lon": None}, state=MagicMock(fhour=3))

    encoded_u = m_encode.call_args.args[0]
    assert np.isnan(encoded_u[:, 2]).all()
    assert not np.isnan(encoded_u[:, :2]).any()
