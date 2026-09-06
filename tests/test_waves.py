#!/usr/bin/env python3
"""Tests for WavesUpdater's land-mask wiring (the caching/coastline-cut logic itself
lives in LandMaskCache -- see tests/test_coastline.py for its own coverage; mirrors
test_currents.py's identical wiring test for CurrentsUpdater)."""
from unittest.mock import MagicMock, patch

import numpy as np

from atmos_gl.tasks.waves import WavesUpdater


def test_init_wires_a_land_mask_cache_labelled_waves():
    """Every LandMaskCache consumer (currents, waves) now shares one GSHHG 'h'
    geometry -- see docs/adr/0013 -- so there's no per-caller resolution to assert
    here anymore, just that WavesUpdater wires up its own labelled cache."""
    from atmos_gl.lib.coastline import LandMaskCache

    def fake_updater_init(self, config, section, map_data):
        self.section = section.lower()
        self.settings = {}

    with patch("atmos_gl.tasks.common.Updater.__init__", fake_updater_init):
        u = WavesUpdater(config=MagicMock(), map_data=MagicMock())

    assert isinstance(u._land_mask, LandMaskCache)
    assert u._land_mask._label == "Waves"


# ---- land mask is applied as-is (dilation, if any, is LandMaskCache/ ------------
# coastline_land_mask's own responsibility -- see tests/test_coastline.py's
# "coastal-bleed fix" tests) -- this just confirms _masked_uv() cuts land straight
# from whatever self._land_mask.get() returns, with no further processing of its own.

def test_masked_uv_masks_land_cells_as_nan_from_the_land_mask_as_is():
    land = np.array([[False, False, True], [False, False, True], [False, False, True]])
    u_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    v_arr = u_arr.copy()
    new_lats = np.array([10.0, 5.0, 0.0])
    new_lons = np.array([0.0, 5.0, 10.0])

    u = WavesUpdater.__new__(WavesUpdater)
    u.regrid_for_lod = MagicMock()
    u._land_mask = MagicMock()
    u._land_mask.get.return_value = land

    with patch(
        "atmos_gl.tasks.waves.nearest_fill_and_regrid_uv",
        return_value=(new_lats, new_lons, u_arr.copy(), v_arr.copy()),
    ):
        _, out_u, out_v = u._masked_uv({"u": u_arr, "v": v_arr, "lat": None, "lon": None})

    assert np.isnan(out_u[:, 2]).all()
    assert not np.isnan(out_u[:, :2]).any()
    assert np.isnan(out_v[:, 2]).all()
    assert not np.isnan(out_v[:, :2]).any()
