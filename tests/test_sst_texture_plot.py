#!/usr/bin/env python3
"""Unit tests for SSTUpdater.plot()'s conversion to a raw client-LUT data texture
(issue #312) -- palette/scale settings must NOT affect the encoded texture's fixed
physical domain, only the live display range written to sst_meta.json for the
client-side LUT to remap onto. See ui/modules/sst.js's buildScaledLUT.
"""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np

from atmos_gl.tasks.sst import SSTUpdater, _ABS_ENCODE_VMIN, _ABS_ENCODE_VMAX, \
    _ANOMALY_ENCODE_VMIN, _ANOMALY_ENCODE_VMAX


def make_bare_updater(settings=None, output_path="/data/sst.png"):
    u = SSTUpdater.__new__(SSTUpdater)
    u.settings = settings or {}
    u.common = {}
    u.output_path = output_path
    u.section = "sst"
    u._write_meta_sidecar = MagicMock()
    return u


class _StackedMocks:
    """Patches everything plot() touches besides the mode/domain-selection logic
    under test, so plot() runs for real against small in-memory arrays. Returned
    from a context manager as a dict of the mocks, keyed by patch target name."""

    def __init__(self, mock_land):
        self.mock_land = mock_land
        self.stack = ExitStack()

    def __enter__(self):
        lat = np.array([0.0, 10.0])
        lon = np.array([0.0, 10.0])
        values = np.array([[5.0, 6.0], [7.0, 8.0]])

        class FakeSlice:
            def __getitem__(self, key):
                m = MagicMock()
                if key == "lat":
                    m.values = lat
                elif key == "lon":
                    m.values = lon
                else:
                    m.values = MagicMock(squeeze=lambda: values)
                return m

        ds = MagicMock()
        ds.isel.return_value = FakeSlice()

        mocks = {
            "open_dataset": self.stack.enter_context(
                patch("atmos_gl.tasks.sst.xr.open_dataset", return_value=ds)
            ),
            "distance_transform_edt": self.stack.enter_context(
                patch("atmos_gl.tasks.sst.distance_transform_edt",
                      side_effect=lambda bad, **kw: np.indices(bad.shape))
            ),
            "coastline_land_mask": self.stack.enter_context(
                patch("atmos_gl.tasks.sst.coastline_land_mask", return_value=self.mock_land)
            ),
            "encode_frames": self.stack.enter_context(
                patch("atmos_gl.tasks.sst.encode_frames", return_value=True)
            ),
        }
        return mocks

    def __exit__(self, *exc):
        self.stack.close()


_NEW_LATS = np.array([0.0, 5.0, 10.0])
_NEW_LONS = np.array([0.0, 5.0, 10.0])
_DISPLAY_DATA = np.array([[5.0, 5.5, 6.0], [6.0, 6.5, 7.0], [7.0, 7.5, 8.0]])


def test_plot_absolute_encodes_with_the_fixed_physical_domain_not_the_settings_range():
    u = make_bare_updater(settings={"min_c": 10, "max_c": 20})
    with _StackedMocks(mock_land=None) as mocks:
        u.regrid_for_lod = MagicMock(return_value=(_NEW_LATS, _NEW_LONS, _DISPLAY_DATA.copy()))
        u.plot("absolute", "/tmp/fake.nc", "/data/sst_absolute.png")

        mocks["encode_frames"].assert_called_once()
        args = mocks["encode_frames"].call_args.args
        assert args[1] == "/data/sst_absolute.png"
        assert args[2] == _ABS_ENCODE_VMIN
        assert args[3] == _ABS_ENCODE_VMAX


def test_plot_anomaly_encodes_with_the_fixed_physical_domain_not_the_calculated_range():
    u = make_bare_updater(settings={})
    with _StackedMocks(mock_land=None) as mocks:
        u.regrid_for_lod = MagicMock(return_value=(_NEW_LATS, _NEW_LONS, _DISPLAY_DATA.copy()))
        u.plot("anomaly", "/tmp/fake.nc", "/data/sst_anomaly.png")

        args = mocks["encode_frames"].call_args.args
        assert args[2] == _ANOMALY_ENCODE_VMIN
        assert args[3] == _ANOMALY_ENCODE_VMAX
        # The LIVE calculated display range still reaches the sidecar for the client LUT.
        u._write_meta_sidecar.assert_called_once()
        sidecar_value = u._write_meta_sidecar.call_args.args[2]
        assert sidecar_value["vmin"] == -sidecar_value["vmax"]


def test_plot_masks_land_cells_as_nan_before_encoding():
    # plot() cuts land straight from coastline_land_mask()'s own return value -- the
    # LINEAR-filter dilation fix now lives inside coastline_land_mask() itself (see
    # tests/test_coastline.py), not here, so this mock mask is used as-is.
    land = np.array([[False, False, True], [False, False, True], [False, False, True]])
    u = make_bare_updater(settings={})
    with _StackedMocks(mock_land=land) as mocks:
        u.regrid_for_lod = MagicMock(return_value=(_NEW_LATS, _NEW_LONS, _DISPLAY_DATA.copy()))
        u.plot("absolute", "/tmp/fake.nc", "/data/sst_absolute.png")

        encoded_frame = mocks["encode_frames"].call_args.args[0][0]
        assert np.isnan(encoded_frame[:, 2]).all()
        assert not np.isnan(encoded_frame[:, :2]).any()


def test_mode_settings_signature_is_empty_since_nothing_affects_the_encoded_texture():
    u = make_bare_updater(settings={"min_c": 5, "max_c": 30, "palette": "vivid", "opacity": 90})
    assert u._mode_settings_signature("absolute") == u._settings_signature({})
    assert u._mode_settings_signature("anomaly") == u._settings_signature({})
