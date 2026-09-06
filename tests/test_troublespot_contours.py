#!/usr/bin/env python3
"""Tests for lib/troublespot_contours.py -- the pure numpy/scipy/contourpy math behind
the Troublespots layer (issue #366): rasterize a binned integer type-count grid, smooth
it, and extract severity-band polygons. No database, no HTTP -- feed a synthetic grid,
assert the right band(s) come out roughly where expected, matching how this codebase
already tests other pure geometry helpers directly (e.g. tests/test_coastline.py).
"""
import numpy as np
import pytest

from atmos_gl.lib.troublespot_contours import (
    MIN_CONVERGENCE_TYPES,
    compute_troublespot_bands,
)


def _grid(n=20, block_value=0, block_slice=slice(8, 12)):
    grid = np.zeros((n, n), dtype=float)
    grid[block_slice, block_slice] = block_value
    lons = np.linspace(-10.0, 10.0, n)
    lats = np.linspace(-10.0, 10.0, n)
    return grid, lons, lats


def test_min_convergence_types_is_two():
    assert MIN_CONVERGENCE_TYPES == 2


def test_all_zero_grid_produces_no_bands():
    grid, lons, lats = _grid(block_value=0)
    assert compute_troublespot_bands(grid, lons, lats) == []


def test_a_uniform_block_below_the_minimum_produces_no_bands():
    # A block of 1 (below MIN_CONVERGENCE_TYPES=2) must never register as a troublespot,
    # regardless of how much area it covers -- convergence requires 2+ distinct types.
    grid, lons, lats = _grid(block_value=1)
    assert compute_troublespot_bands(grid, lons, lats) == []


def test_a_block_of_exactly_two_produces_only_an_elevated_band():
    grid, lons, lats = _grid(block_value=2)
    bands = compute_troublespot_bands(grid, lons, lats)
    band_names = {b["band"] for b in bands}
    assert "elevated" in band_names
    assert "high" not in band_names
    assert "severe" not in band_names


def test_a_block_of_exactly_four_produces_a_severe_band():
    grid, lons, lats = _grid(block_value=4)
    bands = compute_troublespot_bands(grid, lons, lats)
    band_names = {b["band"] for b in bands}
    assert "severe" in band_names


def test_severe_block_polygon_coordinates_fall_within_the_grid_bounds():
    grid, lons, lats = _grid(block_value=4)
    bands = compute_troublespot_bands(grid, lons, lats)
    severe = next(b for b in bands if b["band"] == "severe")
    for ring in severe["rings"]:
        for lon, lat in ring:
            assert lons.min() <= lon <= lons.max()
            assert lats.min() <= lat <= lats.max()


def test_severe_block_ring_is_roughly_centered_on_the_block():
    # Block occupies grid indices [8:12) of a 20-point axis spanning -10..10 -- centered
    # near index 10, i.e. lon/lat close to 0.
    grid, lons, lats = _grid(block_value=4)
    bands = compute_troublespot_bands(grid, lons, lats)
    severe = next(b for b in bands if b["band"] == "severe")
    all_lons = [lon for ring in severe["rings"] for lon, _ in ring]
    all_lats = [lat for ring in severe["rings"] for _, lat in ring]
    assert -5.0 < sum(all_lons) / len(all_lons) < 5.0
    assert -5.0 < sum(all_lats) / len(all_lats) < 5.0


def test_returned_rings_are_plain_json_serializable_floats():
    import json

    grid, lons, lats = _grid(block_value=4)
    bands = compute_troublespot_bands(grid, lons, lats)
    json.dumps(bands)  # must not raise -- no numpy scalars leaking through
    for band in bands:
        for ring in band["rings"]:
            for coord in ring:
                assert isinstance(coord[0], float)
                assert isinstance(coord[1], float)


def test_a_grid_with_two_disjoint_blocks_produces_two_severe_rings():
    n = 30
    grid = np.zeros((n, n), dtype=float)
    grid[3:7, 3:7] = 4
    grid[23:27, 23:27] = 4
    lons = np.linspace(-15.0, 15.0, n)
    lats = np.linspace(-15.0, 15.0, n)
    bands = compute_troublespot_bands(grid, lons, lats)
    severe = next(b for b in bands if b["band"] == "severe")
    assert len(severe["rings"]) == 2


@pytest.mark.parametrize("bad_value", [-1, 5])
def test_out_of_range_type_counts_raise(bad_value):
    grid, lons, lats = _grid(block_value=bad_value)
    with pytest.raises(ValueError):
        compute_troublespot_bands(grid, lons, lats)
