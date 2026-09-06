import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from atmos_gl.lib.flood_risk import (
    JRC_MOSAIC_GRID_STEP_DEG,
    MODIS_FLOOD_VALUE,
    build_jrc_mosaic_grid,
    cached_modis_flood_tiles,
    count_cached_jrc_tiles,
    ensure_jrc_tile_cached,
    ensure_jrc_tile_extents_cached,
    ensure_modis_flood_tile_cached,
    fetch_modis_flood_listing,
    jrc_tile_cache_path,
    jrc_tile_extents_cache_path,
    load_jrc_hazard_mosaic,
    load_jrc_tile_index,
    modis_flood_listing_url,
    modis_flood_tile_bounds,
    modis_flood_tile_cache_path,
    modis_flood_tile_is_current,
    parse_modis_flood_listing,
    prune_stale_modis_flood_tiles,
    resample_jrc_tile_onto_grid,
    resample_modis_flood_tile_onto_grid,
    resolve_earthdata_token,
    save_jrc_hazard_mosaic,
    tile_dst_window,
)


# ---- JRC Global River Flood Hazard Maps (Historical mode) ----------------------


def _make_tile_extents_geojson_bytes(features):
    import json

    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


def _tile_feature(tile_id, name, lon_min, lat_min, lon_max, lat_max):
    return {
        "type": "Feature",
        "properties": {"id": tile_id, "name": name},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon_min, lat_max], [lon_max, lat_max],
                [lon_max, lat_min], [lon_min, lat_min], [lon_min, lat_max],
            ]],
        },
    }


def test_build_jrc_mosaic_grid_covers_the_full_globe_at_the_configured_step():
    lat, lon = build_jrc_mosaic_grid()
    assert lat.shape == (round(180.0 / JRC_MOSAIC_GRID_STEP_DEG),)
    assert lon.shape == (round(360.0 / JRC_MOSAIC_GRID_STEP_DEG),)
    assert lat[0] < 90.0 and lat[0] > lat[-1]  # descending, cell-centred (not exactly 90)
    assert lon[0] > -180.0 and lon[-1] < 180.0


def test_tile_dst_window_maps_a_10x10deg_tile_to_the_expected_cell_block():
    """A tile at the NW-most corner (N90/W180-equivalent bounds) must map to
    row/col 0 -- and every tile must be exactly (10/step_deg) cells square,
    matching JRC's/MODIS's own fixed tiling scheme."""
    n = round(10.0 / JRC_MOSAIC_GRID_STEP_DEG)
    row0, row1, col0, col1 = tile_dst_window((-180.0, 80.0, -170.0, 90.0))
    assert (row0, col0) == (0, 0)
    assert row1 - row0 == n
    assert col1 - col0 == n


def test_tile_dst_window_places_adjacent_tiles_without_gap_or_overlap():
    n = round(10.0 / JRC_MOSAIC_GRID_STEP_DEG)
    north = tile_dst_window((-180.0, 60.0, -170.0, 70.0))
    south = tile_dst_window((-180.0, 50.0, -170.0, 60.0))
    east = tile_dst_window((-170.0, 60.0, -160.0, 70.0))
    assert south[0] == north[1]  # south tile's rows start exactly where north's end
    assert east[2] == north[3]  # east tile's cols start exactly where north's end
    assert north[1] - north[0] == n


def _write_tiny_reclass_tif(path, values, bounds):
    """A tiny real GeoTIFF matching JRC's own reclass tile shape (uint8, nodata=255,
    north-up), sized to exactly `bounds` -- exercises resample_jrc_tile_onto_grid's
    real rasterio.warp.reproject path rather than a mocked one."""
    import rasterio
    from rasterio import Affine

    lon_min, lat_min, lon_max, lat_max = bounds
    height, width = values.shape
    transform = Affine(
        (lon_max - lon_min) / width, 0.0, lon_min,
        0.0, -(lat_max - lat_min) / height, lat_max,
    )
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=255,
    ) as dst:
        dst.write(values, 1)


def test_resample_jrc_tile_onto_grid_takes_the_max_category_per_destination_cell(tmp_path):
    """Categorical hazard data must never let a coarse destination cell hide a
    known worst-case within it -- see the function's own docstring."""
    values = np.array([[1, 1], [1, 4]], dtype=np.uint8)
    path = str(tmp_path / "tile_max.tif")
    _write_tiny_reclass_tif(path, values, (0.0, 0.0, 10.0, 10.0))

    dst_lat = np.array([7.5, 2.5])  # 2 cells, north-first (matches the tile's own row order)
    dst_lon = np.array([2.5, 7.5])
    out = resample_jrc_tile_onto_grid(path, dst_lat, dst_lon)

    assert out.shape == (2, 2)
    assert out[1, 1] == 4  # the one cell overlapping the source's "4" pixel
    assert out[0, 0] == 1


def test_resample_jrc_tile_onto_grid_maps_native_nodata_to_zero_not_255(tmp_path):
    """255 (JRC's own nodata -- areas not modelled, not necessarily hazard-free)
    must never survive into the mosaic as a spuriously extreme category under
    max resampling."""
    values = np.full((2, 2), 255, dtype=np.uint8)
    path = str(tmp_path / "tile_nodata.tif")
    _write_tiny_reclass_tif(path, values, (0.0, 0.0, 10.0, 10.0))

    dst_lat = np.array([5.0])
    dst_lon = np.array([5.0])
    out = resample_jrc_tile_onto_grid(path, dst_lat, dst_lon)

    assert out[0, 0] == 0


def test_load_jrc_tile_index_extracts_id_name_and_bounds(tmp_path):
    features = [_tile_feature(1, "N70_W180", -180.0, 60.0, -170.0, 70.0)]
    path = tmp_path / "tile_extents.geojson"
    path.write_bytes(_make_tile_extents_geojson_bytes(features))

    tiles = load_jrc_tile_index(str(path))

    assert tiles == [{"id": 1, "name": "N70_W180", "bounds": (-180.0, 60.0, -170.0, 70.0)}]


def test_ensure_jrc_tile_extents_cached_downloads_and_caches_when_not_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    features = [_tile_feature(1, "N70_W180", -180.0, 60.0, -170.0, 70.0)]
    valid_bytes = _make_tile_extents_geojson_bytes(features)

    with patch("atmos_gl.lib.gfs.download_whole", return_value=valid_bytes) as mock_download:
        path = ensure_jrc_tile_extents_cached()

    mock_download.assert_called_once()
    assert path == jrc_tile_extents_cache_path()
    assert os.path.exists(path)


def test_ensure_jrc_tile_extents_cached_skips_download_when_already_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    dest = jrc_tile_extents_cache_path()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b'{"type": "FeatureCollection", "features": []}')

    with patch("atmos_gl.lib.gfs.download_whole") as mock_download:
        ensure_jrc_tile_extents_cached()

    mock_download.assert_not_called()


def test_ensure_jrc_tile_extents_cached_raises_and_leaves_no_file_on_corrupt_download(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))

    with patch("atmos_gl.lib.gfs.download_whole", return_value=b"not json at all"):
        with pytest.raises(Exception):
            ensure_jrc_tile_extents_cached()

    dest = jrc_tile_extents_cache_path()
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".tmp")


def test_ensure_jrc_tile_extents_cached_raises_on_an_empty_feature_list(tmp_path, monkeypatch):
    """A response with no features at all is treated as corrupt, same as invalid
    JSON -- an empty tile index would silently make Historical mode's mosaic
    permanently empty rather than retrying."""
    monkeypatch.setenv("HOME", str(tmp_path))

    with patch("atmos_gl.lib.gfs.download_whole", return_value=_make_tile_extents_geojson_bytes([])):
        with pytest.raises(Exception):
            ensure_jrc_tile_extents_cached()


def test_ensure_jrc_tile_cached_downloads_and_caches_when_not_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tif_path = tmp_path / "src.tif"
    _write_tiny_reclass_tif(str(tif_path), np.array([[1, 2]], dtype=np.uint8), (0.0, 0.0, 10.0, 10.0))
    valid_bytes = tif_path.read_bytes()

    with patch("atmos_gl.lib.gfs.download_whole", return_value=valid_bytes) as mock_download:
        path = ensure_jrc_tile_cached(1, "N70_W180")

    mock_download.assert_called_once()
    assert path == jrc_tile_cache_path(1, "N70_W180")
    assert os.path.exists(path)


def test_ensure_jrc_tile_cached_skips_download_when_already_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    dest = jrc_tile_cache_path(1, "N70_W180")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b"already-here")

    with patch("atmos_gl.lib.gfs.download_whole") as mock_download:
        path = ensure_jrc_tile_cached(1, "N70_W180")

    mock_download.assert_not_called()
    assert path == dest


def test_ensure_jrc_tile_cached_raises_and_leaves_no_file_on_corrupt_download(tmp_path, monkeypatch):
    """Same truncation risk as ensure_jrc_tile_extents_cached -- confirmed live during
    issue #371's spike that a 271-tile batch download against this host CAN be
    interrupted mid-transfer for individual tiles."""
    monkeypatch.setenv("HOME", str(tmp_path))

    with patch("atmos_gl.lib.gfs.download_whole", return_value=b"not a real tif"):
        with pytest.raises(Exception):
            ensure_jrc_tile_cached(1, "N70_W180")

    dest = jrc_tile_cache_path(1, "N70_W180")
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".tmp")


def test_count_cached_jrc_tiles_is_none_before_the_tile_index_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert count_cached_jrc_tiles() is None


def test_count_cached_jrc_tiles_counts_only_tiles_present_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    features = [
        _tile_feature(1, "N70_W180", -180.0, 60.0, -170.0, 70.0),
        _tile_feature(2, "N60_W180", -180.0, 50.0, -170.0, 60.0),
    ]
    index_path = jrc_tile_extents_cache_path()
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        f.write(_make_tile_extents_geojson_bytes(features))

    only_tile_1 = jrc_tile_cache_path(1, "N70_W180")
    os.makedirs(os.path.dirname(only_tile_1), exist_ok=True)
    with open(only_tile_1, "wb") as f:
        f.write(b"fake")

    assert count_cached_jrc_tiles() == (1, 2)


def test_save_and_load_jrc_hazard_mosaic_round_trips(tmp_path):
    path = str(tmp_path / "mosaic.nc")
    band = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    lat = np.array([10.0, 0.0])
    lon = np.array([20.0, 30.0])

    save_jrc_hazard_mosaic(path, band, lat, lon)
    loaded_band, loaded_lat, loaded_lon = load_jrc_hazard_mosaic(path)

    assert np.array_equal(loaded_band, band)
    assert list(loaded_lat) == [10.0, 0.0]
    assert list(loaded_lon) == [20.0, 30.0]
    assert not os.path.exists(path + ".tmp")


# ---- NASA LANCE MODIS Flood Product (Live mode) ---------------------------------


def test_modis_flood_tile_bounds_nw_corner_is_h0v0():
    """h counts east from -180, v counts south from +90, both in 10deg steps
    (https://modis-land.gsfc.nasa.gov/MODLAND_grid.html) -- h0v0 is therefore the
    NW-most tile."""
    lon_min, lat_min, lon_max, lat_max = modis_flood_tile_bounds(0, 0)
    assert (lon_min, lat_max) == (-180.0, 90.0)
    assert (lon_max, lat_min) == (-170.0, 80.0)


def test_modis_flood_tile_bounds_steps_10deg_per_hv_increment():
    lon_min_a, lat_min_a, lon_max_a, lat_max_a = modis_flood_tile_bounds(19, 6)
    lon_min_b, lat_min_b, lon_max_b, lat_max_b = modis_flood_tile_bounds(20, 7)
    assert lon_min_b - lon_min_a == 10.0
    assert lat_max_a - lat_max_b == 10.0
    assert lon_max_a - lon_min_a == 10.0
    assert lat_max_a - lat_min_a == 10.0


def test_modis_flood_listing_url_encodes_year_and_day_of_year():
    import datetime

    url = modis_flood_listing_url(datetime.datetime(2026, 8, 30, tzinfo=datetime.timezone.utc))
    assert "temporalRanges=2026-242" in url
    assert "products=MCDWD_L3_F1C_NRT" in url


def test_parse_modis_flood_listing_extracts_h_v_and_builds_a_fallback_download_url():
    payload = {"content": [{"name": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.tif"}]}
    tiles = parse_modis_flood_listing(payload)
    assert tiles == [
        {
            "h": 19,
            "v": 6,
            "filename": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.tif",
            "download_url": (
                "https://nrt3.modaps.eosdis.nasa.gov/archive/allData/61/"
                "MCDWD_L3_F1C_NRT/2026/242/"
                "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.tif"
            ),
        }
    ]


def test_parse_modis_flood_listing_prefers_an_explicit_downloads_link():
    payload = {"content": [{
        "name": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.tif",
        "downloadsLink": "https://example.test/explicit-link.tif",
    }]}
    tiles = parse_modis_flood_listing(payload)
    assert tiles[0]["download_url"] == "https://example.test/explicit-link.tif"


def test_parse_modis_flood_listing_skips_entries_that_dont_match_the_filename_grammar():
    payload = {"content": [{"name": "some-unrelated-file.txt"}, {"name": ""}]}
    assert parse_modis_flood_listing(payload) == []


def test_resolve_earthdata_token_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "  a-token  ")
    assert resolve_earthdata_token("flood_risk_live") == "a-token"


def test_resolve_earthdata_token_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    assert resolve_earthdata_token("flood_risk_live") is None


def test_fetch_modis_flood_listing_sends_a_bearer_token_and_parses_the_response():
    import datetime

    fake_response = MagicMock()
    fake_response.json.return_value = {"content": [
        {"name": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.tif"}
    ]}
    with patch("requests.get", return_value=fake_response) as mock_get:
        tiles = fetch_modis_flood_listing(
            datetime.datetime(2026, 8, 30, tzinfo=datetime.timezone.utc), "tok123"
        )

    assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer tok123"}
    fake_response.raise_for_status.assert_called_once()
    assert tiles[0]["h"] == 19 and tiles[0]["v"] == 6


def test_modis_flood_tile_is_current_false_when_never_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tile = {"h": 19, "v": 6, "filename": "x.tif"}
    assert modis_flood_tile_is_current(tile) is False


def test_ensure_modis_flood_tile_cached_downloads_and_writes_the_name_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tif_path = tmp_path / "src.tif"
    _write_tiny_reclass_tif(str(tif_path), np.array([[0, 3]], dtype=np.uint8), (0.0, 0.0, 10.0, 10.0))
    valid_bytes = tif_path.read_bytes()
    tile = {
        "h": 19, "v": 6, "filename": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.111.tif",
        "download_url": "https://example.test/tile.tif",
    }

    with patch("atmos_gl.lib.gfs.download_whole", return_value=valid_bytes) as mock_download:
        path = ensure_modis_flood_tile_cached(tile, "tok123")

    assert mock_download.call_args.kwargs["headers"] == {"Authorization": "Bearer tok123"}
    assert path == modis_flood_tile_cache_path(19, 6)
    assert os.path.exists(path)
    assert modis_flood_tile_is_current(tile) is True


def test_ensure_modis_flood_tile_cached_raises_and_leaves_no_file_on_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tile = {
        "h": 19, "v": 6, "filename": "MCDWD_L3_F1C_NRT.A2026242.h19v06.061.111.tif",
        "download_url": "https://example.test/tile.tif",
    }

    with patch("atmos_gl.lib.gfs.download_whole", return_value=b"not a real tif"):
        with pytest.raises(Exception):
            ensure_modis_flood_tile_cached(tile, "tok123")

    dest = modis_flood_tile_cache_path(19, 6)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".tmp")


def test_prune_stale_modis_flood_tiles_removes_only_expired_tiles(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tile = {
        "h": 1, "v": 1, "filename": "old.tif",
        "download_url": "https://example.test/tile.tif",
    }
    tif_path = tmp_path / "src.tif"
    _write_tiny_reclass_tif(str(tif_path), np.array([[0]], dtype=np.uint8), (0.0, 0.0, 10.0, 10.0))
    with patch("atmos_gl.lib.gfs.download_whole", return_value=tif_path.read_bytes()):
        old_path = ensure_modis_flood_tile_cached(tile, "tok")
        fresh_path = ensure_modis_flood_tile_cached(
            {**tile, "h": 2, "filename": "fresh.tif"}, "tok"
        )

    from atmos_gl.lib.flood_risk import MODIS_FLOOD_STALE_S

    old_time = time.time() - (MODIS_FLOOD_STALE_S + 3600)
    os.utime(old_path, (old_time, old_time))
    os.utime(old_path + ".name", (old_time, old_time))

    pruned = prune_stale_modis_flood_tiles()

    assert pruned is True
    assert not os.path.exists(old_path)
    assert not os.path.exists(old_path + ".name")
    assert os.path.exists(fresh_path)


def test_prune_stale_modis_flood_tiles_returns_false_when_nothing_cached_yet(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert prune_stale_modis_flood_tiles() is False


def test_cached_modis_flood_tiles_parses_h_v_from_cache_filenames(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tile = {"h": 3, "v": 4, "filename": "x.tif", "download_url": "https://example.test/tile.tif"}
    tif_path = tmp_path / "src.tif"
    _write_tiny_reclass_tif(str(tif_path), np.array([[0]], dtype=np.uint8), (0.0, 0.0, 10.0, 10.0))
    with patch("atmos_gl.lib.gfs.download_whole", return_value=tif_path.read_bytes()):
        ensure_modis_flood_tile_cached(tile, "tok")

    tiles = cached_modis_flood_tiles()

    assert tiles == [(3, 4, modis_flood_tile_cache_path(3, 4))]


def test_resample_modis_flood_tile_onto_grid_binarizes_flood_value_only(tmp_path):
    """Only pixel value 3 (Flood, unusual) becomes 1 -- normal water (1), the
    not-yet-populated recurring-flood code (2), and insufficient-data (255) must
    all render as 0 (transparent), not just 255. The Flood pixel here is
    adjacent to the water (1) pixel, so it survives the water-adjacency filter
    (see MODIS_FLOOD_WATER_ADJACENCY_PX) -- the filter itself is covered
    separately below."""
    values = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    path = str(tmp_path / "tile.tif")
    _write_tiny_reclass_tif(path, values, (0.0, 0.0, 10.0, 10.0))

    dst_lat = np.array([7.5, 2.5])
    dst_lon = np.array([2.5, 7.5])
    out = resample_modis_flood_tile_onto_grid(path, dst_lat, dst_lon)

    assert out[1, 1] == 1  # overlaps the source's "3" (Flood) pixel
    assert set(np.unique(out)) <= {0, 1}


def test_resample_modis_flood_tile_onto_grid_keeps_flood_within_adjacency_radius_of_water(tmp_path):
    """Confirmed live against real NZ tiles (h34v13/h35v12/h35v13): isolated Flood
    pixels far from any mapped water cluster in steep terrain (Fiordland, Aoraki/
    Mt Cook) with no nearby water at all -- this product is cloud-shadow-screened
    but not terrain-shadow-screened, so low-sun-angle terrain shadow gets
    misclassified as flood there. A Flood pixel exactly at the adjacency radius
    from water must still be kept."""
    from atmos_gl.lib.flood_risk import MODIS_FLOOD_WATER_ADJACENCY_PX

    row = [0] * 10
    row[0] = 1  # water
    row[MODIS_FLOOD_WATER_ADJACENCY_PX] = 3  # flood, exactly at the radius
    values = np.array([row], dtype=np.uint8)
    path = str(tmp_path / "tile_near_water.tif")
    _write_tiny_reclass_tif(path, values, (0.0, 0.0, 10.0, 10.0))

    dst_lat = np.array([5.0])
    dst_lon = np.arange(10) + 0.5
    out = resample_modis_flood_tile_onto_grid(path, dst_lat, dst_lon)

    assert out[0, MODIS_FLOOD_WATER_ADJACENCY_PX] == 1


def test_resample_modis_flood_tile_onto_grid_drops_flood_beyond_adjacency_radius_of_water(tmp_path):
    """A Flood pixel one step beyond MODIS_FLOOD_WATER_ADJACENCY_PX from the
    nearest water pixel is dropped -- this is exactly the NZ mountain-terrain
    false-positive case (see the "kept" test above)."""
    from atmos_gl.lib.flood_risk import MODIS_FLOOD_WATER_ADJACENCY_PX

    row = [0] * 10
    row[0] = 1  # water
    row[MODIS_FLOOD_WATER_ADJACENCY_PX + 1] = 3  # flood, one pixel too far
    values = np.array([row], dtype=np.uint8)
    path = str(tmp_path / "tile_far_from_water.tif")
    _write_tiny_reclass_tif(path, values, (0.0, 0.0, 10.0, 10.0))

    dst_lat = np.array([5.0])
    dst_lon = np.arange(10) + 0.5
    out = resample_modis_flood_tile_onto_grid(path, dst_lat, dst_lon)

    assert out[0, MODIS_FLOOD_WATER_ADJACENCY_PX + 1] == 0


def test_resample_modis_flood_tile_onto_grid_treats_insufficient_data_as_no_flood(tmp_path):
    values = np.full((2, 2), 255, dtype=np.uint8)
    path = str(tmp_path / "tile_nodata.tif")
    _write_tiny_reclass_tif(path, values, (0.0, 0.0, 10.0, 10.0))

    out = resample_modis_flood_tile_onto_grid(path, np.array([5.0]), np.array([5.0]))

    assert out[0, 0] == 0


def test_modis_flood_value_is_the_flood_unusual_pixel_code():
    """Pinned so an accidental edit is caught -- 3 is Table 7's "Flood (unusual)"
    code in the official User Guide, not the "surface water" (1) or
    "recurring flood" (2, not yet populated) codes."""
    assert MODIS_FLOOD_VALUE == 3
