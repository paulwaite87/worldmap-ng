#!/usr/bin/env python3
"""retrieve_and_unzip: shared submit-then-poll-then-unzip mechanics behind every
CDS-backed collector (CamsGhgForecastCollector, CamsEgg4BaselineCollector,
AirQualityCollector) -- every CDS dataset in this app delivers data_format=netcdf_zip
(a zip archive), not a raw netCDF, so the fetch isn't complete until the archive's .nc
member is extracted to the real cache path.

retrieve_with_fallback: shared "try each candidate request, stop at the first
success" search behind every CDS-backed FORECAST collector (CamsGhgForecastCollector,
AirQualityCollector -- not CamsEgg4BaselineCollector, which fetches a fixed historical
year with no publish-lag concept). Takes a pre-built, freshest-first list of request
dicts rather than a (date_str) -> request builder callback, since not every dataset's
"which run is newest" search is a plain calendar-date search -- AirQualityCollector's
dataset ALSO needs a time-of-day axis (00Z/12Z), confirmed live against the real ADS
API (a plain date-only request, mirroring greenhouse_gases' shape, 400'd). Covered
indirectly by both collectors' own test suites
(test_greenhouse_gases_forecast_collector.py, test_air_quality_collector.py), plus
directly here.
"""
import concurrent.futures
import glob
import os
from unittest.mock import MagicMock, patch

import pytest

from atmos_gl.lib.cds_client import (
    retrieve_and_unzip,
    retrieve_with_fallback,
)


def test_retrieve_and_unzip_extracts_the_nc_member_to_cache_dest(tmp_path, make_netcdf_zip_bytes):
    zip_bytes = make_netcdf_zip_bytes("some_download_name.nc", b"real-netcdf-bytes")

    def fake_retrieve(dataset, request, target):
        with open(target, "wb") as f:
            f.write(zip_bytes)

    client = MagicMock()
    client.retrieve.side_effect = fake_retrieve
    dest = str(tmp_path / "data" / "cached.nc")

    retrieve_and_unzip(client, "some-dataset", {}, dest, timeout_s=5, label="test")

    assert os.path.exists(dest)
    assert open(dest, "rb").read() == b"real-netcdf-bytes"


def test_retrieve_and_unzip_raises_when_archive_has_no_nc_member(tmp_path):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"not a netcdf")

    def fake_retrieve(dataset, request, target):
        with open(target, "wb") as f:
            f.write(buf.getvalue())

    client = MagicMock()
    client.retrieve.side_effect = fake_retrieve
    dest = str(tmp_path / "data" / "cached.nc")

    with pytest.raises(RuntimeError, match="no .nc file"):
        retrieve_and_unzip(client, "some-dataset", {}, dest, timeout_s=5, label="test")

    assert not os.path.exists(dest)


def _requests(*dates):
    return [{"date": f"{d}/{d}"} for d in dates]


def test_retrieve_with_fallback_succeeds_on_the_first_candidate(tmp_path, make_netcdf_zip_bytes):
    zip_bytes = make_netcdf_zip_bytes("data.nc", b"todays-run")

    def fake_retrieve(dataset, request, target):
        with open(target, "wb") as f:
            f.write(zip_bytes)

    client = MagicMock()
    client.retrieve.side_effect = fake_retrieve
    dest = str(tmp_path / "data" / "cached.nc")

    ok = retrieve_with_fallback(
        client, "some-dataset", _requests("2026-07-29", "2026-07-28"), dest,
        timeout_s=5, label="test",
    )

    assert ok is True
    assert client.retrieve.call_count == 1
    assert open(dest, "rb").read() == b"todays-run"


def test_retrieve_with_fallback_tries_the_next_candidate_when_the_first_fails(
    tmp_path, make_netcdf_zip_bytes
):
    zip_bytes = make_netcdf_zip_bytes("data.nc", b"yesterdays-run")
    seen_dates = []

    def fake_retrieve(dataset, request, target):
        date_str = request["date"].split("/")[0]
        seen_dates.append(date_str)
        if len(seen_dates) == 1:
            raise RuntimeError("400 Client Error: today's run not published yet")
        with open(target, "wb") as f:
            f.write(zip_bytes)

    client = MagicMock()
    client.retrieve.side_effect = fake_retrieve
    dest = str(tmp_path / "data" / "cached.nc")

    ok = retrieve_with_fallback(
        client, "some-dataset", _requests("2026-07-29", "2026-07-28"), dest,
        timeout_s=5, label="test",
    )

    assert ok is True
    assert seen_dates == ["2026-07-29", "2026-07-28"]
    assert open(dest, "rb").read() == b"yesterdays-run"


def test_retrieve_with_fallback_stops_at_a_timeout_without_trying_later_candidates(tmp_path):
    client = MagicMock()
    client.retrieve.side_effect = RuntimeError("unused")
    dest = str(tmp_path / "data" / "cached.nc")

    with patch(
        "atmos_gl.lib.cds_client.retrieve_with_timeout",
        side_effect=concurrent.futures.TimeoutError,
    ):
        ok = retrieve_with_fallback(
            client, "some-dataset", _requests("2026-07-29", "2026-07-28", "2026-07-27"), dest,
            timeout_s=5, label="test",
        )

    assert ok is False
    assert not os.path.exists(dest)


def test_retrieve_with_fallback_gives_up_after_every_candidate_fails(tmp_path):
    client = MagicMock()
    client.retrieve.side_effect = RuntimeError("400 Client Error")
    dest = str(tmp_path / "data" / "cached.nc")

    ok = retrieve_with_fallback(
        client, "some-dataset", _requests("2026-07-29", "2026-07-28", "2026-07-27"), dest,
        timeout_s=5, label="test",
    )

    assert ok is False
    assert client.retrieve.call_count == 3
    assert not os.path.exists(dest)


def test_retrieve_with_fallback_unzip_false_downloads_the_bare_nc_directly(tmp_path):
    """GloFAS's data_format=netcdf/download_format=unarchived delivers a bare .nc, not
    a zip archive -- unzip=False must skip retrieve_and_unzip's archive-extraction
    step entirely and cache the file as-is."""

    def fake_retrieve(dataset, request, target):
        with open(target, "wb") as f:
            f.write(b"bare-netcdf-bytes")

    client = MagicMock()
    client.retrieve.side_effect = fake_retrieve
    dest = str(tmp_path / "data" / "cached.nc")

    ok = retrieve_with_fallback(
        client, "cems-glofas-forecast", _requests("2026-07-29"), dest,
        timeout_s=5, label="test", unzip=False,
    )

    assert ok is True
    assert open(dest, "rb").read() == b"bare-netcdf-bytes"
    assert not os.path.exists(dest + ".tmp")  # atomically renamed away, not left behind


def test_retrieve_with_fallback_unzip_false_leaves_no_partial_file_on_timeout(tmp_path):
    """A timed-out retrieve must never leave a partial/corrupt file at the real cache
    path -- retrieve_with_timeout's own docstring notes a timed-out background
    download isn't cancelled, so it could keep writing after this call returns; the
    tempfile + os.replace step must keep that off the real `dest` path."""
    client = MagicMock()
    dest = str(tmp_path / "data" / "cached.nc")

    with patch(
        "atmos_gl.lib.cds_client.retrieve_with_timeout",
        side_effect=concurrent.futures.TimeoutError,
    ):
        ok = retrieve_with_fallback(
            client, "cems-glofas-forecast", _requests("2026-07-29"), dest,
            timeout_s=5, label="test", unzip=False,
        )

    assert ok is False
    assert not os.path.exists(dest)


def test_retrieve_with_fallback_unzip_false_uses_a_unique_tmp_path_per_call(tmp_path):
    """Regression guard: confirmed live on prod (flood_risk_live) that a fixed
    f"{dest}.tmp" path lets a timed-out call's orphaned background download thread
    (retrieve_with_timeout's own docstring: not cancelled on timeout) collide with a
    LATER call's fresh attempt -- multiurl resumes from the shared partial file's
    on-disk size with no remote content validation, observed as the byte offset
    jumping backwards by over a gigabyte between attempts. Two separate timed-out
    calls must never target the same tmp path."""
    client = MagicMock()
    dest = str(tmp_path / "data" / "cached.nc")
    seen_targets = []

    def fake_timeout(client, dataset, request, target, timeout_s):
        seen_targets.append(target)
        raise concurrent.futures.TimeoutError

    with patch("atmos_gl.lib.cds_client.retrieve_with_timeout", side_effect=fake_timeout):
        retrieve_with_fallback(
            client, "cems-glofas-forecast", _requests("2026-07-29"), dest,
            timeout_s=5, label="test", unzip=False,
        )
        retrieve_with_fallback(
            client, "cems-glofas-forecast", _requests("2026-07-29"), dest,
            timeout_s=5, label="test", unzip=False,
        )

    assert len(seen_targets) == 2
    assert seen_targets[0] != seen_targets[1]


def test_retrieve_with_fallback_unzip_false_cleans_up_a_stale_tmp_from_an_earlier_call(
    tmp_path,
):
    """An orphaned tmp file from an earlier timed-out call (see the unique-path test
    above) must not accumulate forever -- housekeeper.sweep()'s expiry-based cleanup
    only fires once a layer's cache_expiry_days elapses, often unset (keep forever)
    for a file-cache layer like this one, so this function must sweep its own stale
    siblings before starting a new attempt."""
    client = MagicMock()
    dest = str(tmp_path / "data" / "cached.nc")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    stale_tmp = f"{dest}.deadbeefdeadbeefdeadbeefdeadbeef.tmp"
    with open(stale_tmp, "wb") as f:
        f.write(b"orphaned-partial-bytes")

    def fake_retrieve(dataset, request, target):
        with open(target, "wb") as f:
            f.write(b"bare-netcdf-bytes")

    client.retrieve.side_effect = fake_retrieve

    ok = retrieve_with_fallback(
        client, "cems-glofas-forecast", _requests("2026-07-29"), dest,
        timeout_s=5, label="test", unzip=False,
    )

    assert ok is True
    assert not os.path.exists(stale_tmp)
    assert glob.glob(f"{dest}.*.tmp") == []  # no leftover tmp of any kind

