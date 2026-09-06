#!/usr/bin/env python3
"""Route-level tests for GET /api/data_status (architecture review candidate "Give
routers the seam the Fakes are waiting for" -- the deferred status.py follow-on).

This route's real untestability was never the one FieldCatalogAdapter it constructs
(fieldstore.get_store freezes as a process-wide singleton on first use, so overriding
that adapter here wouldn't reliably reach it) -- it's the 23 real collector/task
classes across 5 registries it iterates. Injecting the registries lets these tests
substitute a handful of stub classes and exercise the route's OWN logic (iteration,
per-class exception swallowing, _serialize(), the final envelope) without touching a
real DB, config file, or the fieldstore singleton -- previously untestable at all.

POST /api/data_status/channel_enabled/{key} tests use a throwaway config file (via
CONFIG_PATH) rather than the real one, since this endpoint writes to disk.
"""
import json
from datetime import datetime, timezone

import pytest

from atmos_gl.routes.status import (
    get_collector_classes,
    get_cache_collector_classes,
    get_field_collector_classes,
    get_embeddable_collector_classes,
    get_task_classes,
    get_process_status_adapter,
    _collect_status_rows,
    _serialize,
    _display_name,
    RUNS_PER_DAY_SECTIONS,
)
from atmos_gl.api import app
from atmos_gl.db.process_status_adapter import FakeProcessStatusAdapter


@pytest.fixture(autouse=True)
def _admin_session(admin_client):
    """Every route here is gated by require_admin (issue #304); these tests exercise
    the route's own logic, not the gate itself (see tests/test_admin_gated_routes.py
    for that), so authenticate as admin automatically rather than touching every
    individual test's client.get/post call."""
    pass


class _StubCollector:
    """Ignores whatever constructor args a real collector would need (config, or
    config+store, or config_path) -- proving the route's DI seam means a test never
    has to supply a real one."""

    def __init__(self, *args, **kwargs):
        pass

    def data_status(self):
        return {
            "name": "stub_collector",
            "kind": "collector",
            "percent": 100.0,
            "last_updated": datetime(2026, 6, 13, 18, 0, tzinfo=timezone.utc),
            "next_update": datetime(2026, 6, 13, 19, 0, tzinfo=timezone.utc),
            "enabled": True,
            "detail": None,
        }


class _StubGatedCollector:
    """A stub with channel_key set -- e.g. sst/quakes -- to prove the route reads and
    forwards it, and that a stub with NO channel_key attribute (_StubCollector, matching
    a real un-gated collector like markers) still serializes fine via getattr's
    default rather than raising."""

    channel_key = "stub_channel"

    def __init__(self, *args, **kwargs):
        pass

    def data_status(self):
        return {
            "name": "stub_gated_collector",
            "kind": "collector",
            "percent": 50.0,
            "last_updated": None,
            "next_update": None,
            "enabled": True,
            "detail": None,
        }


class _StubCollectorWithSourceUrl:
    """A stub defining source_url() -- e.g. quakes/sst -- to prove the route reads and
    forwards it, alongside _StubCollector (no source_url attribute at all, matching a
    real sourceless collector like markers) proving that case degrades to None rather
    than raising."""

    def __init__(self, *args, **kwargs):
        pass

    def data_status(self):
        return {
            "name": "stub_source_collector",
            "kind": "collector",
            "percent": 100.0,
            "last_updated": None,
            "next_update": None,
            "enabled": True,
            "detail": None,
        }

    def source_url(self):
        return "https://example.com/source.csv"


class _RaisingCollector:
    def __init__(self, *args, **kwargs):
        pass

    def data_status(self):
        raise RuntimeError("simulated collector failure")


class _StubTask:
    def __init__(self, *args, **kwargs):
        pass

    def layer_status(self):
        return {
            "name": "stub_layer",
            "kind": "layer",
            "percent": 75.0,
            "last_updated": None,
            "next_update": None,
            "enabled": True,
            "detail": None,
        }


# --- _collect_status_rows() (architecture review Candidate 3: the shared shape behind
# get_data_status()'s three collector loops) -- unit-tested directly, without going
# through the FastAPI route/TestClient, since it's a module-level function now. ---


def test_collect_status_rows_serializes_each_class():
    rows = _collect_status_rows(
        (_StubCollector,), {}, construct=lambda cls: cls(),
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "stub_collector"
    assert rows[0]["percent"] == 100.0


def test_collect_status_rows_forwards_channel_key_and_state():
    rows = _collect_status_rows(
        (_StubGatedCollector,), {"stub_channel": False}, construct=lambda cls: cls(),
    )
    assert rows[0]["channel_key"] == "stub_channel"
    assert rows[0]["channel_on"] is False


def test_collect_status_rows_forwards_source_url():
    rows = _collect_status_rows(
        (_StubCollectorWithSourceUrl,), {}, construct=lambda cls: cls(),
    )
    assert rows[0]["source_url"] == "https://example.com/source.csv"


def test_collect_status_rows_defaults_source_url_to_none_without_the_attribute():
    rows = _collect_status_rows(
        (_StubCollector,), {}, construct=lambda cls: cls(),
    )
    assert rows[0]["source_url"] is None


def test_collect_status_rows_applies_cadence_of_per_class():
    rows = _collect_status_rows(
        (_StubCollector,), {}, construct=lambda cls: cls(),
        cadence_of=lambda cls: (24, "stub_collector"),
    )
    assert rows[0]["runs_per_day"] == 24
    assert rows[0]["runs_per_day_section"] == "stub_collector"


def test_collect_status_rows_defaults_cadence_to_none():
    rows = _collect_status_rows(
        (_StubCollector,), {}, construct=lambda cls: cls(),
    )
    assert rows[0]["runs_per_day"] is None
    assert rows[0]["runs_per_day_section"] is None


def test_collect_status_rows_forwards_display_label():
    class _StubWithDisplayLabel(_StubCollector):
        display_label = "Custom Label"

    rows = _collect_status_rows(
        (_StubWithDisplayLabel,), {}, construct=lambda cls: cls(),
    )
    assert rows[0]["display_name"] == "Custom Label"


def test_collect_status_rows_falls_back_to_generic_display_name_without_display_label():
    rows = _collect_status_rows(
        (_StubCollector,), {}, construct=lambda cls: cls(),
    )
    assert rows[0]["display_name"] == _display_name("stub_collector")


def test_collect_status_rows_logs_and_skips_a_raising_class_without_aborting():
    rows = _collect_status_rows(
        (_RaisingCollector, _StubCollector), {}, construct=lambda cls: cls(),
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "stub_collector"


def test_collect_status_rows_uses_the_construct_closure():
    """Proves constructor arity is the call site's business, not this function's --
    the whole point of taking `construct` as a closure instead of hardcoding
    cls(config)."""
    seen = []

    class _ArityStub:
        def __init__(self, *args, **kwargs):
            pass

        def data_status(self):
            return {
                "name": "arity_stub", "kind": "collector", "percent": 0.0,
                "last_updated": None, "next_update": None, "enabled": True, "detail": None,
            }

    def construct(cls):
        seen.append(cls)
        return cls("fake_config", "fake_store")  # 2-arg constructor, e.g. field collectors

    rows = _collect_status_rows((_ArityStub,), {}, construct=construct)

    assert seen == [_ArityStub]
    assert rows[0]["name"] == "arity_stub"


def _override_all_empty():
    app.dependency_overrides[get_collector_classes] = lambda: ()
    app.dependency_overrides[get_cache_collector_classes] = lambda: ()
    app.dependency_overrides[get_field_collector_classes] = lambda: ()
    app.dependency_overrides[get_embeddable_collector_classes] = lambda: []
    app.dependency_overrides[get_task_classes] = lambda: {}


def test_data_status_with_all_empty_registries(client):
    _override_all_empty()

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "data": {"collectors": [], "layers": []}}


def test_data_status_reflects_stub_collector_and_task(client):
    _override_all_empty()
    app.dependency_overrides[get_collector_classes] = lambda: (_StubCollector,)
    app.dependency_overrides[get_task_classes] = lambda: {"stub": _StubTask}

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]["collectors"]) == 1
    assert data["data"]["collectors"][0]["name"] == "stub_collector"
    assert data["data"]["collectors"][0]["last_updated"] == "2026-06-13T18:00:00+00:00"
    assert len(data["data"]["layers"]) == 1
    assert data["data"]["layers"][0]["name"] == "stub_layer"


def test_data_status_defaults_channel_key_to_none_when_collector_class_lacks_it(client):
    """A real un-gated collector (markers) has no channel_key attribute at all --
    must serialize with channel_key: null rather than raising."""
    _override_all_empty()
    app.dependency_overrides[get_collector_classes] = lambda: (_StubCollector,)

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["channel_key"] is None


def test_data_status_defaults_source_url_to_none_when_collector_class_lacks_it(client):
    """A stub (or real collector, e.g. markers) with no source_url() at all must
    serialize with source_url: null rather than raising."""
    _override_all_empty()
    app.dependency_overrides[get_collector_classes] = lambda: (_StubCollector,)

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["source_url"] is None


def test_data_status_forwards_a_collector_classs_source_url(client):
    _override_all_empty()
    app.dependency_overrides[get_collector_classes] = lambda: (_StubCollectorWithSourceUrl,)

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["source_url"] == "https://example.com/source.csv"


def test_data_status_forwards_a_collector_classs_channel_key(client):
    _override_all_empty()
    app.dependency_overrides[get_collector_classes] = lambda: (_StubGatedCollector,)

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["channel_key"] == "stub_channel"


def test_data_status_forwards_a_layers_derived_channel_key(client):
    """isobars is a real TASK_CLASSES entry backed by gfs_atmos in production; here a
    stub field collector plays gfs_atmos's role to prove the layers loop actually
    looks the section up in the derived mapping, not just passes None through."""
    _override_all_empty()

    class _StubFieldCollector:
        channel_key = "stub_gfs"
        products = {"isobars": None}

        def __init__(self, *args, **kwargs):
            pass

        def data_status(self):
            return {
                "name": "stub_gfs", "kind": "collector", "percent": 0.0,
                "last_updated": None, "next_update": None, "enabled": True, "detail": None,
            }

    app.dependency_overrides[get_field_collector_classes] = lambda: (_StubFieldCollector,)
    app.dependency_overrides[get_task_classes] = lambda: {"isobars": _StubTask}

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["layers"][0]["channel_key"] == "stub_gfs"


def test_data_status_swallows_a_failing_collector_and_keeps_the_rest(client):
    _override_all_empty()
    app.dependency_overrides[get_collector_classes] = lambda: (_RaisingCollector, _StubCollector)

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    collectors = resp.json()["data"]["collectors"]
    assert len(collectors) == 1
    assert collectors[0]["name"] == "stub_collector"


def test_data_status_populates_every_registry_independently(client):
    app.dependency_overrides[get_collector_classes] = lambda: (_StubCollector,)
    app.dependency_overrides[get_cache_collector_classes] = lambda: (_StubCollector,)
    app.dependency_overrides[get_field_collector_classes] = lambda: (_StubCollector,)
    app.dependency_overrides[get_embeddable_collector_classes] = lambda: [_StubCollector]
    app.dependency_overrides[get_task_classes] = lambda: {"a": _StubTask, "b": _StubTask}

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data["collectors"]) == 4  # one per registry
    assert len(data["layers"]) == 2


_BARE_STATUS = {
    "name": "x", "kind": "collector", "percent": 0.0,
    "last_updated": None, "next_update": None, "enabled": True, "detail": None,
}


def test_serialize_channel_on_is_none_when_not_gated():
    out = _serialize(_BARE_STATUS, None, {"quakes": False})
    assert out["channel_key"] is None
    assert out["channel_on"] is None


def test_serialize_channel_on_defaults_true_when_key_absent_from_dict():
    """A channel not yet present in channel_enabled (e.g. right after upgrading to
    this feature) must read as on, not off -- matches the collection-side gating's
    own default in _drive()/_collect_fields()."""
    out = _serialize(_BARE_STATUS, "quakes", {})
    assert out["channel_on"] is True


def test_serialize_channel_on_reflects_an_explicit_false():
    out = _serialize(_BARE_STATUS, "quakes", {"quakes": False})
    assert out["channel_on"] is False


def test_serialize_channel_on_reflects_an_explicit_true():
    out = _serialize(_BARE_STATUS, "quakes", {"quakes": True})
    assert out["channel_on"] is True


def _write_temp_config(tmp_path, data_collector_extra=None):
    path = tmp_path / "atmos-gl.json"
    config = {
        "common": {"workdir": "."},
        "data_collector": {"channel_enabled": {"quakes": True}, **(data_collector_extra or {})},
    }
    path.write_text(json.dumps(config))
    return path


def test_set_channel_enabled_persists_a_flip(client, tmp_path, monkeypatch):
    config_path = _write_temp_config(tmp_path)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.post(
        "/api/data_status/channel_enabled/quakes", json={"enabled": False}
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "channel_key": "quakes", "enabled": False}
    saved = json.loads(config_path.read_text())
    assert saved["data_collector"]["channel_enabled"]["quakes"] is False


def test_set_channel_enabled_creates_the_dict_if_missing(client, tmp_path, monkeypatch):
    config_path = _write_temp_config(tmp_path, data_collector_extra={})
    config_path.write_text(json.dumps({"common": {"workdir": "."}, "data_collector": {}}))
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.post(
        "/api/data_status/channel_enabled/gfs_atmos", json={"enabled": False}
    )

    assert resp.status_code == 200
    saved = json.loads(config_path.read_text())
    assert saved["data_collector"]["channel_enabled"]["gfs_atmos"] is False


def test_set_channel_enabled_rejects_a_non_boolean(client, tmp_path, monkeypatch):
    config_path = _write_temp_config(tmp_path)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.post(
        "/api/data_status/channel_enabled/quakes", json={"enabled": "false"}
    )

    assert resp.status_code == 422


def test_display_name_matches_section_labels_for_a_real_section():
    """Matches the Show tab's wording exactly -- section_label()'s own contract."""
    assert _display_name("quakes") == "Earthquakes"
    assert _display_name("pwat") == "Precipitable Water"
    assert _display_name("satellites_collector") == "Satellites Collector"


def test_display_name_no_longer_special_cases_field_collector_status_names():
    """_display_name() itself is now the plain section_label() fallback -- the 3 field
    collectors' proper GFS/RTOFS acronyms come from their own display_label class
    attribute instead (resolved in _collect_status_rows(), see the tests below), not a
    second dict here that had to stay in sync with FIELD_COLLECTOR_CLASSES."""
    assert _display_name("gfs_atmos") == "Gfs Atmos"
    assert _display_name("rtofs_currents") == "Rtofs Currents"


def test_field_collectors_carry_the_proper_acronym_display_label():
    from atmos_gl.collectors.gfs_atmos import GfsAtmosCollector
    from atmos_gl.collectors.gfs_waves import GfsWavesCollector
    from atmos_gl.collectors.rtofs_currents import RtofsCurrentsCollector

    assert GfsAtmosCollector.display_label == "GFS Atmos"
    assert GfsWavesCollector.display_label == "GFS Waves"
    assert RtofsCurrentsCollector.display_label == "RTOFS Currents"


def test_display_name_falls_back_to_title_case_for_an_unknown_key():
    assert _display_name("totally_unknown_key") == "Totally Unknown Key"


def test_serialize_attaches_display_name_without_touching_name():
    out = _serialize(_BARE_STATUS, None, {})
    assert out["name"] == "x"
    assert out["display_name"] == "X"


def test_data_status_groups_collector_suffixed_rows_at_the_bottom(client):
    """satellites_collector/shipping_collector/lightning_collector run a *service*,
    not a data source -- they should sort after every real source regardless of
    registry definition order, with each group's own relative order preserved."""
    _override_all_empty()

    class _FakeQuakes(_StubCollector):
        def data_status(self):
            return {**super().data_status(), "name": "quakes"}

    class _FakeSatellitesCollector(_StubCollector):
        def data_status(self):
            return {**super().data_status(), "name": "satellites_collector"}

    class _FakeStorms(_StubCollector):
        def data_status(self):
            return {**super().data_status(), "name": "storms"}

    # Deliberately interleaved, matching how the real registries are actually ordered.
    app.dependency_overrides[get_collector_classes] = lambda: (
        _FakeQuakes, _FakeSatellitesCollector, _FakeStorms,
    )

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()["data"]["collectors"]]
    assert names == ["quakes", "storms", "satellites_collector"]


# --- runs_per_day: _serialize(), get_data_status() row attachment, and the
# save endpoint ---


def test_serialize_attaches_the_given_runs_per_day():
    out = _serialize(_BARE_STATUS, None, {}, 12)
    assert out["runs_per_day"] == 12


def test_serialize_defaults_runs_per_day_to_none():
    out = _serialize(_BARE_STATUS, None, {})
    assert out["runs_per_day"] is None


def test_serialize_attaches_the_given_runs_per_day_section():
    out = _serialize(_BARE_STATUS, None, {}, 12, None, "quakes")
    assert out["runs_per_day_section"] == "quakes"


def test_serialize_defaults_runs_per_day_section_to_none():
    out = _serialize(_BARE_STATUS, None, {})
    assert out["runs_per_day_section"] is None


def test_serialize_display_label_wins_over_the_generic_derivation():
    out = _serialize(_BARE_STATUS, None, {}, display_label="Custom Label")
    assert out["display_name"] == "Custom Label"


def test_serialize_falls_back_to_display_name_without_a_display_label():
    out = _serialize(_BARE_STATUS, None, {})
    assert out["display_name"] == _display_name(_BARE_STATUS["name"])


def _write_runs_per_day_config(tmp_path, **extra_sections):
    path = tmp_path / "atmos-gl.json"
    config = {
        "common": {"workdir": "."},
        "data_collector": {"channel_enabled": {}, "runs_per_day": 96},
        **extra_sections,
    }
    path.write_text(json.dumps(config))
    return path


def test_data_status_attaches_runs_per_day_for_a_migrated_section(client, tmp_path, monkeypatch):
    """quakes is in RUNS_PER_DAY_SECTIONS -- its row must carry the configured value."""
    _override_all_empty()

    class _FakeQuakes(_StubCollector):
        section = "quakes"

        def data_status(self):
            return {**super().data_status(), "name": "quakes"}

    app.dependency_overrides[get_collector_classes] = lambda: (_FakeQuakes,)
    config_path = _write_runs_per_day_config(tmp_path, quakes={"runs_per_day": 12})
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["runs_per_day"] == 12


def test_data_status_attaches_runs_per_day_for_world_events(client, tmp_path, monkeypatch):
    """world_events has a real runs_per_day config value but had no operator-facing
    control anywhere (not this page, not a Settings-tab field) until it was added to
    RUNS_PER_DAY_SECTIONS."""
    _override_all_empty()

    class _FakeWorldEvents(_StubCollector):
        section = "world_events"

        def data_status(self):
            return {**super().data_status(), "name": "world_events"}

    app.dependency_overrides[get_collector_classes] = lambda: (_FakeWorldEvents,)
    config_path = _write_runs_per_day_config(tmp_path, world_events={"runs_per_day": 24})
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["runs_per_day"] == 24


def test_data_status_attaches_runs_per_day_for_air_quality(client, tmp_path, monkeypatch):
    """Same gap as world_events -- air_quality's runs_per_day was configured but
    unreachable from any UI until it was added to RUNS_PER_DAY_SECTIONS."""
    _override_all_empty()

    class _FakeAirQuality(_StubCollector):
        section = "air_quality"

        def data_status(self):
            return {**super().data_status(), "name": "air_quality"}

    app.dependency_overrides[get_cache_collector_classes] = lambda: (_FakeAirQuality,)
    config_path = _write_runs_per_day_config(tmp_path, air_quality={"runs_per_day": 24})
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["runs_per_day"] == 24


def test_data_status_defaults_runs_per_day_to_1_when_key_absent_from_config(
    client, tmp_path, monkeypatch
):
    """A section in RUNS_PER_DAY_SECTIONS always gets a widget -- if the key is
    missing from config (e.g. an unmigrated live file), the row must show the real
    effective default (CollectorBase.period_s's own fallback of 1), not None, or the
    frontend's `item.runs_per_day != null` gate hides the widget entirely."""
    _override_all_empty()

    class _FakeQuakes(_StubCollector):
        section = "quakes"

        def data_status(self):
            return {**super().data_status(), "name": "quakes"}

    app.dependency_overrides[get_collector_classes] = lambda: (_FakeQuakes,)
    config_path = _write_runs_per_day_config(tmp_path)  # no quakes.runs_per_day set
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["runs_per_day"] == 1


def test_data_status_omits_runs_per_day_for_a_non_migrated_section(client, tmp_path, monkeypatch):
    """isobars is vestigial (never had a real runs_per_day) -- its row must not carry
    one even if a stray key happens to be in config."""
    _override_all_empty()

    class _FakeIsobars(_StubCollector):
        section = "isobars"

        def data_status(self):
            return {**super().data_status(), "name": "isobars"}

    app.dependency_overrides[get_collector_classes] = lambda: (_FakeIsobars,)
    config_path = _write_runs_per_day_config(tmp_path, isobars={"runs_per_day": 12})
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    assert resp.json()["data"]["collectors"][0]["runs_per_day"] is None


def test_data_status_attaches_data_collectors_runs_per_day_only_to_gfs_atmos_row(
    client, tmp_path, monkeypatch
):
    """gfs_atmos, gfs_waves, and rtofs_currents share data_collector.runs_per_day, but
    only the gfs_atmos row surfaces it -- see RUNS_PER_DAY_SECTIONS' docstring."""
    _override_all_empty()

    class _FakeGfsAtmos(_StubCollector):
        status_name = "gfs_atmos"

        def data_status(self):
            return {**super().data_status(), "name": "gfs_atmos"}

    class _FakeGfsWaves(_StubCollector):
        status_name = "gfs_waves"

        def data_status(self):
            return {**super().data_status(), "name": "gfs_waves"}

    app.dependency_overrides[get_field_collector_classes] = lambda: (
        _FakeGfsAtmos, _FakeGfsWaves,
    )
    config_path = _write_runs_per_day_config(tmp_path)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    collectors = {c["name"]: c["runs_per_day"] for c in resp.json()["data"]["collectors"]}
    assert collectors == {"gfs_atmos": 96, "gfs_waves": None}
    sections = {c["name"]: c["runs_per_day_section"] for c in resp.json()["data"]["collectors"]}
    # gfs_atmos's cadence saves under "data_collector", not "gfs_atmos" -- the frontend
    # reads this straight off the row instead of re-deriving the special case itself.
    assert sections == {"gfs_atmos": "data_collector", "gfs_waves": None}


def test_data_status_attaches_flood_risk_runs_per_day_only_to_live_row(
    client, tmp_path, monkeypatch
):
    """flood_risk_live and flood_risk_historical share flood_risk.runs_per_day (via
    settings_section), but only the live row surfaces the cadence widget --
    flood_risk_historical already respects it via its own EventFeedDriver-based
    is_stale() and isn't in RUNS_PER_DAY_SECTIONS, so it deliberately shows none."""
    _override_all_empty()

    class _FakeFloodRiskLive(_StubCollector):
        status_name = "flood_risk_live"

        def data_status(self):
            return {**super().data_status(), "name": "flood_risk_live"}

    class _FakeGfsWaves(_StubCollector):
        status_name = "gfs_waves"

        def data_status(self):
            return {**super().data_status(), "name": "gfs_waves"}

    app.dependency_overrides[get_field_collector_classes] = lambda: (
        _FakeFloodRiskLive, _FakeGfsWaves,
    )
    config_path = _write_runs_per_day_config(tmp_path, flood_risk={"runs_per_day": 12})
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    collectors = {c["name"]: c["runs_per_day"] for c in resp.json()["data"]["collectors"]}
    assert collectors == {"flood_risk_live": 12, "gfs_waves": None}
    sections = {c["name"]: c["runs_per_day_section"] for c in resp.json()["data"]["collectors"]}
    assert sections == {"flood_risk_live": "flood_risk", "gfs_waves": None}


def test_data_status_attaches_greenhouse_gases_runs_per_day_only_to_forecast_row(
    client, tmp_path, monkeypatch
):
    """CamsGhgForecastCollector and CamsEgg4BaselineCollector share greenhouse_gases.
    runs_per_day (via settings_section), but only the ghg_forecast row surfaces the
    cadence widget -- same "one row per shared cadence" rule as gfs_atmos, and
    mirrors build_layer_channel_keys()'s "first collector wins" choice."""
    _override_all_empty()

    class _FakeForecast(_StubCollector):
        section = "ghg_forecast"

        def data_status(self):
            return {**super().data_status(), "name": "ghg_forecast"}

    class _FakeEgg4(_StubCollector):
        section = "ghg_egg4_baseline"

        def data_status(self):
            return {**super().data_status(), "name": "ghg_egg4_baseline"}

    app.dependency_overrides[get_cache_collector_classes] = lambda: (
        _FakeForecast, _FakeEgg4,
    )
    config_path = _write_runs_per_day_config(
        tmp_path, greenhouse_gases={"runs_per_day": 24}
    )
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/api/data_status")

    assert resp.status_code == 200
    collectors = {c["name"]: c["runs_per_day"] for c in resp.json()["data"]["collectors"]}
    assert collectors == {"ghg_forecast": 24, "ghg_egg4_baseline": None}
    sections = {c["name"]: c["runs_per_day_section"] for c in resp.json()["data"]["collectors"]}
    assert sections == {"ghg_forecast": "greenhouse_gases", "ghg_egg4_baseline": None}


def test_runs_per_day_sections_excludes_data_collector_itself():
    """data_collector isn't a section with its own row -- its value is attached to
    the gfs_atmos row specifically (see the field_collector_classes loop above)."""
    assert "data_collector" not in RUNS_PER_DAY_SECTIONS


def test_set_runs_per_day_persists_a_valid_value(client, tmp_path, monkeypatch):
    config_path = _write_runs_per_day_config(tmp_path, quakes={"runs_per_day": 24})
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.post("/api/data_status/runs_per_day/quakes", json={"runs_per_day": 6})

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "section": "quakes", "runs_per_day": 6}
    saved = json.loads(config_path.read_text())
    assert saved["quakes"]["runs_per_day"] == 6


def test_set_runs_per_day_persists_under_data_collector_for_the_gfs_atmos_row(
    client, tmp_path, monkeypatch
):
    config_path = _write_runs_per_day_config(tmp_path)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.post(
        "/api/data_status/runs_per_day/data_collector", json={"runs_per_day": 48}
    )

    assert resp.status_code == 200
    saved = json.loads(config_path.read_text())
    assert saved["data_collector"]["runs_per_day"] == 48


def test_set_runs_per_day_rejects_a_value_outside_the_choice_set(client, tmp_path, monkeypatch):
    config_path = _write_runs_per_day_config(tmp_path)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.post("/api/data_status/runs_per_day/quakes", json={"runs_per_day": 8})

    assert resp.status_code == 422


def test_set_runs_per_day_rejects_a_boolean():
    """1 == True in Python -- guard against a JSON boolean silently passing the `in
    RUNS_PER_DAY_CHOICES` membership check."""
    from atmos_gl.routes.status import set_runs_per_day
    from fastapi import HTTPException
    import pytest

    with pytest.raises(HTTPException) as exc_info:
        set_runs_per_day("quakes", {"runs_per_day": True})
    assert exc_info.value.status_code == 422


# --- Flight Radar hotspot-coverage progress row (issue #215) -- a hand-built Layers
# entry (no TASK_CLASSES/Updater involved), sourced from AircraftCollector's
# record_progress() writes via a dedicated process_status_adapter DI seam.

def _hotspot_row(data):
    return next(r for r in data["data"]["layers"] if r["name"] == "flightradar_hotspot")


def test_flightradar_hotspot_shows_idle_with_no_active_viewport(client):
    _override_all_empty()
    app.dependency_overrides[get_process_status_adapter] = lambda: FakeProcessStatusAdapter()

    resp = client.get("/api/data_status")

    row = _hotspot_row(resp.json())
    assert row["percent"] == 0.0
    assert row["segments"] is None
    assert row["detail"] == "No active viewport"


def test_flightradar_hotspot_reflects_partial_progress(client):
    _override_all_empty()
    fake = FakeProcessStatusAdapter()
    fake.record_progress("flightradar_hotspot", "layer", 3, 8)
    app.dependency_overrides[get_process_status_adapter] = lambda: fake

    resp = client.get("/api/data_status")

    row = _hotspot_row(resp.json())
    assert row["percent"] == 37.5
    assert row["detail"] == "3/8 hotspot cell(s) populated"
    assert len(row["segments"]) == 8
    assert sum(1 for seg in row["segments"] if seg["queried"]) == 3


def test_flightradar_hotspot_shows_full_progress(client):
    _override_all_empty()
    fake = FakeProcessStatusAdapter()
    fake.record_progress("flightradar_hotspot", "layer", 8, 8)
    app.dependency_overrides[get_process_status_adapter] = lambda: fake

    resp = client.get("/api/data_status")

    row = _hotspot_row(resp.json())
    assert row["percent"] == 100.0
    assert all(seg["queried"] for seg in row["segments"])


def test_flightradar_hotspot_is_a_layer_row_not_gated_by_a_channel(client):
    _override_all_empty()
    app.dependency_overrides[get_process_status_adapter] = lambda: FakeProcessStatusAdapter()

    resp = client.get("/api/data_status")

    row = _hotspot_row(resp.json())
    assert row["kind"] == "layer"
    assert row["channel_key"] is None
    assert row["channel_on"] is None
    assert row["display_name"] == "Flight Radar Hotspot"
