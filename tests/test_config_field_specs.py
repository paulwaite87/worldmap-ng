#!/usr/bin/env python3
"""Tests for the schema-driven "Global" tab of the config UI (architecture review
candidate "htmx for the configuration UI", vertical slice). FIELD_SPECS/field_label/
format_slider_badge/validate_against_specs in routes/field_specs.py replace the
~46-branch option-name dispatch in the legacy ui/config/index.html JS; these tests
lock the pure spec functions and the two routes (GET /config, POST /api/config) that
consume them.
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from atmos_gl.api import app
from atmos_gl.lib.auth import SESSION_COOKIE_NAME
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.routes.auth import get_user_adapter
from atmos_gl.routes.field_specs import (
    SliderSpec,
    FIELD_SPECS,
    field_label,
    section_label,
    format_slider_badge,
    clamp_slider_value,
    to_display_value,
    initial_color_render,
    is_long_or_url_field,
    is_api_key_field,
    validate_against_specs,
)
from tests.conftest import make_signed_in_session

client = TestClient(app)


# GET /config, GET /config/section_defaults/{section}, and POST /api/config are gated by
# require_admin (issue #304); this file's `client` is a module-level singleton reused
# across every test below (never cleared), so authenticate it as admin here rather than
# at each of the ~25 call sites. autouse + re-applied per test (not a one-time module-
# level assignment) because another file's `client` fixture teardown
# (app.dependency_overrides.clear()) can run between collection and these tests actually
# executing when the whole suite runs together, wiping a one-time override. These tests
# exercise the config UI/routes' existing behaviour, not the admin gate itself (see
# tests/test_admin_gated_routes.py for that).
@pytest.fixture(autouse=True)
def _admin_session():
    fake, token = make_signed_in_session(is_admin=True)
    app.dependency_overrides[get_user_adapter] = lambda: fake
    client.cookies.set(SESSION_COOKIE_NAME, token)


# --- Pure spec helpers ---


def test_field_label_uses_override_for_animation_fields():
    assert field_label("animation", "stepping_rate") == "Forecast stepping rate"
    assert (
        field_label("animation", "forecast_stepping")
        == "Forecast stepping (hourly playback)"
    )


def test_field_label_falls_back_to_spaced_capitalised():
    assert field_label("common", "auto_rotate_speed") == "Auto rotate speed"


def test_format_slider_badge_raw_when_no_decimals():
    spec = SliderSpec(min=0, max=100, step=1)
    assert format_slider_badge(spec, 45) == "45"


def test_format_slider_badge_applies_decimals_and_suffix():
    spec = SliderSpec(min=-90, max=90, step=1, decimals=1, suffix=" deg")
    assert format_slider_badge(spec, 12.345) == "12.3 deg"


def test_clamp_slider_value_passes_through_in_range_values():
    spec = SliderSpec(min=-90, max=90, step=1)
    assert clamp_slider_value(spec, 45) == 45


def test_clamp_slider_value_clamps_a_stale_out_of_range_value():
    """Guards the badge/slider-position mismatch that a stored value outside the
    (now-corrected) range would otherwise cause -- e.g. a starting_latitude left
    over from before the swapped-range bug was fixed."""
    spec = SliderSpec(min=-90, max=90, step=1)
    assert clamp_slider_value(spec, 165) == 90
    assert clamp_slider_value(spec, -165) == -90


def test_starting_lat_lon_ranges_are_geographically_correct():
    """Regression guard for the swapped min/max bug in the legacy JS (latitude got
    +/-180, longitude got +/-90)."""
    lat = FIELD_SPECS[("common", "starting_latitude")]
    lon = FIELD_SPECS[("common", "starting_longitude")]
    assert (lat.min, lat.max) == (-90.0, 90.0)
    assert (lon.min, lon.max) == (-180.0, 180.0)


def test_validate_against_specs_accepts_in_range_slider():
    assert validate_against_specs({"common": {"auto_rotate_speed": 0.5}}) == []


def test_validate_against_specs_rejects_out_of_range_slider():
    errors = validate_against_specs({"common": {"auto_rotate_speed": 99}})
    assert len(errors) == 1
    assert "auto_rotate_speed" in errors[0]


def test_validate_against_specs_rejects_unknown_select_option():
    errors = validate_against_specs({"common": {"basemap": "not-a-real-style"}})
    assert len(errors) == 1


def test_validate_against_specs_accepts_int_value_against_string_select_options():
    """level_of_detail is stored/posted as an int (1) but SelectSpec options are
    declared as strings ("1") -- regression guard: a legitimate value must not be
    rejected."""
    assert validate_against_specs({"precipitation": {"level_of_detail": 1}}) == []


def test_validate_against_specs_ignores_fields_without_a_spec():
    """Fields with no FIELD_SPECS entry stay permissive, matching legacy behaviour --
    both for genuinely generic fields and for tabs not yet migrated."""
    assert validate_against_specs({"common": {"workdir": "literally anything"}}) == []


def test_validate_against_specs_ignores_missing_sections():
    assert validate_against_specs({"quakes": {"min_mag": 4.5}}) == []


# --- personalizable (issue #305/#314): opt-in per key, default False ---


def test_personalizable_defaults_false_on_a_fresh_spec():
    assert SliderSpec(min=0, max=100, step=1).personalizable is False


def test_precipitations_opacity_palette_and_min_mm_hr_are_personalizable():
    """The three keys #314 proves the personalization mechanism on -- #315 curates the
    rest of the app's fields, including precipitation's own enabled/level_of_detail/
    cache_expiry_days, which stay non-personalizable here."""
    assert FIELD_SPECS[("precipitation", "opacity")].personalizable is True
    assert FIELD_SPECS[("precipitation", "palette")].personalizable is True
    assert FIELD_SPECS[("precipitation", "min_mm_hr")].personalizable is True


def test_precipitations_non_personalizable_keys_stay_false():
    """enabled became personalizable under #315 (every layer's visibility flag did,
    via _ENABLED_PERSONALIZABLE) -- level_of_detail/cache_expiry_days are genuine
    collector-cost concerns and stay non-personalizable everywhere."""
    assert FIELD_SPECS[("precipitation", "enabled")].personalizable is True
    assert FIELD_SPECS[("precipitation", "level_of_detail")].personalizable is False
    assert FIELD_SPECS[("precipitation", "cache_expiry_days")].personalizable is False


def test_most_other_sections_opacity_is_personalizable_after_315():
    """#314 deliberately gave precipitation.opacity its own dedicated SliderSpec so
    flagging it personalizable couldn't silently flip every section sharing _OPACITY.
    #315 then flipped the shared _OPACITY constant itself, so every section reusing it
    (sst/currents/temperature among them) is personalizable too -- EXCEPT isobars,
    volcanoes and air_quality, whose opacity is baked server-side or has no live
    effect, so those three get their own dedicated, non-personalizable instances
    instead of the shared one (see field_specs.py's own comments)."""
    assert FIELD_SPECS[("sst", "opacity")].personalizable is True
    assert FIELD_SPECS[("currents", "opacity")].personalizable is True
    assert FIELD_SPECS[("temperature", "opacity")].personalizable is True
    assert FIELD_SPECS[("isobars", "opacity")].personalizable is False
    assert FIELD_SPECS[("air_quality", "opacity")].personalizable is False


# --- Events / Misc / Shipping batch: prefix badges, shared shapes, new kinds ---


def test_field_label_section_specific_override_for_quakes_min_mag():
    assert field_label("quakes", "min_mag") == "Minimum magnitude"


def test_format_slider_badge_applies_prefix():
    spec = FIELD_SPECS[("quakes", "min_mag")]
    assert format_slider_badge(spec, 4.5) == "M 4.5"


def test_runs_per_day_has_no_field_specs_entry_anywhere():
    """runs_per_day moved to the Data Status page's per-row widget (routes/status.py's
    set_runs_per_day) -- it must not have a FIELD_SPECS entry for any section, whether
    still-real (quakes, clouds, ...) or fully vestigial (isobars, wind, ...)."""
    assert not any(option == "runs_per_day" for _, option in FIELD_SPECS)


def test_satellites_collector_and_data_collector_lost_their_bespoke_cadence_specs():
    """update_hours/update_minutes were replaced by runs_per_day (also not in
    FIELD_SPECS -- see test_runs_per_day_has_no_field_specs_entry_anywhere)."""
    assert ("satellites_collector", "update_hours") not in FIELD_SPECS
    assert ("data_collector", "update_minutes") not in FIELD_SPECS


def test_no_backend_collector_section_has_a_personalizable_key():
    """Regression guard for issue #305/#315: personalizable=True is only ever safe on a
    setting that's pure client-side display (see _merge_personal_overrides's docstring
    in routes/config.py) -- a collector/housekeeper section's keys gate backend
    data-acquisition cost, not anything rendered on the map, so none of them may ever be
    personalizable. Without this test, someone could flip one True in field_specs.py and
    every other test would still pass -- config_field_specs.py has no other check tying
    "collector section" to "never personalizable"."""
    collector_sections = {
        "shipping_collector", "lightning_collector", "flightradar_collector",
        "satellites_collector", "data_collector", "housekeeper",
    }
    offending = [
        (section, option)
        for (section, option), spec in FIELD_SPECS.items()
        if section in collector_sections and getattr(spec, "personalizable", False)
    ]
    assert offending == []


def test_initial_color_render_resolves_named_color_to_hex():
    assert initial_color_render("Violet") == ("#ee82ee", "Violet")


def test_initial_color_render_passes_through_raw_hex():
    assert initial_color_render("#070b18") == ("#070b18", "#070b18")


def test_initial_color_render_defaults_empty_value_to_white():
    assert initial_color_render("") == ("#ffffff", "White")


def test_is_long_or_url_field_flags_url_named_options():
    assert is_long_or_url_field("url", "short") is True


def test_is_long_or_url_field_flags_long_values_regardless_of_name():
    assert is_long_or_url_field("outfile", "x" * 40) is True


def test_is_long_or_url_field_false_for_short_non_url_values():
    assert is_long_or_url_field("outfile", "data/quakes.json") is False


def test_is_api_key_field_matches_injected_secret_fields():
    assert is_api_key_field("api_key") is True
    assert is_api_key_field("min_mag") is False


def test_validate_against_specs_accepts_valid_multiselect_subset():
    assert validate_against_specs({"satellites": {"sat_names": ["ISS (ZARYA)", "HST"]}}) == []


def test_validate_against_specs_rejects_multiselect_with_unknown_option():
    errors = validate_against_specs({"satellites": {"sat_names": ["ISS (ZARYA)", "nope"]}})
    assert len(errors) == 1


def test_validate_against_specs_rejects_non_list_multiselect_value():
    errors = validate_against_specs({"satellites": {"sat_names": "ISS (ZARYA)"}})
    assert len(errors) == 1


def test_validate_against_specs_accepts_valid_grouped_transfer_subset():
    assert validate_against_specs(
        {"satellites_collector": {"groups": ["stations", "starlink"]}}
    ) == []


def test_validate_against_specs_rejects_grouped_transfer_with_unknown_option():
    errors = validate_against_specs(
        {"satellites_collector": {"groups": ["stations", "nope"]}}
    )
    assert len(errors) == 1


def test_validate_against_specs_rejects_non_list_grouped_transfer_value():
    errors = validate_against_specs({"satellites_collector": {"groups": "stations"}})
    assert len(errors) == 1


# --- GET /config: Events / Misc / Shipping tabs render correctly ---


def test_config_page_renders_prefixed_slider_badge():
    resp = client.get("/config")
    assert 'id="badge-quakes__min_mag"' in resp.text


def _write_temp_config(tmp_path, **sections):
    """Writes a minimal config JSON containing just the given sections and points
    CONFIG_PATH at it (same pattern as test_status_route.py's _write_temp_config) --
    render_tab_group's `if section_data is not none` guard skips any section absent
    from the dict, so a fixture only needs the section(s) a given test actually cares
    about. Isolates a test from whatever the real config/atmos-gl.json currently holds
    -- that file is gitignored and "drifts constantly during normal use" (CLAUDE.md's
    Settings changes section), so asserting against its live values (as three tests
    here used to) is inherently flaky: it only passes by coincidence of the dev
    environment's current state, not anything the code under test guarantees."""
    path = tmp_path / "atmos-gl.json"
    path.write_text(json.dumps(sections))
    return path


def test_config_page_selects_correct_option_despite_stored_int_vs_string_options(tmp_path, monkeypatch):
    """precipitation.level_of_detail is stored as an int (1) here but SelectSpec
    options are strings ("1") -- regression guard for the type-mismatch bug this
    exposed."""
    config_path = _write_temp_config(
        tmp_path,
        precipitation={
            "enabled": False, "level_of_detail": 1, "min_mm_hr": 0.2,
            "opacity": 75, "palette": "standard", "cache_expiry_days": 3,
        },
    )
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/config")
    html = resp.text
    idx = html.index('id="precipitation__level_of_detail"')
    select_html = html[idx : idx + 800]
    assert '<option value="1" selected>' in select_html


def test_config_page_renders_multiselect_with_correct_options_checked(tmp_path, monkeypatch):
    config_path = _write_temp_config(
        tmp_path,
        satellites={
            "enabled": False, "sat_names": ["ISS (ZARYA)"], "extra_satellite_names": "",
            "past_minutes": 3, "future_minutes": 45, "step_seconds": 30, "color": "White",
        },
    )
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/config")
    html = resp.text
    assert 'id="satellites__sat_names"' in html
    assert 'array-select' in html
    idx = html.index('id="satellites__sat_names"')
    select_html = html[idx : idx + 1500]
    assert '<option value="ISS (ZARYA)" selected>' in select_html


def test_config_page_renders_grouped_transfer_with_active_options_on_the_right(tmp_path, monkeypatch):
    """satellites_collector.groups -- active groups render in the right ("active"/
    saved) select, grouped under the same <optgroup> headings as the left
    ("available") select, and never in both at once."""
    config_path = _write_temp_config(
        tmp_path,
        satellites_collector={
            "enabled": True, "groups": ["stations", "weather", "science", "resource"],
            "runs_per_day": 6, "log_level": "INFO",
        },
    )
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    resp = client.get("/config")
    html = resp.text
    assert 'id="satellites_collector__groups__available"' in html
    assert 'id="satellites_collector__groups"' in html
    assert '<optgroup label="Special-Interest Satellites">' in html

    avail_idx = html.index('id="satellites_collector__groups__available"')
    active_idx = html.index('id="satellites_collector__groups"', avail_idx)
    available_html = html[avail_idx:active_idx]
    active_html = html[active_idx : html.index("</select>", active_idx)]

    # "stations" is active (in the fixture) -> right box only.
    assert '<option value="stations">' in active_html
    assert '<option value="stations">' not in available_html
    # "starlink" is not active -> left box only.
    assert '<option value="starlink">' in available_html
    assert '<option value="starlink">' not in active_html


def test_config_page_renders_color_picker_with_resolved_hex():
    resp = client.get("/config")
    html = resp.text
    assert 'id="markers__marker_color"' in html
    assert 'value="#ffffff"' in html  # White


def test_config_page_renders_unstructured_color_for_terminator():
    """terminator.shade_color saves as raw hex, not a named colour -- must not carry
    the structured-color-name-picker class."""
    resp = client.get("/config")
    html = resp.text
    idx = html.index('id="terminator__shade_color"')
    input_html = html[max(0, idx - 300) : idx + 50]
    assert "structured-color-name-picker" not in input_html


# is_long_or_url_field's full-width-by-name behaviour used to have a live example here
# (storms.jtwc_url) -- removed once storms' ATCF mirror URLs moved into
# data_collector.datasources (rendered via the dedicated accordion, not this generic
# fallback path), leaving no bare url-named field on the page anymore. The pure logic
# stays covered by test_is_long_or_url_field_flags_url_named_options and its siblings
# above.


# --- GET /config: renders the schema-driven Global tab ---


def test_config_page_renders_slider_bounds_and_fixed_lat_lon_ranges():
    resp = client.get("/config")
    assert resp.status_code == 200
    html = resp.text
    assert 'min="0.01"' in html and 'max="1.0"' in html  # auto_rotate_speed
    assert 'min="-90.0"' in html  # starting_latitude, fixed
    assert 'min="-180.0"' in html  # starting_longitude, fixed


def test_config_page_renders_select_options_with_current_value_selected():
    resp = client.get("/config")
    html = resp.text
    assert '<option value="satellite"' in html
    assert "selected" in html


def test_config_page_renders_toggle_as_checkbox():
    resp = client.get("/config")
    html = resp.text
    assert 'id="common__atmosphere"' in html
    assert 'type="checkbox"' in html


def test_config_page_falls_back_to_text_input_for_unspecced_field():
    """satellites.extra_satellite_names has no FIELD_SPECS entry -- must still render
    via the generic fallback."""
    resp = client.get("/config")
    html = resp.text
    assert 'id="satellites__extra_satellite_names"' in html


# --- POST /api/config: spec-based validation ---


def test_update_config_rejects_out_of_range_slider():
    resp = client.post("/api/config", json={"common": {"auto_rotate_speed": 99}})
    assert resp.status_code == 422


def test_update_config_rejects_invalid_select_option():
    resp = client.post("/api/config", json={"common": {"basemap": "not-a-real-style"}})
    assert resp.status_code == 422


def test_update_config_accepts_valid_payload(tmp_path):
    """Uses a throwaway config file so this test can't corrupt config/atmos-gl.json."""
    tmp_config = tmp_path / "atmos-gl.json"
    tmp_config.write_text('{"common": {"auto_rotate_speed": 0.5}}')

    with patch(
        "atmos_gl.routes.config.load_config",
        return_value=AtmosGLConfig(str(tmp_config)),
    ):
        resp = client.post("/api/config", json={"common": {"auto_rotate_speed": 0.5}})

    assert resp.status_code == 200
    assert json.loads(tmp_config.read_text())["common"]["auto_rotate_speed"] == 0.5


def test_update_config_strips_outfile_before_saving(tmp_path):
    """outfile is injected read-time-only by _build_config_data() (see OUTFILES) --
    the browser's masterConfigCache echoes it back on every save (it has no
    dynamic-input to override it), so it must be stripped here or it'd get written to
    disk as if it were a real stored setting."""
    tmp_config = tmp_path / "atmos-gl.json"
    tmp_config.write_text('{"isobars": {"opacity": 50}}')

    with patch(
        "atmos_gl.routes.config.load_config",
        return_value=AtmosGLConfig(str(tmp_config)),
    ):
        resp = client.post(
            "/api/config",
            json={"isobars": {"opacity": 50, "outfile": "data/isobars.png"}},
        )

    assert resp.status_code == 200
    saved = json.loads(tmp_config.read_text())
    assert saved["isobars"]["opacity"] == 50
    assert "outfile" not in saved["isobars"]


# --- Weather / Climate batch: whole-step int display, sentinel badges,
# byte<->percent transform, unspecced-boolean fallback, section-conditional selects ---


def test_clamp_slider_value_returns_int_for_whole_step_sliders():
    """Regression guard: clamp_slider_value used to always float()-coerce, so a
    whole-step slider's badge showed "12.0px" instead of "12px" for any value that
    wasn't exactly at a boundary (the earlier scalar-field-style bug this batch's
    live verification caught)."""
    spec = SliderSpec(min=6, max=24, step=1)
    result = clamp_slider_value(spec, 12)
    assert result == 12
    assert isinstance(result, int)


def test_clamp_slider_value_keeps_float_for_fractional_step_sliders():
    spec = SliderSpec(min=0, max=5, step=0.25)
    result = clamp_slider_value(spec, 0.5)
    assert result == 0.5
    assert isinstance(result, float)


def test_format_slider_badge_whole_step_has_no_decimal_point():
    spec = SliderSpec(min=0, max=5000, step=100, suffix="J/Kg")
    assert format_slider_badge(spec, clamp_slider_value(spec, 1200)) == "1200J/Kg"


def test_format_slider_badge_zero_label_overrides_normal_formatting():
    spec = SliderSpec(min=0, max=5, step=0.25, suffix=" m", zero_label="off")
    assert format_slider_badge(spec, 0) == "off"
    assert format_slider_badge(spec, 0.5) == "0.5 m"


def test_format_slider_badge_pluralizes_suffix_based_on_count():
    spec = FIELD_SPECS[("clouds", "cache_expiry_days")]
    assert format_slider_badge(spec, 0) == "keep forever"
    assert format_slider_badge(spec, 1) == "1 day"
    assert format_slider_badge(spec, 5) == "5 days"


def test_to_display_value_converts_byte_to_percent_for_clouds_threshold():
    spec = FIELD_SPECS[("clouds", "threshold")]
    assert to_display_value(spec, 168) == 66  # round((168/255)*100)


def test_to_display_value_is_a_noop_for_ordinary_sliders():
    spec = FIELD_SPECS[("quakes", "min_mag")]
    assert to_display_value(spec, 4.5) == 4.5


def test_validate_against_specs_uses_raw_max_for_byte_to_percent_field():
    """clouds.threshold is posted in raw byte space (0-255), not the displayed
    slider's 0-100 percent space -- validation must check against the byte range."""
    assert validate_against_specs({"clouds": {"threshold": 255}}) == []
    errors = validate_against_specs({"clouds": {"threshold": 256}})
    assert len(errors) == 1
    assert "255" in errors[0]  # bound reported is the raw max, not 100


def test_shared_constants_reused_across_many_sections():
    """_ALPHA, _LEVEL_OF_DETAIL etc. are declared once and referenced under every
    field that needs them -- not redeclared per section. isobars.opacity is
    deliberately EXCLUDED from this (see field_specs.py's own comment): it has no
    live effect on the WebGL fill-mode render, so #315 gave it its own dedicated,
    non-personalizable instance rather than the shared, now-personalizable _OPACITY."""
    assert FIELD_SPECS[("wind", "opacity")] is FIELD_SPECS[("sst", "opacity")]
    assert (
        FIELD_SPECS[("isobars", "level_of_detail")]
        is FIELD_SPECS[("stormwatch", "level_of_detail")]
    )
    assert (
        FIELD_SPECS[("currents", "particle_speed")]
        is FIELD_SPECS[("animation", "stepping_rate")]
    )


def test_section_conditional_palette_options_differ_per_section():
    """palette is one legacy branch keyed on section -- each section's option list
    must stay independent, not accidentally share one shared constant."""
    sst_values = {v for v, _ in FIELD_SPECS[("sst", "palette")].options}
    ozone_values = {v for v, _ in FIELD_SPECS[("ozone", "palette")].options}
    pwat_values = {v for v, _ in FIELD_SPECS[("pwat", "palette")].options}
    assert sst_values == {"thermal", "vivid", "deep", "ocean"}
    assert ozone_values == {"alert", "high_contrast"}
    assert pwat_values == {"standard", "atmospheric_river", "deep_teal"}
    assert sst_values.isdisjoint(ozone_values)
    assert sst_values.isdisjoint(pwat_values)


# --- GET /config: Weather / Climate tabs render correctly ---


def test_config_page_renders_byte_to_percent_slider_with_extra_class():
    resp = client.get("/config")
    html = resp.text
    idx = html.index('id="clouds__threshold"')
    input_html = html[max(0, idx - 100) : idx + 200]
    assert 'max="100"' in input_html  # displayed range, not the raw 0-255
    assert "cloud-threshold-slider" in input_html


def test_config_page_renders_unspecced_boolean_as_toggle_not_number_fallback():
    """ozone.stormwatch has no FIELD_SPECS entry distinct from other booleans --
    still must render as a checkbox, not a broken type=number input with value=True."""
    resp = client.get("/config")
    html = resp.text
    idx = html.index('id="ozone__stormwatch"')
    input_html = html[max(0, idx - 50) : idx + 50]
    assert 'type="checkbox"' in input_html


def test_config_page_renders_prefixed_gamma_slider():
    resp = client.get("/config")
    assert 'id="badge-clouds__gamma"' in resp.text


# --- Background batch: final tab, shared log_level, datasources accordion,
# the fallback-section-X regression fix, and the dead legacy JS deletion ---


def test_shared_log_level_reused_across_common_and_collector_sections():
    assert FIELD_SPECS[("common", "log_level")] is FIELD_SPECS[("data_collector", "log_level")]
    assert (
        FIELD_SPECS[("shipping_collector", "log_level")]
        is FIELD_SPECS[("lightning_collector", "log_level")]
    )


def test_format_slider_badge_combines_prefix_and_pluralized_suffix():
    spec = FIELD_SPECS[("housekeeper", "days_between_runs")]
    assert format_slider_badge(spec, 1) == "every 1 day"
    assert format_slider_badge(spec, 5) == "every 5 days"


def test_housekeeper_enabled_has_an_explicit_spec():
    """Every section's enabled setting has a real FIELD_SPECS entry (issue #313), but
    housekeeper is the only one NOT also skipped by render_tab_group's generic
    'enabled' filter -- everywhere else it stays validated but rendered exclusively by
    the Show tab's own dedicated grid, not inline in the properties tab."""
    assert FIELD_SPECS[("housekeeper", "enabled")].kind == "toggle"


@pytest.mark.parametrize("section", [
    "quakes", "volcanoes", "fires", "satellites", "terminator", "markers", "flightradar",
    "landmass", "shipping", "clouds", "isobars", "wind", "jetstream", "precipitation",
    "pwat", "lightning", "storms", "sst", "currents", "waves", "temperature", "ozone",
    "stormwatch", "greenhouse_gases", "air_quality", "shipping_collector",
    "lightning_collector", "satellites_collector", "data_collector",
])
def test_every_show_tab_section_has_an_explicit_enabled_spec(section):
    """Issue #313: layer visibility is promoted into FIELD_SPECS for every section
    with a Show-tab checkbox/radio, so it's validated like every other setting."""
    assert FIELD_SPECS[(section, "enabled")].kind == "toggle"


def test_validate_against_specs_now_validates_enabled_as_a_boolean():
    """Before #313, `enabled` had no FIELD_SPECS entry for these sections, so
    validate_against_specs silently accepted any value type. Now it's checked like
    every other toggle."""
    assert validate_against_specs({"clouds": {"enabled": True}}) == []
    errors = validate_against_specs({"clouds": {"enabled": "yes"}})
    assert len(errors) == 1
    assert "clouds.enabled" in errors[0]


@pytest.mark.parametrize("section", [
    "clouds", "shipping", "shipping_collector",
    "sst", "currents", "temperature", "ozone", "stormwatch", "greenhouse_gases",
])
def test_config_page_still_renders_each_enabled_checkbox_exactly_once(section):
    """Regression guard: now that these sections have a FIELD_SPECS entry for
    `enabled`, render_field_group's existing exclusion (_field_macros.html) must still
    keep it out of the generic properties-tab rendering, so the Show tab's own
    hardcoded checkbox remains the only one -- no duplicate id in the page. Climate
    sections (sst/currents/.../greenhouse_gases) used to render as a mutually-exclusive
    radio group (`radio__{section}`) instead -- since independent Climate layers can now
    be shown together, they're plain `{section}__enabled` checkboxes like every other
    section."""
    resp = client.get("/config")
    html = resp.text
    assert html.count(f'id="{section}__enabled"') == 1


def test_config_page_renders_housekeeper_enabled_as_a_visible_toggle():
    resp = client.get("/config")
    html = resp.text
    idx = html.index('id="housekeeper__enabled"')
    input_html = html[max(0, idx - 50) : idx + 50]
    assert 'type="checkbox"' in input_html


def test_config_page_renders_datasources_accordion_with_existing_entries():
    """data_collector.datasources deliberately has no FIELD_SPECS entry -- it's
    rendered by its own dedicated macro (render_datasources_accordion), mirroring
    the legacy buildDatasourcesHTML() JS function server-side."""
    resp = client.get("/config")
    html = resp.text
    assert 'id="datasources-accordion-data_collector"' in html
    idx = html.index('id="datasources-accordion-data_collector"')
    accordion_html = html[idx : idx + 3500]
    assert ">gfs<" in accordion_html
    assert ">currents<" in accordion_html
    assert "addDatasource('data_collector')" in html[idx:]


def test_config_page_never_renders_channel_enabled_as_a_generic_field():
    """Regression guard: channel_enabled is a dict with no FIELD_SPECS entry, same as
    datasources, but has no dedicated accordion here -- it's edited exclusively via
    the Data Status tab's toggle switches (routes/status.py's POST endpoint). Without
    an explicit exclusion it falls through to render_field's generic text-input
    fallback, which stringifies the dict -- saving the Background tab's form then
    writes that string back over the real dict and breaks GET /api/data_status
    (channel_enabled.get(...) on a str)."""
    resp = client.get("/config")
    html = resp.text
    assert 'id="data_collector__channel_enabled"' not in html


def test_config_page_never_renders_runs_per_day_as_a_generic_field():
    """Regression guard: runs_per_day still lives in quakes'/clouds'/etc. config dict
    (just edited via the Data Status tab now, not here), so without the
    _field_macros.html exclusion it would fall through to the generic number-input
    fallback and reappear on the Settings tab."""
    resp = client.get("/config")
    html = resp.text
    assert 'id="quakes__runs_per_day"' not in html
    assert 'id="clouds__runs_per_day"' not in html


def test_config_page_renders_fallback_section_for_gated_layers():
    """Regression guard: render_tab_group previously never emitted the
    fallback-section-X div toggleSectionVisibility() depends on, so toggling a
    layer off in the Show tab silently stopped hiding its settings fields."""
    resp = client.get("/config")
    html = resp.text
    assert 'id="fallback-section-quakes"' in html
    assert "Layer Display Off" in html


def test_config_page_omits_fallback_section_for_exempt_sections():
    resp = client.get("/config")
    html = resp.text
    assert 'id="fallback-section-common"' not in html
    assert 'id="fallback-section-housekeeper"' not in html


def test_section_label_matches_the_show_tab_wording():
    assert section_label("pwat") == "Precipitable Water"
    assert section_label("sst") == "Sea Surface Temp"
    assert section_label("storms") == "Storm Track"
    assert section_label("temperature") == "Air Temperature"
    assert section_label("waves") == "Waves"


def test_section_label_falls_back_to_title_case_for_sections_without_a_show_tab_entry():
    assert section_label("map_builder") == "Map Builder"
    assert section_label("animation") == "Animation"


def test_config_page_renders_friendly_section_headings_not_raw_bracket_keys():
    """The settings heading and the "enable it in the Show tab" fallback prompt both
    used to show the raw config key in brackets (e.g. "[pwat] Properties") -- both now
    use the same friendly name the Show tab itself uses for that layer's toggle."""
    resp = client.get("/config")
    html = resp.text
    assert "Precipitable Water Properties" in html
    assert "[pwat] Properties" not in html
    assert "Enable <strong>Earthquakes</strong> in the Show tab to edit." in html
    assert "[quakes]" not in html


def test_config_page_renders_pwat_as_a_plain_toggle_not_a_climate_radio():
    """pwat isn't mutually exclusive with the sst/currents/temperature/ozone/
    stormwatch climate base layer -- it must get its own Show-tab checkbox (like
    precipitation), never a radio__pwat entry in the exclusive_climate group."""
    resp = client.get("/config")
    html = resp.text
    assert 'type="checkbox" id="pwat__enabled"' in html
    assert 'id="radio__pwat"' not in html


def test_config_page_renders_waves_as_a_plain_toggle_not_a_climate_radio():
    """waves moved off the Climate radio group onto the Weather tab as a plain
    checkbox (like wind) -- it doesn't cover landmasses the way wind's heatmap does,
    so it has much less potential to visually clash with another base layer, and it's
    GFS-sourced (like wind/jetstream) rather than the RTOFS source currents uses."""
    resp = client.get("/config")
    html = resp.text
    assert 'type="checkbox" id="waves__enabled"' in html
    assert 'id="radio__waves"' not in html


def test_config_page_renders_pwat_fields_section_and_gated_fallback():
    resp = client.get("/config")
    html = resp.text
    assert 'id="fields-section-pwat"' in html
    assert 'id="fallback-section-pwat"' in html
    assert 'id="pwat__critical_pwat"' in html
    assert 'id="pwat__palette"' in html


# --- "Set to Defaults" button + GET /config/section_defaults/{section} ---


def test_config_page_renders_set_to_defaults_button_opposite_every_section_title():
    resp = client.get("/config")
    html = resp.text
    assert "resetSectionToDefaults('precipitation', 'Precipitation Properties')" in html
    assert "resetSectionToDefaults('common', 'Global Settings')" in html


def test_section_defaults_route_renders_values_from_the_template_not_the_live_config():
    """config/atmos-gl.json.tmpl's air_quality.pm2_5_min (35) must come through
    regardless of whatever the live config.json currently holds for that field."""
    resp = client.get("/config/section_defaults/air_quality")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="air_quality__pm2_5_min"' in html
    idx = html.index('id="air_quality__pm2_5_min"')
    assert 'value="35"' in html[idx : idx + 300]


def test_section_defaults_route_omits_the_enabled_field_same_as_the_live_page():
    resp = client.get("/config/section_defaults/precipitation")
    assert 'id="precipitation__enabled"' not in resp.text


def test_section_defaults_route_404s_for_an_unknown_section():
    resp = client.get("/config/section_defaults/not_a_real_section")
    assert resp.status_code == 404


def test_config_page_has_no_remaining_legacy_dispatch_code():
    """TAB_GROUPS/renderTabContainers became fully dead once every tab migrated --
    this guards against either being silently reintroduced."""
    resp = client.get("/config")
    html = resp.text
    assert "TAB_GROUPS" not in html
    assert "renderTabContainers" not in html


def test_config_page_still_has_the_interactive_datasource_functions():
    """These stay -- they handle add/remove/rename after initial load, unrelated
    to the deleted TAB_GROUPS-driven dispatch."""
    resp = client.get("/config")
    html = resp.text
    for fn in ("addDatasource", "updateDatasourceName", "updateDatasourceUrl", "deleteDatasource"):
        assert f"function {fn}" in html


# --- Jet Stream (#184) ---


def test_jetstream_reuses_currents_shaped_shared_constants():
    """Jet stream is speed-colored particles with no heatmap, like currents -- it
    shares currents' particle-tuning constants, not wind's rescaled ones."""
    assert (
        FIELD_SPECS[("jetstream", "particle_speed")]
        is FIELD_SPECS[("currents", "particle_speed")]
    )
    assert (
        FIELD_SPECS[("jetstream", "trail_length")]
        is FIELD_SPECS[("currents", "trail_length")]
    )
    assert (
        FIELD_SPECS[("jetstream", "trail_thickness")]
        is FIELD_SPECS[("currents", "trail_thickness")]
    )
    assert (
        FIELD_SPECS[("jetstream", "opacity")] is FIELD_SPECS[("currents", "opacity")]
    )


def test_jetstream_reuses_winds_flow_coherence_radius_not_currents_lack_of_one():
    """Jetstream reads the same noisy 0.25deg GFS grid wind does (unlike currents'
    smooth RTOFS source, which has no flow_coherence_radius spec at all) -- it must
    share WIND's coherence-smoothing spec, not currents' particle-tuning shape."""
    assert (
        FIELD_SPECS[("jetstream", "flow_coherence_radius")]
        is FIELD_SPECS[("wind", "flow_coherence_radius")]
    )
    assert ("currents", "flow_coherence_radius") not in FIELD_SPECS


def test_jetstream_palette_options_match_the_backend_updater():
    """Must stay in sync with JetStreamUpdater.PALETTES (tasks/jetstream.py) -- an
    option here with no matching backend palette would 500 on save/render."""
    values = {v for v, _ in FIELD_SPECS[("jetstream", "palette")].options}
    assert values == {"stratosphere", "aurora", "inferno"}


def test_jetstream_has_no_ocean_only_fields():
    """Unlike currents, jetstream has no land/sea distinction (250mb wind blows
    over both) and no regrid-for-crispness step -- current_speed_minimum/
    fill_floor/fill_knee must not be carried over."""
    for option in ("current_speed_minimum", "fill_floor", "fill_knee"):
        assert ("jetstream", option) not in FIELD_SPECS


def test_config_page_renders_jetstream_toggle_on_show_tab():
    resp = client.get("/config")
    html = resp.text
    assert 'type="checkbox" id="jetstream__enabled"' in html


def test_config_page_renders_jetstream_fields_section_and_gated_fallback():
    resp = client.get("/config")
    html = resp.text
    assert 'id="fields-section-jetstream"' in html
    assert 'id="fallback-section-jetstream"' in html
    assert 'id="jetstream__palette"' in html
    assert 'id="jetstream__trail_length"' in html
    assert 'id="jetstream__flow_coherence_radius"' in html


def test_config_page_renders_jetstream_palette_select_with_stratosphere_option():
    resp = client.get("/config")
    html = resp.text
    idx = html.index('id="jetstream__palette"')
    select_html = html[idx : html.index("</select>", idx)]
    assert '<option value="stratosphere"' in select_html
    assert '<option value="aurora"' in select_html
    assert '<option value="inferno"' in select_html


# --- Landmass outlines ---


def test_landmass_reuses_the_shared_opacity_spec():
    assert FIELD_SPECS[("landmass", "opacity")] is FIELD_SPECS[("sst", "opacity")]


def test_landmass_has_two_independent_color_fields():
    """Main stroke + halo are separate, independently-saved colours (see
    ui/modules/landmass.js's halo technique), not one shared colour spec."""
    assert FIELD_SPECS[("landmass", "color")].kind == "color"
    assert FIELD_SPECS[("landmass", "halo_color")].kind == "color"
    assert (
        FIELD_SPECS[("landmass", "color")]
        is not FIELD_SPECS[("landmass", "halo_color")]
    )


def test_section_label_for_landmass_matches_the_show_tab_wording():
    assert section_label("landmass") == "Landmass Outlines"


def test_config_page_renders_landmass_toggle_on_show_tab():
    resp = client.get("/config")
    html = resp.text
    assert 'type="checkbox" id="landmass__enabled"' in html


def test_config_page_renders_landmass_fields_section_and_gated_fallback():
    resp = client.get("/config")
    html = resp.text
    assert 'id="fields-section-landmass"' in html
    assert 'id="fallback-section-landmass"' in html
    assert 'id="landmass__color"' in html
    assert 'id="landmass__halo_color"' in html
    assert 'id="landmass__linewidth"' in html


# --- Flood Risk (issue #371) ---


def test_config_page_renders_flood_risk_toggle_on_show_tab():
    resp = client.get("/config")
    html = resp.text
    assert 'type="checkbox" id="flood_risk__enabled"' in html


def test_config_page_renders_flood_risk_fields_section_and_gated_fallback():
    resp = client.get("/config")
    html = resp.text
    assert 'id="fields-section-flood_risk"' in html
    assert 'id="fallback-section-flood_risk"' in html
    assert 'id="flood_risk__mode"' in html
    assert 'id="flood_risk__opacity"' in html


def test_config_page_renders_flood_risk_mode_select_with_both_options():
    resp = client.get("/config")
    html = resp.text
    idx = html.index('id="flood_risk__mode"')
    select_html = html[idx : html.index("</select>", idx)]
    assert '<option value="live"' in select_html
    assert '<option value="historical"' in select_html


def test_config_page_renders_glofas_warning_markup_for_the_flood_risk_gate():
    """Locks that the admin page has the key-warn-glofas element the JS in
    config.html looks for (`data.flood_risk.RULE__missing_earthdata_token` ->
    classList.remove('d-none')) -- the gate itself (mode-specific, only Live mode
    needs EARTHDATA_TOKEN) is exercised at the /api/config data layer by
    test_flood_risk_config_gate.py; this just confirms the page has somewhere to
    surface that flag. Element id kept as "key-warn-glofas" (not renamed) even
    though the credential itself changed -- see collectors/flood_risk.py's module
    docstring for Live mode's data-source pivot away from GloFAS."""
    resp = client.get("/config")
    html = resp.text
    assert 'id="key-warn-glofas"' in html
    assert "RULE__missing_earthdata_token" in html
