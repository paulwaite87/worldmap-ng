#!/usr/bin/env python3
"""Tests for PrecipitationUpdater's meaningful-floor pipeline. The legend key is
entirely client-side now (issue #302, see ui/modules/precipitation.js's own
PALETTES/lutFor) -- PrecipitationUpdater no longer renders one at all."""
import numpy as np

from atmos_gl.tasks.precipitation import PrecipitationUpdater


def make_floor_updater(sigma_cells=1.2):
    u = PrecipitationUpdater.__new__(PrecipitationUpdater)
    u.MEANINGFUL_PRECIP_MM_HR = 0.1
    u.EDGE_SMOOTH_SIGMA_CELLS = 0.75
    u._smooth_sigma_cells = sigma_cells
    return u


def test_apply_meaningful_floor_zeroes_noise_far_from_any_real_core():
    """Widespread low-level noise (e.g. blur/quantization residue) with no real
    precipitation core anywhere nearby must be zeroed entirely -- a value-only
    threshold (plain clip or smoothstep) can't tell this apart from a real core's own
    falloff, since both live in the same value range; connectivity to an actual
    >=floor core is what distinguishes them."""
    u = make_floor_updater()
    arr = np.full((40, 40), 0.05, dtype=np.float32)  # < floor everywhere, no core at all

    result = u._apply_meaningful_floor(arr)

    assert (result == 0.0).all()


def test_apply_meaningful_floor_keeps_a_smooth_halo_around_a_real_core():
    """A real core (>= floor) keeps its immediate falloff smooth (nonzero, scaled
    down but not hard-clipped) within the blur's own support radius, while noise well
    outside that halo -- even at an identical raw magnitude -- is zeroed."""
    u = make_floor_updater(sigma_cells=1.2)  # radius = round(3*1.2) = 4
    arr = np.zeros((40, 40), dtype=np.float32)
    arr[20, 20] = 5.0  # a real core, well above floor
    arr[20, 22] = 0.05  # 2 cells away -- inside the halo (radius 4)
    arr[20, 35] = 0.05  # far away -- outside the halo, identical magnitude

    result = u._apply_meaningful_floor(arr)

    # A single-pixel spike is an unrealistic stand-in for a real core (which arrives
    # here already spread over many cells by _smooth_global_field's own blur) -- the
    # final edge-softening blur (EDGE_SMOOTH_SIGMA_CELLS) spreads a lone spike's peak
    # a lot more than it would a real, already-blurred core. So assert relative
    # dominance (still clearly the peak, not attenuated to noise level) rather than
    # an exact value.
    assert result[20, 20] > result[20, 22] > 0.0  # core still dominates its halo neighbour
    assert result[20, 35] == 0.0  # outside the halo: zeroed despite equal magnitude


def test_apply_meaningful_floor_returns_all_zero_when_no_core_exists():
    """No core anywhere (e.g. a completely dry field) -- must not error, and must
    zero the whole field rather than leaving stray sub-floor noise."""
    u = make_floor_updater()
    arr = np.zeros((10, 10), dtype=np.float32)

    result = u._apply_meaningful_floor(arr)

    assert (result == 0.0).all()


def make_settings_sig_updater(settings=None):
    u = PrecipitationUpdater.__new__(PrecipitationUpdater)
    u.settings = settings or {}
    return u


def test_render_settings_signature_changes_for_each_render_relevant_setting():
    """Bug: min_mm_hr/opacity/palette are baked directly into the static per-hour PNG
    (plot()'s min_rate/alpha/palette_name), but should_plot_for_hour used to compare
    only the output file's mtime against the DATA's updated_at -- a settings-only edit
    touched neither, so an already-cached hour was never re-rendered with the new
    value. _render_settings_signature (wired into SingleHourScalarUpdater.run() via
    render_all_hours' settings_sig) closes that gap; this locks its reaction to each
    setting it covers."""
    base = {"min_mm_hr": 0.1, "opacity": 50, "palette": "standard"}
    base_sig = make_settings_sig_updater(dict(base))._render_settings_signature()
    for key, changed in (
        ("min_mm_hr", 1.0),
        ("opacity", 80),
        ("palette", "ocean_blue"),
    ):
        variant = dict(base)
        variant[key] = changed
        variant_sig = make_settings_sig_updater(variant)._render_settings_signature()
        assert variant_sig != base_sig, f"{key} change did not alter the signature"


def test_render_settings_signature_stable_for_identical_settings():
    u = make_settings_sig_updater({"min_mm_hr": 0.5, "opacity": 60, "palette": "high_contrast"})
    assert u._render_settings_signature() == u._render_settings_signature()
