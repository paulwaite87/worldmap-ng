#!/usr/bin/env python3
"""EARTHDATA_TOKEN gate for the flood_risk layer -- unlike greenhouse_gases/
air_quality's single-key-disables-whole-section shape, this gate is MODE-SPECIFIC:
only Live mode (NASA LANCE MODIS observed flooding) needs a credential; Historical
mode (JRC hazard maps, static/no-auth) needs none, so it must stay enabled/available
even without EARTHDATA_TOKEN configured. See issue #371 and its follow-up grilling
(collectors/flood_risk.py's module docstring), routes/config.py::
_build_config_data()/update_config().
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from atmos_gl.api import app
from atmos_gl.lib.auth import SESSION_COOKIE_NAME
from atmos_gl.lib.config import AtmosGLConfig
from atmos_gl.db.user_settings_adapter import FakeUserSettingsAdapter
from atmos_gl.routes.auth import get_user_adapter
from atmos_gl.routes.config import get_user_settings_adapter
from tests.conftest import make_signed_in_session

client = TestClient(app)


@pytest.fixture(autouse=True)
def _admin_session():
    fake, token = make_signed_in_session(is_admin=True)
    app.dependency_overrides[get_user_adapter] = lambda: fake
    app.dependency_overrides[get_user_settings_adapter] = lambda: FakeUserSettingsAdapter()
    client.cookies.set(SESSION_COOKIE_NAME, token)


def _with_temp_config(tmp_path, initial: dict):
    tmp_config = tmp_path / "atmos-gl.json"
    tmp_config.write_text(json.dumps(initial))
    return patch(
        "atmos_gl.routes.config.load_config",
        return_value=AtmosGLConfig(str(tmp_config)),
    ), tmp_config


def test_get_config_disables_live_mode_when_token_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    patcher, _ = _with_temp_config(
        tmp_path, {"flood_risk": {"enabled": True, "mode": "live"}}
    )
    with patcher:
        resp = client.get("/api/config")

    data = resp.json()["data"]["flood_risk"]
    assert data["RULE__missing_earthdata_token"] is True
    assert data["enabled"] is False


def test_get_config_does_not_flag_or_disable_when_token_present(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "some-token")
    patcher, _ = _with_temp_config(
        tmp_path, {"flood_risk": {"enabled": True, "mode": "live"}}
    )
    with patcher:
        resp = client.get("/api/config")

    data = resp.json()["data"]["flood_risk"]
    assert "RULE__missing_earthdata_token" not in data
    assert data["enabled"] is True


def test_get_config_historical_mode_stays_enabled_without_a_token(tmp_path, monkeypatch):
    """Historical mode (JRC hazard maps) needs no credential at all -- must not be
    disabled just because EARTHDATA_TOKEN (a Live-mode-only requirement) is unset."""
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    patcher, _ = _with_temp_config(
        tmp_path, {"flood_risk": {"enabled": True, "mode": "historical"}}
    )
    with patcher:
        resp = client.get("/api/config")

    data = resp.json()["data"]["flood_risk"]
    assert "RULE__missing_earthdata_token" not in data
    assert data["enabled"] is True


def test_update_config_strips_missing_earthdata_token_rule_before_saving(tmp_path):
    patcher, tmp_config = _with_temp_config(
        tmp_path, {"flood_risk": {"enabled": True, "mode": "live"}}
    )
    with patcher:
        resp = client.post(
            "/api/config",
            json={
                "flood_risk": {
                    "enabled": True,
                    "mode": "live",
                    "RULE__missing_earthdata_token": True,
                }
            },
        )

    assert resp.status_code == 200
    saved = json.loads(tmp_config.read_text())
    assert "RULE__missing_earthdata_token" not in saved["flood_risk"]
