"""Tests for pipeline configuration."""

from pathlib import Path

from methane_sentinel_labels.config import PipelineConfig


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.max_time_delta_hours == 120.0
        assert cfg.max_cloud_cover_pct == 30.0
        assert cfg.max_matches_per_detection == 3
        assert cfg.patch_half_size_m == 2560.0
        assert cfg.target_resolution_m == 20.0
        assert cfg.min_cloud_free_fraction == 0.5
        assert cfg.output_dir == Path("output")
        assert "B11" in cfg.bands
        assert "B12" in cfg.bands
        assert "SCL" in cfg.bands

    def test_custom_values(self):
        cfg = PipelineConfig(
            max_time_delta_hours=48.0,
            max_cloud_cover_pct=20.0,
            output_dir=Path("/tmp/test"),
            limit=10,
        )
        assert cfg.max_time_delta_hours == 48.0
        assert cfg.max_cloud_cover_pct == 20.0
        assert cfg.output_dir == Path("/tmp/test")
        assert cfg.limit == 10

    def test_env_var_credentials(self, monkeypatch):
        monkeypatch.setenv("CM_EMAIL", "test@example.com")
        monkeypatch.setenv("CM_PASSWORD", "secret123")
        cfg = PipelineConfig()
        assert cfg.cm_email == "test@example.com"
        assert cfg.cm_password == "secret123"

    def test_missing_credentials_default_empty(self, monkeypatch):
        monkeypatch.delenv("CM_EMAIL", raising=False)
        monkeypatch.delenv("CM_PASSWORD", raising=False)
        cfg = PipelineConfig()
        assert cfg.cm_email == ""
        assert cfg.cm_password == ""

    def test_frozen(self):
        cfg = PipelineConfig()
        try:
            cfg.max_time_delta_hours = 999.0  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_stac_defaults(self):
        cfg = PipelineConfig()
        assert "earth-search" in cfg.stac_url
        assert cfg.stac_collection == "sentinel-2-l2a"
