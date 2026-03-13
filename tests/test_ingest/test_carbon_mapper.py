"""Tests for Carbon Mapper label ingestion."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.ingest.carbon_mapper import (
    fetch_detections,
    load_detections,
    save_detections,
    _authenticate,
    _parse_plume,
)
from methane_sentinel_labels.models import Detection


@pytest.fixture
def token_response(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "cm_token.json").read_text())


@pytest.fixture
def page1_response(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "cm_plumes_page1.json").read_text())


@pytest.fixture
def page2_response(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "cm_plumes_page2.json").read_text())


@pytest.fixture
def empty_response() -> dict:
    return {"items": [], "total_count": 0, "limit": 1000, "offset": 0}


class TestParsePlume:
    def test_parse_valid_plume(self, page1_response: dict):
        item = page1_response["items"][0]
        detection = _parse_plume(item)
        assert detection.source_id == "tan20240615t143000c22s4001-A"
        assert detection.latitude == 31.872
        assert detection.longitude == -103.245
        assert detection.emission_rate_kg_hr == 150.5
        assert detection.emission_uncertainty_kg_hr == 30.2
        assert detection.sensor == "tan"
        assert detection.provider == "carbon_mapper"
        assert detection.detection_time == datetime(
            2024, 6, 15, 14, 30, tzinfo=timezone.utc
        )

    def test_parse_null_emission_fields(self, page1_response: dict):
        item = page1_response["items"][2]
        detection = _parse_plume(item)
        assert detection.emission_rate_kg_hr is None
        assert detection.emission_uncertainty_kg_hr is None
        assert detection.source_id == "ang20240616t134500c33s5001-A"


class TestAuthenticate:
    def test_auth_returns_token(self, token_response: dict):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = httpx.Response(
            200, json=token_response, request=httpx.Request("POST", "http://test")
        )
        cfg = PipelineConfig(cm_email="test@example.com", cm_password="secret")
        token = _authenticate(mock_client, cfg)
        assert token == token_response["access"]
        mock_client.post.assert_called_once()

    def test_auth_failure_raises(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = httpx.Response(
            401,
            json={"detail": "Invalid credentials"},
            request=httpx.Request("POST", "http://test"),
        )
        cfg = PipelineConfig(cm_email="bad@example.com", cm_password="wrong")
        with pytest.raises(httpx.HTTPStatusError):
            _authenticate(mock_client, cfg)


class TestFetchDetections:
    def test_single_page_no_auth(self, page1_response: dict):
        """Published plumes endpoint works without authentication."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = httpx.Response(
            200, json=page1_response, request=httpx.Request("GET", "http://test")
        )
        cfg = PipelineConfig()
        detections = fetch_detections(cfg, client=mock_client)
        assert len(detections) == 3
        assert all(isinstance(d, Detection) for d in detections)
        mock_client.post.assert_not_called()

    def test_single_page_with_auth(self, page1_response: dict, token_response: dict):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = httpx.Response(
            200, json=token_response, request=httpx.Request("POST", "http://test")
        )
        mock_client.get.return_value = httpx.Response(
            200, json=page1_response, request=httpx.Request("GET", "http://test")
        )
        cfg = PipelineConfig(cm_email="test@example.com", cm_password="secret")
        detections = fetch_detections(cfg, client=mock_client)
        assert len(detections) == 3
        mock_client.post.assert_called_once()

    def test_multi_page_pagination(
        self,
        page1_response: dict,
        page2_response: dict,
        token_response: dict,
        monkeypatch,
    ):
        # Set page size to 3 so page1 (3 items) triggers a second page fetch
        monkeypatch.setattr(
            "methane_sentinel_labels.ingest.carbon_mapper._PAGE_SIZE", 3
        )
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = httpx.Response(
            200, json=token_response, request=httpx.Request("POST", "http://test")
        )
        mock_client.get.side_effect = [
            httpx.Response(
                200, json=page1_response, request=httpx.Request("GET", "http://test")
            ),
            httpx.Response(
                200, json=page2_response, request=httpx.Request("GET", "http://test")
            ),
        ]
        cfg = PipelineConfig(cm_email="test@example.com", cm_password="secret")
        detections = fetch_detections(cfg, client=mock_client)
        assert len(detections) == 4

    def test_empty_response(self, empty_response: dict, token_response: dict):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = httpx.Response(
            200, json=token_response, request=httpx.Request("POST", "http://test")
        )
        mock_client.get.return_value = httpx.Response(
            200, json=empty_response, request=httpx.Request("GET", "http://test")
        )
        cfg = PipelineConfig(cm_email="test@example.com", cm_password="secret")
        detections = fetch_detections(cfg, client=mock_client)
        assert detections == []

    def test_deduplication(
        self, page1_response: dict, token_response: dict, monkeypatch
    ):
        # Set page size to 3 so both pages get fetched
        monkeypatch.setattr(
            "methane_sentinel_labels.ingest.carbon_mapper._PAGE_SIZE", 3
        )
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = httpx.Response(
            200, json=token_response, request=httpx.Request("POST", "http://test")
        )
        empty = {"items": [], "total_count": 0, "limit": 3, "offset": 6}
        # Return same page twice then empty to simulate duplicate items
        mock_client.get.side_effect = [
            httpx.Response(
                200, json=page1_response, request=httpx.Request("GET", "http://test")
            ),
            httpx.Response(
                200, json=page1_response, request=httpx.Request("GET", "http://test")
            ),
            httpx.Response(
                200, json=empty, request=httpx.Request("GET", "http://test")
            ),
        ]
        cfg = PipelineConfig(cm_email="test@example.com", cm_password="secret")
        detections = fetch_detections(cfg, client=mock_client)
        # Should deduplicate by source_id
        assert len(detections) == 3

    def test_limit(self, page1_response: dict, token_response: dict):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = httpx.Response(
            200, json=token_response, request=httpx.Request("POST", "http://test")
        )
        mock_client.get.return_value = httpx.Response(
            200, json=page1_response, request=httpx.Request("GET", "http://test")
        )
        cfg = PipelineConfig(
            cm_email="test@example.com", cm_password="secret", limit=2
        )
        detections = fetch_detections(cfg, client=mock_client)
        assert len(detections) == 2

    def test_limit_passed_to_api(self, page1_response: dict):
        """When limit < PAGE_SIZE, API request should use limit as page size."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = httpx.Response(
            200, json=page1_response, request=httpx.Request("GET", "http://test")
        )
        cfg = PipelineConfig(limit=2)
        fetch_detections(cfg, client=mock_client)
        # Verify the API was called with limit=2, not PAGE_SIZE (1000)
        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["params"]["limit"] == 2

    def test_limit_stops_pagination_early(
        self, page1_response: dict, monkeypatch
    ):
        """Pagination should stop once enough items are fetched, not fetch all pages."""
        monkeypatch.setattr(
            "methane_sentinel_labels.ingest.carbon_mapper._PAGE_SIZE", 3
        )
        mock_client = MagicMock(spec=httpx.Client)
        # page1 has 3 items, should stop after first page since limit=2
        mock_client.get.return_value = httpx.Response(
            200, json=page1_response, request=httpx.Request("GET", "http://test")
        )
        cfg = PipelineConfig(limit=2)
        detections = fetch_detections(cfg, client=mock_client)
        assert len(detections) == 2
        # Should only have made 1 GET request (no second page)
        assert mock_client.get.call_count == 1


class TestSaveLoadDetections:
    def test_roundtrip(self, tmp_output: Path, page1_response: dict):
        detections = [_parse_plume(item) for item in page1_response["items"]]
        parquet_path = tmp_output / "detections.parquet"
        save_detections(detections, parquet_path)
        assert parquet_path.exists()

        loaded = load_detections(parquet_path)
        assert len(loaded) == len(detections)
        for orig, loaded_d in zip(detections, loaded):
            assert orig.source_id == loaded_d.source_id
            assert orig.latitude == loaded_d.latitude
            assert orig.longitude == loaded_d.longitude
            assert orig.sensor == loaded_d.sensor

    def test_roundtrip_preserves_none(self, tmp_output: Path, page1_response: dict):
        detections = [_parse_plume(page1_response["items"][2])]
        parquet_path = tmp_output / "detections.parquet"
        save_detections(detections, parquet_path)
        loaded = load_detections(parquet_path)
        assert loaded[0].emission_rate_kg_hr is None


@pytest.mark.integration
def test_fetch_real_api():
    """Integration test: hits the real Carbon Mapper API (no auth needed for published plumes)."""
    cfg = PipelineConfig(limit=5)
    detections = fetch_detections(cfg)
    assert len(detections) > 0
    assert all(isinstance(d, Detection) for d in detections)
