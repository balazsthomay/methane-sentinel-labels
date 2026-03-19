"""Tests for the CLI."""

from unittest.mock import patch

import pytest

from methane_sentinel_labels.cli import main


class TestCLI:
    def test_no_command_exits_cleanly(self):
        """No subcommand shows help and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_help_flag(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_ingest_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["ingest", "--help"])
        assert exc_info.value.code == 0

    def test_run_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--help"])
        assert exc_info.value.code == 0

    @patch("methane_sentinel_labels.cli.fetch_detections")
    @patch("methane_sentinel_labels.cli.save_detections")
    def test_ingest_calls_fetch(self, mock_save, mock_fetch, tmp_path):
        mock_fetch.return_value = []
        main(["ingest", "--output-dir", str(tmp_path)])
        mock_fetch.assert_called_once()
        mock_save.assert_called_once()

    @patch("methane_sentinel_labels.cli.fetch_detections")
    @patch("methane_sentinel_labels.cli.save_detections")
    @patch("methane_sentinel_labels.cli.find_matches")
    @patch("methane_sentinel_labels.cli.extract_patches")
    @patch("methane_sentinel_labels.cli.assemble_dataset")
    def test_run_empty_detections(
        self, mock_assemble, mock_extract, mock_matches, mock_save, mock_fetch, tmp_path
    ):
        mock_fetch.return_value = []
        main(["run", "--output-dir", str(tmp_path)])
        mock_fetch.assert_called_once()
        mock_matches.assert_not_called()  # Should stop early

    def test_msat_ingest_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["msat-ingest", "--help"])
        assert exc_info.value.code == 0

    def test_msat_run_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["msat-run", "--help"])
        assert exc_info.value.code == 0

    @patch("methane_sentinel_labels.cli.ingest_methanesat")
    def test_msat_ingest_calls_ingest(self, mock_ingest, tmp_path):
        mock_ingest.return_value = ([], [])
        main(["msat-ingest", "--output-dir", str(tmp_path)])
        mock_ingest.assert_called_once()

    @patch("methane_sentinel_labels.cli.ingest_methanesat")
    @patch("methane_sentinel_labels.cli.find_sentinel2_matches")
    def test_msat_run_empty_masks(self, mock_matches, mock_ingest, tmp_path):
        mock_ingest.return_value = ([], [])
        main(["msat-run", "--output-dir", str(tmp_path)])
        mock_ingest.assert_called_once()
        mock_matches.assert_not_called()  # Should stop early
