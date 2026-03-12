"""CLI entry point for methane-sentinel-labels."""

import argparse
import logging
import sys
from pathlib import Path

from methane_sentinel_labels.assembly.dataset import assemble_dataset
from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.extraction.patches import extract_patches
from methane_sentinel_labels.ingest.carbon_mapper import (
    fetch_detections,
    load_detections,
    save_detections,
)
from methane_sentinel_labels.matching.sentinel2 import find_matches

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="methane-sentinel-labels",
        description="Cross-sensor methane plume label dataset generator",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    common.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of detections to process",
    )
    common.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # ingest subcommand
    subparsers.add_parser("ingest", parents=[common], help="Fetch detections from Carbon Mapper")

    # match subcommand
    match_parser = subparsers.add_parser("match", parents=[common], help="Match detections to Sentinel-2 scenes")
    match_parser.add_argument("--max-time-delta", type=float, default=120.0, help="Max time delta in hours")
    match_parser.add_argument("--max-cloud-cover", type=float, default=30.0, help="Max cloud cover percent")

    # extract subcommand
    extract_parser = subparsers.add_parser("extract", parents=[common], help="Extract patches from matched scenes")
    extract_parser.add_argument("--patch-size", type=float, default=2560.0, help="Patch half-size in meters")
    extract_parser.add_argument("--min-cloud-free", type=float, default=0.5, help="Min cloud-free fraction")

    # run subcommand (full pipeline)
    run_parser = subparsers.add_parser("run", parents=[common], help="Run full pipeline")
    run_parser.add_argument("--max-time-delta", type=float, default=120.0, help="Max time delta in hours")
    run_parser.add_argument("--max-cloud-cover", type=float, default=30.0, help="Max cloud cover percent")
    run_parser.add_argument("--patch-size", type=float, default=2560.0, help="Patch half-size in meters")
    run_parser.add_argument("--min-cloud-free", type=float, default=0.5, help="Min cloud-free fraction")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "match":
        _cmd_match(args)
    elif args.command == "extract":
        _cmd_extract(args)
    elif args.command == "run":
        _cmd_run(args)


def _cmd_ingest(args: argparse.Namespace) -> None:
    cfg = PipelineConfig(output_dir=args.output_dir, limit=args.limit)
    detections = fetch_detections(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_detections(detections, args.output_dir / "detections.parquet")
    logger.info("Ingested %d detections", len(detections))


def _cmd_match(args: argparse.Namespace) -> None:
    cfg = PipelineConfig(
        output_dir=args.output_dir,
        limit=args.limit,
        max_time_delta_hours=args.max_time_delta,
        max_cloud_cover_pct=args.max_cloud_cover,
    )
    detections_path = args.output_dir / "detections.parquet"
    if not detections_path.exists():
        logger.error("No detections found. Run 'ingest' first.")
        sys.exit(1)
    detections = load_detections(detections_path)
    matches = find_matches(detections, cfg)
    logger.info("Found %d matches", len(matches))


def _cmd_extract(args: argparse.Namespace) -> None:
    logger.info("Extract requires running the full pipeline. Use 'run' instead.")
    sys.exit(1)


def _cmd_run(args: argparse.Namespace) -> None:
    cfg = PipelineConfig(
        output_dir=args.output_dir,
        limit=args.limit,
        max_time_delta_hours=args.max_time_delta,
        max_cloud_cover_pct=args.max_cloud_cover,
        patch_half_size_m=args.patch_size,
        min_cloud_free_fraction=args.min_cloud_free,
    )

    # Step A: Ingest
    logger.info("=== Step A: Ingesting detections ===")
    detections = fetch_detections(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_detections(detections, cfg.output_dir / "detections.parquet")

    if not detections:
        logger.warning("No detections found. Stopping.")
        return

    # Step B: Match
    logger.info("=== Step B: Matching to Sentinel-2 scenes ===")
    matches = find_matches(detections, cfg)

    if not matches:
        logger.warning("No matches found. Stopping.")
        return

    # Step C: Extract patches
    logger.info("=== Step C: Extracting patches ===")
    all_records = []
    for detection in detections:
        det_matches = [m for m in matches if m.detection_source_id == detection.source_id]
        if not det_matches:
            continue
        records = extract_patches(
            det_matches,
            cfg,
            latitude=detection.latitude,
            longitude=detection.longitude,
        )
        # Fill in emission rate from the detection
        for r in records:
            filled = type(r)(
                detection_source_id=r.detection_source_id,
                scene_id=r.scene_id,
                patch_path=r.patch_path,
                latitude=r.latitude,
                longitude=r.longitude,
                emission_rate_kg_hr=detection.emission_rate_kg_hr,
                time_delta_hours=r.time_delta_hours,
                cloud_free_fraction=r.cloud_free_fraction,
                crs=r.crs,
                bbox=r.bbox,
            )
            all_records.append(filled)

    # Step D: Assemble dataset
    logger.info("=== Step D: Assembling dataset ===")
    assemble_dataset(all_records, cfg.output_dir)

    logger.info(
        "Pipeline complete: %d detections → %d patches",
        len(detections),
        len(all_records),
    )


if __name__ == "__main__":
    main()
