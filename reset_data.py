"""Reset generated RoadWatch test data without touching source uploads or models."""
import argparse
import os
from pathlib import Path

import config
import database


def _clear_generated_files(directory):
    root = Path(directory).resolve()
    expected_parent = Path(config.BASE_DIR).resolve() / "data"
    if root.parent != expected_parent or root.name not in {"snapshots", "outputs"}:
        raise RuntimeError(f"Refusing to clear unexpected directory: {root}")
    removed = 0
    for item in root.rglob("*"):
        if item.is_file():
            item.unlink()
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="confirm deletion of detections, violation events, snapshots, and generated outputs")
    args = parser.parse_args()
    if not args.yes:
        answer = input("Clear RoadWatch event data, snapshots, and generated outputs? Source uploads and models are preserved. [y/N] ")
        if answer.strip().lower() != "y":
            print("Reset cancelled.")
            return
    database.init_db()
    database.clear_all()
    snapshots = _clear_generated_files(config.SNAPSHOT_DIR)
    outputs = _clear_generated_files(config.OUTPUT_DIR)
    print(f"Reset complete. Removed {snapshots} snapshot/plate images and {outputs} generated output files.")
    print("Source uploads, model files, configuration, application files, and database schema were preserved.")


if __name__ == "__main__":
    main()
