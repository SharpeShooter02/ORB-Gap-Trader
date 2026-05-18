"""
scripts/backup_state_store.py — Nightly backup to B2 / R2 / S3-compatible storage.

Reads credentials from environment:
    BACKUP_BUCKET_URL   — e.g. s3://my-bucket/orb-live/ or b2://bucket/prefix/
    BACKUP_ACCESS_KEY   — S3/B2 access key (or AWS_ACCESS_KEY_ID)
    BACKUP_SECRET_KEY   — S3/B2 secret key (or AWS_SECRET_ACCESS_KEY)

Uploads:
  1. state_store SQLite via sqlite3 .backup (WAL-safe consistent snapshot)
  2. Last 7 days of data/archive/**/*.parquet (incremental)
  3. Last 7 days of data/logs/*.log files

Usage:
    python -m orb_live.scripts.backup_state_store
    python -m orb_live.scripts.backup_state_store --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backup state_store and archives")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be uploaded without uploading")
    p.add_argument("--db-path", default=None,
                   help="Override path to state_store SQLite file")
    p.add_argument("--data-dir", default=None,
                   help="Override data/ directory")
    return p.parse_args()


def _backup_sqlite(db_path: Path, target_path: Path) -> None:
    """Create a consistent SQLite backup using the sqlite3 .backup API."""
    src  = sqlite3.connect(str(db_path))
    dest = sqlite3.connect(str(target_path))
    try:
        src.backup(dest)
    finally:
        dest.close()
        src.close()


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_uploads(db_path: Path, data_dir: Path) -> list[tuple[Path, str]]:
    """
    Returns list of (local_path, remote_key) tuples to upload.
    remote_key is relative to the bucket prefix.
    """
    uploads: list[tuple[Path, str]] = []
    today    = date.today()
    cutoff   = today - timedelta(days=7)

    # 1. state_store DB snapshot (created in temp dir at upload time)
    uploads.append((db_path, f"state_store/live_{today}.db"))

    # 2. Archive parquets from last 7 days (by file modification time)
    archive_dir = data_dir / "archive"
    if archive_dir.exists():
        for pq_file in sorted(archive_dir.rglob("*.parquet")):
            from datetime import datetime
            mtime = datetime.fromtimestamp(pq_file.stat().st_mtime).date()
            if mtime >= cutoff:
                rel   = pq_file.relative_to(data_dir)
                uploads.append((pq_file, f"archive/{rel}"))

    # 3. Log files from last 7 days
    log_dir = data_dir / "logs"
    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.log")):
            from datetime import datetime
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime).date()
            if mtime >= cutoff:
                uploads.append((log_file, f"logs/{log_file.name}"))

    return uploads


def _upload_with_boto3(
    uploads: list[tuple[Path, str]],
    bucket_url: str,
    access_key: str,
    secret_key: str,
    dry_run: bool = False,
) -> int:
    """Upload files using boto3.  Returns number of files uploaded."""
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore

    # Parse bucket URL: s3://bucket/prefix or b2://bucket/prefix
    url = bucket_url.rstrip("/")
    scheme = url.split("://")[0]
    rest   = url.split("://", 1)[1]
    bucket = rest.split("/")[0]
    prefix = "/".join(rest.split("/")[1:])
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    endpoint_map = {
        "b2": "https://s3.us-west-004.backblazeb2.com",
        "r2": os.getenv("R2_ENDPOINT_URL", ""),
    }
    endpoint = endpoint_map.get(scheme) if scheme in endpoint_map else None

    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )

    uploaded = 0
    for local_path, remote_key in uploads:
        full_key = prefix + remote_key
        if dry_run:
            print(f"  [dry-run] {local_path} → s3://{bucket}/{full_key}")
            uploaded += 1
            continue
        try:
            s3.upload_file(str(local_path), bucket, full_key)
            print(f"  uploaded: {local_path.name} → {full_key}")
            uploaded += 1
        except Exception as exc:
            print(f"  ERROR uploading {local_path}: {exc}", file=sys.stderr)

    return uploaded


def main() -> None:
    args = _parse_args()

    import orb_live  # noqa — sys.path setup
    from orb_live.config.live_config import load_live_config
    cfg = load_live_config()

    db_path  = Path(args.db_path)  if args.db_path  else cfg.db_path
    data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir.parent

    bucket_url = os.getenv("BACKUP_BUCKET_URL", "")
    access_key = os.getenv("BACKUP_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID",  ""))
    secret_key = os.getenv("BACKUP_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", ""))

    if not bucket_url and not args.dry_run:
        print("ERROR: BACKUP_BUCKET_URL not set.  Use --dry-run to preview.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Collecting uploads from {db_path} and {data_dir}/...")
    uploads = _collect_uploads(db_path, data_dir)
    print(f"  {len(uploads)} files to upload")

    if not bucket_url:
        # dry-run without bucket URL: just list files
        for local_path, remote_key in uploads:
            size = local_path.stat().st_size if local_path.exists() else 0
            print(f"  [would upload] {local_path}  ({size/1024:.1f} KB) → {remote_key}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        if db_path.exists():
            snap = Path(tmp) / "live_snapshot.db"
            _backup_sqlite(db_path, snap)
            # Replace db_path entry with the snapshot
            uploads = [
                (snap if lp == db_path else lp, rk)
                for lp, rk in uploads
            ]

        n = _upload_with_boto3(uploads, bucket_url, access_key, secret_key,
                               dry_run=args.dry_run)
    print(f"Backup complete: {n} files.")


if __name__ == "__main__":
    main()
