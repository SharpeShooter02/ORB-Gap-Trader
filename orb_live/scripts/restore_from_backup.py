"""
scripts/restore_from_backup.py — Disaster-recovery: pull a backup from S3/B2/R2
and restore the state_store SQLite.

Usage:
    python -m orb_live.scripts.restore_from_backup --list
    python -m orb_live.scripts.restore_from_backup --date 2025-04-01
    python -m orb_live.scripts.restore_from_backup --date 2025-04-01 --dry-run

Environment variables (same as backup_state_store):
    BACKUP_BUCKET_URL   — e.g. s3://my-bucket/orb-live/
    BACKUP_ACCESS_KEY   — access key
    BACKUP_SECRET_KEY   — secret key

IMPORTANT: stop the live session process before running this script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore state_store from B2/S3 backup")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list",   action="store_true",
                   help="List available backup dates")
    g.add_argument("--date",   metavar="YYYY-MM-DD",
                   help="Restore from this date's snapshot")
    p.add_argument("--dry-run", action="store_true",
                   help="Download and verify but do not overwrite local DB")
    p.add_argument("--db-path", default=None,
                   help="Override path to local state_store SQLite file")
    return p.parse_args()


def _build_s3_client(access_key: str, secret_key: str, bucket_url: str):
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore

    url    = bucket_url.rstrip("/")
    scheme = url.split("://")[0]
    endpoint_map = {
        "b2": "https://s3.us-west-004.backblazeb2.com",
        "r2": os.getenv("R2_ENDPOINT_URL", ""),
    }
    endpoint = endpoint_map.get(scheme) if scheme in endpoint_map else None

    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )


def _parse_bucket_prefix(bucket_url: str) -> tuple[str, str]:
    url    = bucket_url.rstrip("/")
    rest   = url.split("://", 1)[1]
    bucket = rest.split("/")[0]
    prefix = "/".join(rest.split("/")[1:])
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _list_backups(s3, bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "state_store/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return sorted(keys)


def _verify_sqlite(path: Path) -> bool:
    """Return True if the file is a valid, readable SQLite database."""
    try:
        con = sqlite3.connect(str(path))
        con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        con.close()
        return True
    except Exception:
        return False


def cmd_list(s3, bucket: str, prefix: str) -> None:
    keys = _list_backups(s3, bucket, prefix)
    if not keys:
        print("No backups found.")
        return
    print(f"Available backups ({len(keys)}):")
    for k in keys:
        print(f"  {k}")


def cmd_restore(
    s3,
    bucket: str,
    prefix: str,
    restore_date: str,
    db_path: Path,
    dry_run: bool,
) -> None:
    remote_key = f"{prefix}state_store/live_{restore_date}.db"
    print(f"Downloading s3://{bucket}/{remote_key} ...")

    with tempfile.TemporaryDirectory() as tmp:
        local_snap = Path(tmp) / "restored.db"
        try:
            s3.download_file(bucket, remote_key, str(local_snap))
        except Exception as exc:
            print(f"ERROR: download failed — {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"  Downloaded: {local_snap.stat().st_size / 1024:.1f} KB")

        if not _verify_sqlite(local_snap):
            print("ERROR: downloaded file is not a valid SQLite database.",
                  file=sys.stderr)
            sys.exit(1)
        print("  SQLite integrity: OK")

        if dry_run:
            print("[dry-run] Would restore to:", db_path)
            return

        # Back up existing DB before overwriting
        if db_path.exists():
            bak = db_path.with_suffix(".db.pre_restore")
            shutil.copy2(db_path, bak)
            print(f"  Existing DB backed up to: {bak}")

        shutil.copy2(local_snap, db_path)
        print(f"  Restored to: {db_path}")

    print("Restore complete.")


def main() -> None:
    args = _parse_args()

    import orb_live  # noqa — sys.path setup
    from orb_live.config.live_config import load_live_config
    cfg = load_live_config()

    db_path    = Path(args.db_path) if args.db_path else cfg.db_path
    bucket_url = os.getenv("BACKUP_BUCKET_URL", "")
    access_key = os.getenv("BACKUP_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID",  ""))
    secret_key = os.getenv("BACKUP_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", ""))

    if not bucket_url:
        print("ERROR: BACKUP_BUCKET_URL not set.", file=sys.stderr)
        sys.exit(1)
    if not access_key or not secret_key:
        print("ERROR: BACKUP_ACCESS_KEY / BACKUP_SECRET_KEY not set.", file=sys.stderr)
        sys.exit(1)

    s3             = _build_s3_client(access_key, secret_key, bucket_url)
    bucket, prefix = _parse_bucket_prefix(bucket_url)

    if args.list:
        cmd_list(s3, bucket, prefix)
    else:
        cmd_restore(s3, bucket, prefix, args.date, db_path, args.dry_run)


if __name__ == "__main__":
    main()
