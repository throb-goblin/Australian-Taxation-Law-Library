from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


JURISDICTIONS = [
    "Australian Capital Territory",
    "Commonwealth",
    "New South Wales",
    "Northern Territory",
    "Queensland",
    "South Australia",
    "Tasmania",
    "Victoria",
    "Western Australia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all jurisdiction sync bots sequentially",
    )
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--backoff-base-seconds", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/rewrite even when version_id unchanged",
    )
    return parser.parse_args()


def build_passthrough_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []

    if args.user_agent is not None:
        out += ["--user-agent", args.user_agent]
    if args.sleep_seconds is not None:
        out += ["--sleep-seconds", str(args.sleep_seconds)]
    if args.timeout_seconds is not None:
        out += ["--timeout-seconds", str(args.timeout_seconds)]
    if args.max_retries is not None:
        out += ["--max-retries", str(args.max_retries)]
    if args.backoff_base_seconds is not None:
        out += ["--backoff-base-seconds", str(args.backoff_base_seconds)]
    if args.limit is not None:
        out += ["--limit", str(args.limit)]

    if args.force:
        out += ["--force"]

    return out


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    passthrough = build_passthrough_args(args)

    failures: list[tuple[str, int]] = []

    for jurisdiction in JURISDICTIONS:
        jurisdiction_dir = repo_root / jurisdiction
        bot_pkg = jurisdiction_dir / "bot" / "sync.py"
        if not bot_pkg.exists():
            print(f"[SKIP] {jurisdiction}: bot not found at {bot_pkg}")
            continue

        cmd = [sys.executable, "-m", "bot.sync", *passthrough]
        print(f"\n=== {jurisdiction} ===")
        completed = subprocess.run(cmd, cwd=str(jurisdiction_dir))
        if completed.returncode != 0:
            print(f"[FAIL] {jurisdiction} returned {completed.returncode}")
            failures.append((jurisdiction, completed.returncode))
            continue

    print("\nAll jurisdiction bots completed.")
    if failures:
        print("\nFailures:")
        for name, code in failures:
            print(f"- {name}: {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
