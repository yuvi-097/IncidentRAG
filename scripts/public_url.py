"""Print the public link of the demo started with docker-compose.public.yml.

    python scripts/public_url.py              # waits up to 60 s for the tunnel's link
    python scripts/public_url.py --wait 0     # print it if there is one, otherwise fail

The Cloudflare quick tunnel gets a new random https://….trycloudflare.com link every time
it starts (also when Docker restarts it), so run this after each start.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.public.yml"]
LINK = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def latest_link(logs: str) -> str | None:
    """The newest tunnel link in the logs (earlier runs of the container leave older ones)."""
    found = LINK.findall(logs)
    return found[-1] if found else None


def tunnel_logs() -> str:
    result = subprocess.run(
        [*COMPOSE, "logs", "--no-log-prefix", "tunnel"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr  # cloudflared logs to stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for a link")
    args = parser.parse_args(argv)
    deadline = time.monotonic() + args.wait
    while True:
        link = latest_link(tunnel_logs())
        if link:
            print(link)
            return 0
        if time.monotonic() >= deadline:
            print(
                "No tunnel link yet. Is the stack running with docker-compose.public.yml? "
                "Check: docker compose -f docker-compose.yml -f docker-compose.public.yml ps",
                file=sys.stderr,
            )
            return 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
