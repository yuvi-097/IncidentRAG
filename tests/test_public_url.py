"""The public demo's link is read from the tunnel's logs (scripts/public_url.py)."""

from __future__ import annotations

from scripts.public_url import latest_link


def test_the_newest_link_wins() -> None:
    logs = (
        "INF Your quick Tunnel has been created! Visit it at:\n"
        "INF |  https://old-words-here.trycloudflare.com  |\n"
        "INF Registered tunnel connection\n"
        "INF |  https://new-words-here.trycloudflare.com  |\n"
    )
    assert latest_link(logs) == "https://new-words-here.trycloudflare.com"


def test_no_link_before_the_tunnel_is_up() -> None:
    assert latest_link("INF Starting tunnel tunnelID=\nERR failed to request quick Tunnel") is None
    assert latest_link("see https://www.cloudflare.com/website-terms/") is None
