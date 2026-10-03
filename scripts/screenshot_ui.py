"""Take screenshots of every page of the running Streamlit frontend (for the README).

    python scripts/screenshot_ui.py [--url http://127.0.0.1:8501] [--out docs/screenshots]

Needs the API and the frontend running, and Playwright's Chromium
(``pip install playwright && python -m playwright install chromium``). The README's
screenshots were taken as the admin demo user (``OPSRAG_DEMO_USER=noor.hassan`` when
starting Streamlit), so that no hop of the traced change is withheld. It drives the
UI like a user: asks a question in Chat, then opens the other pages from the sidebar
(so the session, and its answers, carry over), and saves one full-page PNG per page.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Which commit and file caused INC-0246?"
TRACE_SEARCH = "payment requests returning HTTP 500 after v2.6.16 deploy"
VIEWPORT = {"width": 1440, "height": 900}


# What each page draws last. With the minimal toolbar Streamlit shows no "running"
# indicator to wait on, so a page counts as drawn once its last element is there.
READY = {
    "Chat": "text=Try one of these",
    "Evidence Viewer": '[data-testid="stExpander"]',
    "Incident Explorer": "text=Select a row to open the incident",
    "Evaluation": "text=How these numbers were produced",
    "System Metrics": "text=Errors and outcomes",
}


def settle(page: Page, ready: str, timeout: float = 120_000) -> None:
    """Wait until the page's last element is drawn, then for its charts."""
    page.wait_for_selector(ready, timeout=timeout)
    page.wait_for_timeout(2500)


def ask(page: Page, question: str, attempts: int = 3) -> None:
    """Submit a question in Chat and wait for the answer's tabs.

    A submit sent while the page is still wiring up its widgets can be dropped, so it is
    retried until the question shows up in the conversation.
    """
    box = page.get_by_placeholder("Ask about incidents, deployments, code, logs or runbooks…")
    for attempt in range(attempts):
        box.fill(question)
        box.press("Enter")
        try:
            page.wait_for_selector('[data-testid="stChatMessage"]', timeout=15_000)
            break
        except PlaywrightTimeout:
            if attempt == attempts - 1:
                raise
    tab = page.get_by_role("tab", name=re.compile(r"^Recommendations"))
    tab.wait_for(timeout=300_000)


def trace_incident(page: Page, search: str) -> None:
    """In the Explorer: search, open the first result and trace it to its change."""
    page.get_by_placeholder("e.g. connection pool timeouts after deploy").fill(search)
    page.get_by_test_id("stBaseButton-primaryFormSubmit").click()  # not the table's search
    page.wait_for_selector("text=ranked by relevance", timeout=60_000)
    page.wait_for_timeout(1500)
    # The table is drawn on a canvas: click the first row's selection box.
    grid = page.locator('[data-testid="stDataFrame"]').first.bounding_box()
    assert grid is not None
    page.mouse.click(grid["x"] + 16, grid["y"] + 35 + 17)
    page.get_by_role("button", name="Trace the change").click(timeout=60_000)
    page.wait_for_selector("text=From the incident to the change", timeout=60_000)
    page.wait_for_timeout(1500)


def open_page(page: Page, title: str) -> None:
    page.get_by_role("link", name=title).click()
    settle(page, READY[title])


# The tallest scrolling area: Streamlit scrolls inside a container, not the document.
CONTENT_HEIGHT = """() => Math.max(
    document.body.scrollHeight,
    ...[...document.querySelectorAll('*')]
        .filter(e => ['auto', 'scroll'].includes(getComputedStyle(e).overflowY))
        .map(e => e.scrollHeight))"""


def shoot(page: Page, out: Path, name: str) -> None:
    """Save the whole page.

    Streamlit scrolls inside its main section, not the document, so a "full page"
    screenshot would stop at the window: grow the window to the content first.
    """
    size = page.viewport_size or VIEWPORT
    page.set_viewport_size(
        {"width": size["width"], "height": max(size["height"], page.evaluate(CONTENT_HEIGHT))}
    )
    page.wait_for_timeout(1500)  # charts re-layout to the new size
    path = out / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size(VIEWPORT)
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument(
        "--trace-search",
        default=TRACE_SEARCH,
        help="Explorer search whose first result is opened and traced",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.goto(args.url)
        settle(page, READY["Chat"])
        shoot(page, args.out, "chat-empty")
        ask(page, args.question)
        page.wait_for_timeout(2500)
        shoot(page, args.out, "chat")
        page.get_by_role("tab", name="Recommendations").click()
        page.wait_for_timeout(800)
        shoot(page, args.out, "chat-recommendations")
        open_page(page, "Evidence Viewer")
        shoot(page, args.out, "evidence")
        open_page(page, "Incident Explorer")
        shoot(page, args.out, "incidents")
        trace_incident(page, args.trace_search)
        shoot(page, args.out, "incident-trace")
        open_page(page, "Evaluation")
        shoot(page, args.out, "evaluation")
        open_page(page, "System Metrics")
        shoot(page, args.out, "metrics")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
