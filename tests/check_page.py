#!/usr/bin/env python3
"""Small dependency-free structural smoke test for the deployed HTML shell."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "index.html"

REQUIRED_IDS = {
    "sim-canvas", "reward-chart", "launch-btn", "pause-btn", "reset-btn",
    "policy-select", "train-btn", "stop-train-btn", "evaluate-btn",
    "export-btn", "import-file", "progress-bar", "progress-label",
    "progress-percent", "episodes-value", "success-value", "reward-value",
    "quality-value", "tele-vx", "tele-vy", "tele-angle", "tele-omega",
    "tele-thrust", "report-data", "report-empty", "action-notice",
}

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.scripts = []
        self.styles = []
        self.title_count = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.styles.append(attrs.get("href", ""))
        if tag == "title":
            self.title_count += 1

def main():
    html = HTML_PATH.read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(html)
    duplicates = sorted(k for k, v in Counter(parser.ids).items() if v > 1)
    missing = sorted(REQUIRED_IDS - set(parser.ids))
    errors = []
    if duplicates:
        errors.append("duplicate IDs: " + ", ".join(duplicates))
    if missing:
        errors.append("missing required UI IDs: " + ", ".join(missing))
    if parser.title_count != 1 or "<title>Descent Lab" not in html:
        errors.append("expected one Descent Lab document title")
    for asset in ("src/core.js", "src/main.js"):
        if asset not in parser.scripts:
            errors.append(f"missing script asset: {asset}")
        elif not (ROOT / asset).is_file():
            errors.append(f"script asset not found: {asset}")
    if "src/styles.css" not in parser.styles or not (ROOT / "src/styles.css").is_file():
        errors.append("missing stylesheet asset src/styles.css")
    if errors:
        print("Page checks failed:")
        for error in errors:
            print(" -", error)
        return 1
    print(f"Page checks passed: {len(parser.ids)} IDs, {len(parser.scripts)} scripts, {len(parser.styles)} stylesheets.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
