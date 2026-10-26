#!/usr/bin/env python3
"""Dependency-free structural smoke tests for the deployed app shell."""
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
    id_set = set(parser.ids)
    duplicates = sorted(k for k, v in Counter(parser.ids).items() if v > 1)
    missing = sorted(REQUIRED_IDS - id_set)
    errors = []
    if duplicates:
        errors.append("duplicate IDs: " + ", ".join(duplicates))
    if missing:
        errors.append("missing required UI IDs: " + ", ".join(missing))
    if parser.title_count != 1 or "<title>Descent Lab" not in html:
        errors.append("expected one Descent Lab document title")

    main_js_path = ROOT / "src/main.js"
    core_js_path = ROOT / "src/core.js"
    css_path = ROOT / "src/styles.css"
    for asset, label in ((core_js_path, "src/core.js"), (main_js_path, "src/main.js"), (css_path, "src/styles.css")):
        if not asset.is_file():
            errors.append("asset not found: " + label)
    if "src/styles.css" not in parser.styles:
        errors.append("page must load src/styles.css")
    if "src/core.js" not in parser.scripts or "src/main.js" not in parser.scripts:
        errors.append("page must load src/core.js and src/main.js")
    if parser.scripts.index("src/core.js") > parser.scripts.index("src/main.js") if "src/core.js" in parser.scripts and "src/main.js" in parser.scripts else False:
        errors.append("src/core.js must load before src/main.js")

    if main_js_path.is_file():
        source_refs = set(re.findall(r"\$\(['\"]([^'\"]+)['\"]\)", main_js_path.read_text(encoding="utf-8")))
        missing_refs = sorted(source_refs - id_set)
        if missing_refs:
            errors.append("IDs referenced by src/main.js but missing from page: " + ", ".join(missing_refs))

    if errors:
        print("Page checks failed:")
        for error in errors:
            print(" -", error)
        return 1
    print(f"Page checks passed: {len(parser.ids)} IDs; all app selectors exist; CSS and JS assets are linked.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
