"""Build the documentation site in docs/ from the committed results.

    python make_docs.py

Run it last, after the pipeline. Every number on every page is read from
results/ and data/ at build time (docs_src/numbers.py), so the site cannot drift
from the code the way hand-maintained documents did. GitHub Pages serves docs/
as static files; nothing else is needed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from docs_src import numbers, pages, site

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs"


def main() -> None:
    N = numbers.load()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    (OUT / "assets" / "fig").mkdir(parents=True, exist_ok=True)
    (OUT / "assets" / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "demo").mkdir(parents=True, exist_ok=True)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    (OUT / "assets" / "site.css").write_text(site.CSS, encoding="utf-8")
    for f in sorted((ROOT / "results").glob("fig*.svg")):
        shutil.copy2(f, OUT / "assets" / "fig" / f.name)
    for f in sorted((ROOT / "reports").glob("*.pdf")):
        shutil.copy2(f, OUT / "assets" / "reports" / f.name)
    demo = ROOT / "demo" / "inspector.html"
    if demo.exists():
        shutil.copy2(demo, OUT / "demo" / "inspector.html")

    for slug, fn in pages.PAGES.items():
        html = site.page(slug, fn(N), commit=commit, version=N["version"])
        (OUT / f"{slug}.html").write_text(html, encoding="utf-8")
    print(f"wrote {len(pages.PAGES)} pages to {OUT}")


if __name__ == "__main__":
    main()
