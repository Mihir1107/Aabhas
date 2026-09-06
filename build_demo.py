"""Splice the exported real pipeline data into the inspector shell.

Produces demo/inspector.html, a single self-contained file that runs offline
with no server and no build step. Regenerate with:

    python make_demo_data.py && python build_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

TPL = Path("demo/_template.html")
DATA = Path("demo/inspector_data.json")
OUT = Path("demo/inspector.html")

def main() -> None:
    tpl = TPL.read_text(encoding="utf-8")
    raw = DATA.read_text(encoding="utf-8")
    json.loads(raw)                      # refuse to ship malformed data
    # The payload sits in a <script type="application/json"> block, so the only
    # sequence that can break out of it is a literal "</script". Escaping the
    # slash is invisible to JSON.parse and closes that hole.
    safe = raw.replace("</", "<" + chr(92) + "/")
    assert "__DATA__" in tpl, "template lost its __DATA__ placeholder"
    OUT.write_text(tpl.replace("__DATA__", safe), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
