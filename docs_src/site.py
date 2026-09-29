"""Page shell, stylesheet and small HTML helpers for the documentation site."""
from __future__ import annotations

import html
import math

CHAPTERS = [
    # (slug, short title, full title) -- the pipeline really is a sequence,
    # so the rail numbers the chapters that follow the data through it
    ("index", "Overview", "Burn-in screening that sees what datasheet limits cannot"),
    ("problem", "The problem", "Why a part that passes every limit can still fail"),
    ("dataset", "The dataset", "A synthetic burn-in dataset with hidden ground truth"),
    ("module-a", "Module A", "Module A: screening each part against its own lot"),
    ("module-b", "Module B", "Module B: forecasting the 168 h value from the first 24 h"),
    ("guarantee", "Escape-rate bound", "A bounded escape rate: conformal risk control"),
    ("decisions", "Decision layer", "Six decision tiers, and the rules behind them"),
    ("explainability", "Explainability", "Explanations an inspector can check by hand"),
    ("demo", "Inspector demo", "The QA inspector's screen"),
    ("method", "Evaluation & reproduction", "How it was evaluated, and how to reproduce it"),
    ("limitations", "Limitations & audit", "What is not solved, and what the final audit changed"),
    ("reference", "Reference", "Repository map, glossary and references"),
]
NUMBERED = {s: i for i, (s, _, _) in enumerate(CHAPTERS[1:10], start=1)}

CSS = r"""
:root{
  --bg:#F3F5F6; --panel:#FFFFFF; --ink:#18212B; --ink2:#46525E; --mute:#6B7785;
  --rule:#D5DBE0; --mask:#1E5B42; --mask-soft:#DCEBE3; --heat:#B8741A;
  --heat-soft:#F6E7CF; --fault:#A8321F; --fault-soft:#F4DDD8; --steel:#5A6B7B;
  --plate:#FFFFFF; --focus:#1E5B42;
  --t-pass:#2A6A41; --t-watch:#1F5C86; --t-review:#8A5C05; --t-reject:#A32E19;
  --t-fixture:#5C3E8E; --t-invalid:#5B6668; --empty:#E3E7EA;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --sans:"Archivo","Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12171C; --panel:#1A2128; --ink:#E6EBEF; --ink2:#B7C1CA; --mute:#8D99A5;
  --rule:#2C3640; --mask:#6FC39A; --mask-soft:#1D3329; --heat:#E0A24E;
  --heat-soft:#3A2C17; --fault:#E47C68; --fault-soft:#3B211C; --steel:#9AA8B5;
  --plate:#F7F8F9; --focus:#6FC39A; --empty:#26303A;
}}
:root[data-theme="dark"]{
  --bg:#12171C; --panel:#1A2128; --ink:#E6EBEF; --ink2:#B7C1CA; --mute:#8D99A5;
  --rule:#2C3640; --mask:#6FC39A; --mask-soft:#1D3329; --heat:#E0A24E;
  --heat-soft:#3A2C17; --fault:#E47C68; --fault-soft:#3B211C; --steel:#9AA8B5;
  --plate:#F7F8F9; --focus:#6FC39A; --empty:#26303A;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--serif);
  font-size:18px;line-height:1.62;font-optical-sizing:auto}
a{color:var(--mask);text-underline-offset:.18em;text-decoration-thickness:1px}
a:hover{text-decoration-thickness:2px}
:focus-visible{outline:2px solid var(--focus);outline-offset:3px;border-radius:2px}
.skip{position:absolute;left:-999px;top:0;background:var(--panel);padding:8px 12px;z-index:9}
.skip:focus{left:12px}

/* ---------- frame ---------- */
.frame{display:grid;grid-template-columns:268px minmax(0,1fr);min-height:100vh}
.rail{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 22px 28px 26px;
  border-right:1px solid var(--rule);font-family:var(--sans);font-size:14.5px;line-height:1.35}
.brand{display:block;text-decoration:none;color:var(--ink);margin-bottom:26px}
.brand b{display:block;font-size:19px;font-weight:700;font-stretch:112%;letter-spacing:-.01em}
.brand span{display:block;color:var(--mute);font-size:13px;margin-top:3px}
.rail ol{list-style:none;margin:0;padding:0}
.rail li a{display:grid;grid-template-columns:22px 1fr;gap:6px;padding:7px 8px;margin:1px 0;
  border-radius:6px;color:var(--ink2);text-decoration:none}
.rail li a:hover{background:var(--mask-soft);color:var(--ink)}
.rail li a[aria-current="page"]{background:var(--mask);color:#fff}
.rail li a[aria-current="page"] .n{color:inherit}
:root[data-theme="dark"] .rail li a[aria-current="page"]{color:#10261b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .rail li a[aria-current="page"]{color:#10261b}}
.rail .n{font-variant-numeric:tabular-nums;color:var(--mute);font-weight:600}
.rail .sep{height:1px;background:var(--rule);margin:12px 8px}
.rail .foot{margin-top:28px;color:var(--mute);font-size:13px}
.rail .foot a{color:var(--mute)}
.theme{margin-top:14px;font:inherit;font-size:13px;background:none;border:1px solid var(--rule);
  color:var(--ink2);border-radius:6px;padding:5px 10px;cursor:pointer}
.menu{display:none}

main{padding:56px clamp(20px,5vw,72px) 96px;min-width:0}
.col{max-width:44rem}
.wide{max-width:64rem}

/* ---------- type ---------- */
h1,h2,h3,h4{font-family:var(--sans);color:var(--ink);line-height:1.12;font-weight:700}
h1{font-size:clamp(2.1rem,4.4vw,3.35rem);letter-spacing:-.022em;font-stretch:108%;
  margin:0 0 .5em;max-width:18ch}
h2{font-size:1.72rem;letter-spacing:-.012em;margin:2.6em 0 .55em;font-stretch:104%}
h3{font-size:1.2rem;margin:2em 0 .45em}
h4{font-size:1rem;margin:1.6em 0 .3em}
p{margin:0 0 1em}
.lede{font-size:1.24rem;line-height:1.5;color:var(--ink2);max-width:40rem}
.chapter{font-family:var(--sans);color:var(--mask);font-weight:600;font-size:15px;margin:0 0 10px}
small,.small{font-size:.86rem;color:var(--mute)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.84em;
  background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:.05em .35em}
pre{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px;
  overflow:auto;font-size:14.5px;line-height:1.5}
pre code{border:0;padding:0;background:none;font-size:inherit}
ul,ol{padding-left:1.25em;margin:0 0 1.1em}
li{margin:.25em 0}
strong{font-weight:650}
hr{border:0;border-top:1px solid var(--rule);margin:3em 0}

/* ---------- a measured number: value + the condition it was measured under ---------- */
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:0;
  margin:1.6em 0 2em;border-top:2px solid var(--ink)}
.col .facts{grid-template-columns:repeat(2,minmax(0,1fr))}
.fact{padding:14px 18px 16px 0;border-bottom:1px solid var(--rule)}
.fact .v{font-family:var(--sans);font-size:2.05rem;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.05;font-stretch:104%}
.fact .l{font-family:var(--sans);font-size:15px;font-weight:600;margin-top:6px;line-height:1.3}
.fact .c{font-size:13.5px;color:var(--mute);margin-top:6px;line-height:1.4}
.fact.good .v{color:var(--mask)} .fact.warn .v{color:var(--heat)} .fact.bad .v{color:var(--fault)}

/* ---------- tables ---------- */
.tbl{overflow-x:auto;margin:1.2em 0 1.6em;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;font-family:var(--sans);font-size:14.5px;line-height:1.35;
  font-variant-numeric:tabular-nums;min-width:100%}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:2px solid var(--ink);
  padding:7px 14px 7px 0;vertical-align:bottom}
td{border-bottom:1px solid var(--rule);padding:7px 14px 7px 0;vertical-align:top}
td.num,th.num{text-align:right}
tr.hl td{background:var(--mask-soft)}
caption{caption-side:bottom;text-align:left;font-family:var(--serif);font-size:13.5px;
  color:var(--mute);padding-top:8px}

/* ---------- notes ---------- */
.note{border-left:3px solid var(--steel);padding:2px 0 2px 16px;margin:1.4em 0;color:var(--ink2)}
.note.heat{border-color:var(--heat)} .note.fault{border-color:var(--fault)}
.note.mask{border-color:var(--mask)}
.note b:first-child{font-family:var(--sans);color:var(--ink)}

/* ---------- figures: result plots are printed plates, light in both themes ---------- */
figure{margin:1.8em 0 2.2em}
figure .plate{background:var(--plate);border:1px solid var(--rule);border-radius:8px;padding:10px;
  overflow-x:auto}
figure .plate img{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{font-size:14px;color:var(--mute);margin-top:8px;max-width:44rem;line-height:1.45}

/* ---------- the board ---------- */
.hero{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:48px;align-items:center;
  margin-bottom:24px}
.boards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.board{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:7px}
.board .g{display:grid;grid-template-columns:repeat(8,1fr);gap:2px}
.board .g i{display:block;aspect-ratio:1;border-radius:1.5px;background:var(--empty)}
.board .bl{font-family:var(--sans);font-size:11px;color:var(--mute);margin-top:5px;
  display:flex;justify-content:space-between}
.board.hot{border-color:var(--t-fixture)}
.tl{display:flex;flex-wrap:wrap;gap:4px 14px;font-family:var(--sans);font-size:13px;
  color:var(--ink2);margin-top:12px}
.tl span{display:inline-flex;align-items:center;gap:6px}
.tl i{width:11px;height:11px;border-radius:2px;display:inline-block}
.boardcap{font-size:14px;color:var(--mute);margin-top:10px;line-height:1.45}

.t-PASS{background:var(--t-pass)!important}.t-WATCH{background:var(--t-watch)!important}
.t-REVIEW{background:var(--t-review)!important}.t-REJECT{background:var(--t-reject)!important}
.t-FIXTURE_SUSPECT{background:var(--t-fixture)!important}
.t-MEASUREMENT_INVALID{background:var(--t-invalid)!important}
.board .g i.t-PASS{background:color-mix(in srgb,var(--t-pass) 26%,var(--empty))!important}

.chip{display:inline-block;font-family:var(--sans);font-size:13px;font-weight:600;
  padding:2px 9px;border-radius:999px;color:#fff;white-space:nowrap}

/* ---------- chapter list on the overview ---------- */
.toc{list-style:none;padding:0;margin:1.4em 0;border-top:1px solid var(--rule)}
.toc li{margin:0;border-bottom:1px solid var(--rule)}
.toc a{display:grid;grid-template-columns:34px minmax(0,1fr);gap:10px;padding:13px 0;
  text-decoration:none;color:var(--ink)}
.toc a:hover .t{color:var(--mask)}
.toc .n{font-family:var(--sans);font-weight:700;color:var(--mute);font-variant-numeric:tabular-nums}
.toc .t{font-family:var(--sans);font-weight:600;display:block}
.toc .d{display:block;color:var(--ink2);font-size:15.5px;margin-top:2px}

.pipeline{margin:1.6em 0 2em}
.pipeline svg{width:100%;height:auto;display:block}

.btn{display:inline-block;font-family:var(--sans);font-weight:600;font-size:15.5px;padding:10px 18px;
  border-radius:7px;background:var(--mask);color:#fff;text-decoration:none;margin:6px 10px 6px 0}
.btn.ghost{background:none;color:var(--mask);border:1.5px solid var(--mask)}
[data-theme="dark"] .btn:not(.ghost){color:#10261b}

.pager{display:flex;justify-content:space-between;gap:16px;margin-top:4.5em;padding-top:18px;
  border-top:1px solid var(--rule);font-family:var(--sans);font-size:15px}
.pager a{text-decoration:none;display:block;max-width:48%}
.pager small{display:block;color:var(--mute);font-size:12.5px}
.pager .next{text-align:right;margin-left:auto}

@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .btn:not(.ghost){color:#10261b}}
@media (max-width:980px){
  .hero{grid-template-columns:1fr;gap:28px}
}
@media (max-width:820px){
  body{font-size:17px}
  .frame{display:block}
  .rail{position:static;height:auto;border-right:0;border-bottom:1px solid var(--rule);padding:14px 16px}
  .brand{margin-bottom:0}
  .rail nav{display:none;margin-top:14px}
  .rail.open nav{display:block}
  .menu{display:inline-block;position:absolute;right:16px;top:16px;font:600 14px var(--sans);
    background:none;border:1px solid var(--rule);color:var(--ink);border-radius:6px;padding:6px 12px}
  main{padding:32px 16px 72px}
  .boards{grid-template-columns:repeat(2,1fr)}
  .facts{grid-template-columns:1fr 1fr}
  .fact .v{font-size:1.65rem}
}
@media (prefers-reduced-motion:no-preference){
  .board .g i{transition:transform .15s}
  .board .g i:hover{transform:scale(1.35)}
}
"""

JS = r"""
(function(){
  var r=document.documentElement,b=document.getElementById('theme');
  try{var s=localStorage.getItem('theme'); if(s) r.setAttribute('data-theme',s);}catch(e){}
  function label(){var d=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
    if(b) b.textContent=d==='dark'?'Light theme':'Dark theme';}
  if(b){b.onclick=function(){var d=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
    d=d==='dark'?'light':'dark'; r.setAttribute('data-theme',d); try{localStorage.setItem('theme',d)}catch(e){} label();};}
  label();
  var m=document.getElementById('menu'),rail=document.querySelector('.rail');
  if(m) m.onclick=function(){var o=rail.classList.toggle('open'); m.setAttribute('aria-expanded',o);};
})();
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100..125,400..800'
         '&family=Source+Serif+4:ital,opsz,wght@0,8..60,400..700;1,8..60,400..600&display=swap" '
         'rel="stylesheet">')

ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E"
        "%3Crect width='16' height='16' rx='3' fill='%231E5B42'/%3E"
        "%3Cg fill='%23DCEBE3'%3E%3Crect x='3' y='3' width='4' height='4'/%3E%3Crect x='9' y='3' "
        "width='4' height='4'/%3E%3Crect x='3' y='9' width='4' height='4'/%3E%3C/g%3E"
        "%3Crect x='9' y='9' width='4' height='4' fill='%23E0A24E'/%3E%3C/svg%3E")


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def page(slug: str, body: str, *, commit: str, version: str) -> str:
    idx = [s for s, _, _ in CHAPTERS].index(slug)
    _, short, full = CHAPTERS[idx]
    title = "Burn-in screening" if slug == "index" else f"{short} | Burn-in screening"
    nav = []
    for i, (s, sh, _) in enumerate(CHAPTERS):
        cur = ' aria-current="page"' if s == slug else ""
        n = NUMBERED.get(s)
        num = f"{n}" if n else ""
        if i == 10:
            nav.append('<li aria-hidden="true"><div class="sep"></div></li>')
        nav.append(f'<li><a href="{s}.html"{cur}><span class="n">{num}</span>'
                   f'<span>{esc(sh)}</span></a></li>')
    prev_ = CHAPTERS[idx - 1] if idx > 0 else None
    next_ = CHAPTERS[idx + 1] if idx + 1 < len(CHAPTERS) else None
    pager = '<nav class="pager" aria-label="Chapter">'
    if prev_:
        pager += f'<a href="{prev_[0]}.html"><small>Previous</small>{esc(prev_[1])}</a>'
    if next_:
        pager += f'<a class="next" href="{next_[0]}.html"><small>Next</small>{esc(next_[1])}</a>'
    pager += "</nav>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(full)}">
<link rel="icon" href="{ICON}">
{FONTS}
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="frame">
<aside class="rail">
  <a class="brand" href="index.html"><b>Burn-in screening</b><span>SIH26170 &middot; ISRO</span></a>
  <button class="menu" id="menu" aria-expanded="false" aria-controls="chapters">Chapters</button>
  <nav id="chapters" aria-label="Chapters"><ol>{''.join(nav)}</ol>
  <div class="foot">Generated from <code>results/</code> by <code>make_docs.py</code><br>
  {esc(version)}, commit <code>{esc(commit)}</code><br>
  <button class="theme" id="theme" type="button">Dark theme</button></div></nav>
</aside>
<main id="main">
{body}
{pager}
</main>
</div>
<script>{JS}</script>
</body>
</html>
"""


# ---------------- helpers ----------------

def pct(v, d=1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}%"


def num(v, d=3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{d}f}"


def fact(value: str, label: str, cond: str, tone: str = "") -> str:
    return (f'<div class="fact {tone}"><div class="v">{value}</div>'
            f'<div class="l">{label}</div><div class="c">{cond}</div></div>')


def facts(*items) -> str:
    return '<div class="facts">' + "".join(items) + "</div>"


def table(head: list[str], rows: list[list], *, numeric: set[int] | None = None,
          caption: str = "", hl: set[int] | None = None) -> str:
    numeric = numeric or set()
    hl = hl or set()
    th = "".join(f'<th class="num">{h}</th>' if i in numeric else f"<th>{h}</th>"
                 for i, h in enumerate(head))
    body = []
    for r_i, r in enumerate(rows):
        tds = "".join(f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>"
                      for i, c in enumerate(r))
        cls = ' class="hl"' if r_i in hl else ""
        body.append(f"<tr{cls}>{tds}</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f'<div class="tbl"><table>{cap}<thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def fig(src: str, alt: str, caption: str) -> str:
    return (f'<figure><div class="plate"><img src="{src}" alt="{esc(alt)}" loading="lazy">'
            f'</div><figcaption>{caption}</figcaption></figure>')


def note(text: str, tone: str = "") -> str:
    return f'<div class="note {tone}">{text}</div>'


TIER_VAR = {"PASS": "--t-pass", "WATCH": "--t-watch", "REVIEW": "--t-review",
            "REJECT": "--t-reject", "FIXTURE_SUSPECT": "--t-fixture",
            "MEASUREMENT_INVALID": "--t-invalid"}


def chip(tier: str) -> str:
    return (f'<span class="chip" style="background:var({TIER_VAR[tier]})">'
            f'{esc(tier.replace("_", " ").title().replace("Suspect", "suspect").replace("Invalid", "invalid"))}</span>')
