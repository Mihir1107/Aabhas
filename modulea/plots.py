"""Dependency-free SVG plotting (matplotlib is not installed in this env).

Every axis is labelled with units because these go into a deck.
"""
from __future__ import annotations

import numpy as np

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#17becf", "#8c564b", "#7f7f7f", "#bcbd22"]


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def line_chart(path, series, title, xlabel, ylabel, xlim=None, ylim=None,
               logx=False, width=900, height=520, legend_title=None,
               annotations=()):
    ml, mr, mt, mb = 78, 190, 46, 62
    xs = np.concatenate([s["x"] for s in series])
    ys = np.concatenate([s["y"] for s in series])
    x0, x1 = xlim or (float(np.nanmin(xs)), float(np.nanmax(xs)))
    y0, y1 = ylim or (float(np.nanmin(ys)), float(np.nanmax(ys)))
    if logx:
        x0 = max(x0, 1e-4)

    def sx(v):
        v = max(v, x0) if logx else v
        f = ((np.log10(v) - np.log10(x0)) / (np.log10(x1) - np.log10(x0))
             if logx else (v - x0) / (x1 - x0))
        return ml + f * (width - ml - mr)

    def sy(v):
        return height - mb - (v - y0) / (y1 - y0) * (height - mb - mt)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="system-ui,-apple-system,sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#fff"/>',
         f'<text x="{ml}" y="26" font-size="15" font-weight="600">{_esc(title)}</text>']

    xt = ([10 ** e for e in range(int(np.floor(np.log10(x0))), int(np.ceil(np.log10(x1))) + 1)]
          if logx else list(np.linspace(x0, x1, 6)))
    for v in xt:
        if v < x0 or v > x1:
            continue
        o.append(f'<line x1="{sx(v):.1f}" y1="{mt}" x2="{sx(v):.1f}" y2="{height - mb}" '
                 f'stroke="#eee"/>')
        lab = f"{v:g}"
        o.append(f'<text x="{sx(v):.1f}" y="{height - mb + 18}" font-size="11" '
                 f'text-anchor="middle" fill="#444">{lab}</text>')
    for v in np.linspace(y0, y1, 6):
        o.append(f'<line x1="{ml}" y1="{sy(v):.1f}" x2="{width - mr}" y2="{sy(v):.1f}" '
                 f'stroke="#eee"/>')
        o.append(f'<text x="{ml - 8}" y="{sy(v) + 4:.1f}" font-size="11" '
                 f'text-anchor="end" fill="#444">{v:g}</text>')
    o.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#333"/>')
    o.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" stroke="#333"/>')
    o.append(f'<text x="{(ml + width - mr) / 2:.0f}" y="{height - 16}" font-size="12" '
             f'text-anchor="middle" fill="#222">{_esc(xlabel)}</text>')
    o.append(f'<text x="18" y="{(mt + height - mb) / 2:.0f}" font-size="12" fill="#222" '
             f'transform="rotate(-90 18 {(mt + height - mb) / 2:.0f})" '
             f'text-anchor="middle">{_esc(ylabel)}</text>')

    for i, s in enumerate(series):
        c = s.get("color", PALETTE[i % len(PALETTE)])
        pts = [(sx(a), sy(b)) for a, b in zip(s["x"], s["y"])
               if np.isfinite(a) and np.isfinite(b) and a >= x0]
        if len(pts) > 1:
            d = " ".join(f"{'M' if j == 0 else 'L'}{a:.1f},{b:.1f}"
                         for j, (a, b) in enumerate(pts))
            o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" '
                     f'stroke-dasharray="{s.get("dash", "")}"/>')
        yy = mt + 6 + i * 19
        o.append(f'<line x1="{width - mr + 8}" y1="{yy}" x2="{width - mr + 30}" y2="{yy}" '
                 f'stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<text x="{width - mr + 35}" y="{yy + 4}" font-size="11" fill="#222">'
                 f'{_esc(s["label"])}</text>')
    if legend_title:
        o.append(f'<text x="{width - mr + 8}" y="{mt - 8}" font-size="11" '
                 f'font-weight="600" fill="#222">{_esc(legend_title)}</text>')
    for a in annotations:
        o.append(f'<line x1="{sx(a["x"]):.1f}" y1="{mt}" x2="{sx(a["x"]):.1f}" '
                 f'y2="{height - mb}" stroke="#c00" stroke-width="1" stroke-dasharray="4,3"/>')
        o.append(f'<text x="{sx(a["x"]) + 5:.1f}" y="{mt + 14}" font-size="10.5" '
                 f'fill="#c00">{_esc(a["label"])}</text>')
    o.append("</svg>")
    Pathwrite(path, "\n".join(o))


def grouped_bars(path, groups, categories, values, title, ylabel,
                 width=900, height=520, note=None, vmax=None):
    """values[g][c]. groups on the x axis, categories are the coloured bars."""
    ml, mr, mt, mb = 78, 190, 46, 78
    if vmax is None:
        vmax = max(max(v for v in row if np.isfinite(v)) for row in values) or 1.0
        vmax *= 1.12
        if "%" in ylabel:      # a percentage axis must not run past 100
            vmax = min(vmax, 100.0)
    pw = (width - ml - mr) / max(len(groups), 1)
    bw = pw / (len(categories) + 0.6)

    def sy(v):
        return height - mb - v / vmax * (height - mb - mt)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="system-ui,-apple-system,sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#fff"/>',
         f'<text x="{ml}" y="26" font-size="15" font-weight="600">{_esc(title)}</text>']
    for v in np.linspace(0, vmax, 6):
        o.append(f'<line x1="{ml}" y1="{sy(v):.1f}" x2="{width - mr}" y2="{sy(v):.1f}" stroke="#eee"/>')
        o.append(f'<text x="{ml - 8}" y="{sy(v) + 4:.1f}" font-size="11" text-anchor="end" '
                 f'fill="#444">{v:.0f}</text>')
    o.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#333"/>')
    for gi, g in enumerate(groups):
        for ci, c in enumerate(categories):
            v = values[gi][ci]
            if not np.isfinite(v):
                continue
            x = ml + gi * pw + 0.3 * bw + ci * bw
            o.append(f'<rect x="{x:.1f}" y="{sy(v):.1f}" width="{bw * 0.88:.1f}" '
                     f'height="{height - mb - sy(v):.1f}" fill="{PALETTE[ci % len(PALETTE)]}"/>')
        o.append(f'<text x="{ml + gi * pw + pw / 2:.1f}" y="{height - mb + 16}" font-size="10.5" '
                 f'text-anchor="middle" fill="#444">{_esc(g)}</text>')
    for ci, c in enumerate(categories):
        yy = mt + 6 + ci * 19
        o.append(f'<rect x="{width - mr + 8}" y="{yy - 8}" width="20" height="11" '
                 f'fill="{PALETTE[ci % len(PALETTE)]}"/>')
        o.append(f'<text x="{width - mr + 34}" y="{yy + 2}" font-size="11" fill="#222">'
                 f'{_esc(c)}</text>')
    o.append(f'<text x="18" y="{(mt + height - mb) / 2:.0f}" font-size="12" fill="#222" '
             f'transform="rotate(-90 18 {(mt + height - mb) / 2:.0f})" '
             f'text-anchor="middle">{_esc(ylabel)}</text>')
    if note:
        o.append(f'<text x="{ml}" y="{height - 14}" font-size="10.5" fill="#666">{_esc(note)}</text>')
    o.append("</svg>")
    Pathwrite(path, "\n".join(o))


def scatter(path, x, y, title, xlabel, ylabel, band=None, diagonal=True,
            colors=None, legend=None, width=900, height=560, note=None,
            max_points=4000, seed=0):
    """Scatter with an optional prediction-interval band."""
    ml, mr, mt, mb = 84, 180, 46, 64
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    cl = None if colors is None else np.asarray(colors)[ok]
    bd = None if band is None else np.asarray(band, float)[ok]
    if len(x) > max_points:
        idx = rng.choice(len(x), max_points, replace=False)
        x, y = x[idx], y[idx]
        cl = None if cl is None else cl[idx]
        bd = None if bd is None else bd[idx]
    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    pad = 0.05 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    sx = lambda v: ml + (v - lo) / (hi - lo) * (width - ml - mr)
    sy = lambda v: height - mb - (v - lo) / (hi - lo) * (height - mb - mt)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="system-ui,-apple-system,sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#fff"/>',
         f'<text x="{ml}" y="26" font-size="15" font-weight="600">{_esc(title)}</text>']
    for v in np.linspace(lo, hi, 6):
        o.append(f'<line x1="{ml}" y1="{sy(v):.1f}" x2="{width - mr}" y2="{sy(v):.1f}" stroke="#f0f0f0"/>')
        o.append(f'<text x="{ml - 8}" y="{sy(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#444">{v:.3g}</text>')
        o.append(f'<line x1="{sx(v):.1f}" y1="{mt}" x2="{sx(v):.1f}" y2="{height - mb}" stroke="#f0f0f0"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{height - mb + 18}" font-size="11" text-anchor="middle" fill="#444">{v:.3g}</text>')
    if bd is not None:
        pts_u = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in
                         sorted(zip(x, bd), key=lambda t: t[0]))
        o.append(f'<polyline points="{pts_u}" fill="none" stroke="#d62728" '
                 f'stroke-width="1.2" opacity="0.55"/>')
    for i in range(len(x)):
        c = "#1f77b4" if cl is None else ("#d62728" if cl[i] else "#8fa8bd")
        op = 0.5 if cl is None else (0.95 if cl[i] else 0.25)
        o.append(f'<circle cx="{sx(x[i]):.1f}" cy="{sy(y[i]):.1f}" r="1.8" '
                 f'fill="{c}" opacity="{op}"/>')
    if diagonal:
        o.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" '
                 f'y2="{sy(hi):.1f}" stroke="#333" stroke-width="1" stroke-dasharray="5,4"/>')
        o.append(f'<text x="{width - mr - 60}" y="{sy(hi) + 40:.0f}" font-size="10.5" '
                 f'fill="#555">perfect prediction</text>')
    o.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#333"/>')
    o.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" stroke="#333"/>')
    o.append(f'<text x="{(ml + width - mr) / 2:.0f}" y="{height - 16}" font-size="12" '
             f'text-anchor="middle" fill="#222">{_esc(xlabel)}</text>')
    o.append(f'<text x="20" y="{(mt + height - mb) / 2:.0f}" font-size="12" fill="#222" '
             f'transform="rotate(-90 20 {(mt + height - mb) / 2:.0f})" text-anchor="middle">{_esc(ylabel)}</text>')
    if legend:
        for i, (lab, col) in enumerate(legend):
            yy = mt + 8 + i * 19
            o.append(f'<circle cx="{width - mr + 16}" cy="{yy}" r="4" fill="{col}"/>')
            o.append(f'<text x="{width - mr + 26}" y="{yy + 4}" font-size="11" fill="#222">{_esc(lab)}</text>')
    if note:
        o.append(f'<text x="{ml}" y="{height - 2}" font-size="10.5" fill="#666">{_esc(note)}</text>')
    o.append("</svg>")
    Pathwrite(path, "\n".join(o))


def Pathwrite(path, text):
    from pathlib import Path
    Path(path).write_text(text)
