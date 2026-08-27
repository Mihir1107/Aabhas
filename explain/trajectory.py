"""Reusable trajectory plot. Standalone so the presentation team can call it."""
from __future__ import annotations

import numpy as np

from modulea import plots as pl


def trajectory_svg(path, checkpoints, values, lot_median, lot_lo, lot_hi,
                   limit_hi, limit_lo=None, forecast=None, forecast_lo=None,
                   forecast_hi=None, param="", unit="", title="", subtitle="",
                   width=720, height=420, safe_limit=None):
    """One part's trajectory against its lot.

    Shows: the lot's robust envelope (median +/- k sigma), the lot median
    trajectory, the datasheet limit, the lot-derived safe limit, the part's
    measured points, and the forecast with its uncertainty band. Everything an
    inspector needs to see whether the part is unusual FOR ITS LOT, which is the
    judgement the whole system rests on.
    """
    ml, mr, mt, mb = 74, 168, 44, 56
    t = np.asarray(checkpoints, float)
    v = np.asarray(values, float)
    ys = [y for y in np.concatenate([v, lot_lo, lot_hi, [limit_hi]]) if np.isfinite(y)]
    if forecast_hi is not None and np.isfinite(forecast_hi):
        ys.append(forecast_hi)
    lo_y, hi_y = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad = 0.08 * (hi_y - lo_y or 1)
    lo_y, hi_y = lo_y - pad, hi_y + pad
    tmax = float(t.max()) * 1.06

    def sx(a):
        return ml + a / tmax * (width - ml - mr)

    def sy(a):
        return height - mb - (a - lo_y) / (hi_y - lo_y) * (height - mb - mt)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="system-ui,-apple-system,sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#fff"/>',
         f'<text x="{ml}" y="20" font-size="13" font-weight="600">{pl._esc(title)}</text>']
    if subtitle:
        o.append(f'<text x="{ml}" y="35" font-size="10.5" fill="#666">{pl._esc(subtitle)}</text>')
    for a in np.linspace(lo_y, hi_y, 5):
        o.append(f'<line x1="{ml}" y1="{sy(a):.1f}" x2="{width - mr}" y2="{sy(a):.1f}" stroke="#f2f2f2"/>')
        o.append(f'<text x="{ml - 7}" y="{sy(a) + 4:.1f}" font-size="10" text-anchor="end" fill="#555">{a:.3g}</text>')
    # lot robust envelope
    env = (" ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(t, lot_hi)) + " " +
           " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(t[::-1], lot_lo[::-1])))
    o.append(f'<polygon points="{env}" fill="#4a90d9" opacity="0.13"/>')
    o.append('<path d="' + " ".join(
        f"{'M' if i == 0 else 'L'}{sx(a):.1f},{sy(b):.1f}"
        for i, (a, b) in enumerate(zip(t, lot_median))) +
        '" fill="none" stroke="#4a90d9" stroke-width="1.6" stroke-dasharray="6,3"/>')
    # limits
    o.append(f'<line x1="{ml}" y1="{sy(limit_hi):.1f}" x2="{width - mr}" y2="{sy(limit_hi):.1f}" '
             f'stroke="#c0392b" stroke-width="1.5"/>')
    o.append(f'<text x="{width - mr - 4}" y="{sy(limit_hi) - 4:.1f}" font-size="9.5" '
             f'text-anchor="end" fill="#c0392b">datasheet limit {limit_hi:g} {pl._esc(unit)}</text>')
    if safe_limit is not None and np.isfinite(safe_limit):
        o.append(f'<line x1="{ml}" y1="{sy(safe_limit):.1f}" x2="{width - mr}" y2="{sy(safe_limit):.1f}" '
                 f'stroke="#e67e22" stroke-width="1.2" stroke-dasharray="4,3"/>')
        o.append(f'<text x="{width - mr - 4}" y="{sy(safe_limit) - 4:.1f}" font-size="9.5" '
                 f'text-anchor="end" fill="#e67e22">lot-derived safe limit {safe_limit:.3g}</text>')
    # forecast band
    if forecast is not None and np.isfinite(forecast):
        tf = float(t.max())
        if forecast_hi is not None and np.isfinite(forecast_hi):
            o.append(f'<line x1="{sx(tf):.1f}" y1="{sy(forecast_lo):.1f}" '
                     f'x2="{sx(tf):.1f}" y2="{sy(forecast_hi):.1f}" stroke="#7b3fbf" '
                     f'stroke-width="7" opacity="0.30"/>')
        o.append(f'<circle cx="{sx(tf):.1f}" cy="{sy(forecast):.1f}" r="4.5" '
                 f'fill="none" stroke="#7b3fbf" stroke-width="2"/>')
    # measured trajectory
    fin = [(a, b) for a, b in zip(t, v) if np.isfinite(b)]
    if len(fin) > 1:
        o.append('<path d="' + " ".join(
            f"{'M' if i == 0 else 'L'}{sx(a):.1f},{sy(b):.1f}"
            for i, (a, b) in enumerate(fin)) +
            '" fill="none" stroke="#111" stroke-width="2.2"/>')
    for a, b in fin:
        o.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="3.6" fill="#111"/>')
    for a in t:
        o.append(f'<text x="{sx(a):.1f}" y="{height - mb + 16}" font-size="10" '
                 f'text-anchor="middle" fill="#555">{a:.0f}h</text>')
    o.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#333"/>')
    o.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" stroke="#333"/>')
    o.append(f'<text x="{(ml + width - mr) / 2:.0f}" y="{height - 10}" font-size="11" '
             f'text-anchor="middle" fill="#222">Burn-in time (hours at 125 C)</text>')
    o.append(f'<text x="16" y="{(mt + height - mb) / 2:.0f}" font-size="11" fill="#222" '
             f'transform="rotate(-90 16 {(mt + height - mb) / 2:.0f})" text-anchor="middle">'
             f'{pl._esc(param)} ({pl._esc(unit)})</text>')
    leg = [("this component", "#111"), ("lot median", "#4a90d9"),
           ("lot robust envelope", "#4a90d9"), ("forecast + 95% band", "#7b3fbf"),
           ("datasheet limit", "#c0392b")]
    for i, (lab, col) in enumerate(leg):
        yy = mt + 6 + i * 16
        o.append(f'<line x1="{width - mr + 6}" y1="{yy}" x2="{width - mr + 24}" y2="{yy}" '
                 f'stroke="{col}" stroke-width="2.4"/>')
        o.append(f'<text x="{width - mr + 28}" y="{yy + 3.5}" font-size="9.5" fill="#222">{pl._esc(lab)}</text>')
    o.append("</svg>")
    pl.Pathwrite(path, "\n".join(o))
    return path
