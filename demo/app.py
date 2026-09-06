"""QA inspector for burn-in screening \u2014 Streamlit front end.

Run from the repository root:

    streamlit run demo/app.py

This is not a mock-up and it does not read a frozen export. It loads the same
`explain/` stack that writes the PDF disposition reports and calls it live, so
every tier, evidence line, score and forecast on screen is the committed
pipeline's actual output for that component. Any of the 24,000 held-out test
components can be inspected, not a curated subset.

Ground truth is loaded (it lives in the same Dataset object) but is never an
input to anything displayed. It is revealed only behind the "reveal" control,
after the inspector has seen the evidence and decided.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explain.decision import ACTIONS, DecisionPolicy          # noqa: E402
from explain.evidence import Evidence                          # noqa: E402
from explain.rules import REASON_TEXT, UNITS, PartExplanation  # noqa: E402

TIER_ORDER = ["REJECT", "REVIEW", "FIXTURE_SUSPECT", "MEASUREMENT_INVALID",
              "WATCH", "PASS"]
TIER_COLOR = {"PASS": "#2A6A41", "WATCH": "#1F5C86", "REVIEW": "#8A5C05",
              "REJECT": "#A32E19", "FIXTURE_SUSPECT": "#5C3E8E",
              "MEASUREMENT_INVALID": "#5B6668"}
# The worklist column is narrow: full tier names truncate to "FIXTURE_SU..." and
# push the score column off the edge entirely. Short labels are used ONLY in
# that table; every other surface shows the real tier name.
TIER_SHORT = {"PASS": "PASS", "WATCH": "WATCH", "REVIEW": "REVIEW",
              "REJECT": "REJECT", "FIXTURE_SUSPECT": "FIXTURE",
              "MEASUREMENT_INVALID": "INVALID"}

st.set_page_config(page_title="Burn-In Inspector", page_icon="\U0001f321\ufe0f",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
  .block-container{padding-top:2.2rem;padding-bottom:2rem;max-width:1500px}
  h1,h2,h3{letter-spacing:-.01em}
  .prov{font-family:ui-monospace,"IBM Plex Mono",Consolas,monospace;
        font-size:.72rem;opacity:.7}
  .provnote{font-size:.8rem;opacity:.85;max-width:95ch;margin:.3rem 0 1rem}
  .tierstrip{display:flex;border:1px solid rgba(128,145,145,.35);margin-bottom:1rem}
  .tiercell{flex:1;padding:8px 12px;border-right:1px solid rgba(128,145,145,.35)}
  .tiercell:last-child{border-right:none}
  .tiercell .n{font-size:1.5rem;font-weight:700;line-height:1.1;
               font-variant-numeric:tabular-nums}
  .tiercell .t{font-size:.62rem;font-weight:700;letter-spacing:.09em;
               text-transform:uppercase}
  .chip{display:inline-block;font-size:.66rem;font-weight:700;letter-spacing:.08em;
        text-transform:uppercase;padding:3px 9px;border:1px solid currentColor}
  .evid{width:100%;border-collapse:collapse;font-size:.8rem}
  .evid td{padding:6px 10px 6px 0;border-bottom:1px solid rgba(128,145,145,.25);
           vertical-align:top}
  .evid td:first-child{width:170px;font-size:.68rem;font-weight:700;opacity:.65;
        letter-spacing:.05em;text-transform:uppercase;white-space:nowrap}
  .evid td:last-child{font-family:ui-monospace,"IBM Plex Mono",Consolas,monospace}
  .action{border-left:3px solid currentColor;padding:7px 0 7px 13px;
          margin:.6rem 0;font-size:.95rem}
  .note{font-size:.76rem;opacity:.62;max-width:88ch;margin-top:.4rem}
  .cellgrid{display:grid;gap:2px}
  .cellgrid div{aspect-ratio:1;min-width:10px}
  .legend{font-size:.72rem;opacity:.7;margin:.3rem 0 .8rem}
  .legend i{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;
            vertical-align:-1px}
  .legend i:first-of-type{margin-left:0}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the frozen pipeline (once)\u2026")
def load():
    e = Evidence(str(ROOT / "data"))
    pol = DecisionPolicy(e)
    pool = e.scores["CUM_C4"] >= pol.thr_review["CUM_C4"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=str(ROOT), capture_output=True,
                                text=True).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return e, pol, pool, commit


@st.cache_data(show_spinner="Dispositioning every component in the lot\u2026")
def lot_frame(lot: str) -> pd.DataFrame:
    """Tier + headline fields for every part in one lot.

    Uses DecisionPolicy.decide, which is the same call the PDF reports and the
    explainability metrics go through. Cached per lot: the first visit costs a
    few seconds, afterwards it is instant.
    """
    e, pol, pool, _ = load()
    rows = []
    for cid in e.gt.index[e.lot == lot]:
        d = pol.decide(cid, pool)
        rows.append({
            "component": cid,
            "tier": d["tier"],
            "why": d["reasons"][0] if d["reasons"] else "",
            "score": float(e.scores.loc[cid, "CUM_C4"]),
        })
    df = pd.DataFrame(rows)
    df["_rank"] = df["tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    return df.sort_values(["_rank", "score"], ascending=[True, False]) \
             .drop(columns="_rank").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def explain(cid: str) -> dict:
    e, pol, pool, _ = load()
    x = PartExplanation(e, pol, cid, pool)
    return {
        "tier": x.tier, "primary": x.primary, "secondary": x.secondary,
        "lines": [list(t) for t in x.lines],
        "hypotheses": [h[0] if isinstance(h, (list, tuple)) else str(h)
                       for h in x.hypotheses],
        "board": str(x.meta["board_id"]),
        "row": int(x.meta["socket_row"]), "col": int(x.meta["socket_col"]),
    }


# --------------------------------------------------------------------------
# trajectory chart, drawn as inline SVG so it matches the report styling
# --------------------------------------------------------------------------

def traj_svg(e, cid: str, lot: str, prm: str, ink: str) -> str:
    W, H, L, R, T, B = 330, 150, 52, 10, 12, 26
    cps = list(e.checkpoints)
    med, lo, hi, vals = [], [], [], []
    for t in cps:
        st_ = e.lot_stats(lot, t).loc[prm]
        m, s = float(st_["median"]), float(st_["sigma_mad"])
        med.append(m)
        lo.append(max(m - 6 * s, e.limits[prm][0]))
        hi.append(min(m + 6 * s, e.limits[prm][1]))
        v = e.wide.loc[cid, (prm, t)]
        vals.append(float(v) if np.isfinite(v) else None)
    fin = [v for v in vals if v is not None]
    allv = med + lo + hi + fin
    mn, mx = min(allv), max(allv)
    if mn == mx:
        mn, mx = mn - 1, mx + 1
    pad = (mx - mn) * .08
    mn, mx = mn - pad, mx + pad
    X = lambda i: L + (W - L - R) * (cps[i] - cps[0]) / (cps[-1] - cps[0])
    Y = lambda v: T + (H - T - B) * (1 - (v - mn) / (mx - mn))
    line = lambda a: " ".join(("L" if i else "M") + f"{X(i):.1f},{Y(v):.1f}"
                              for i, v in enumerate(a))
    band = line(hi) + " " + " ".join(f"L{X(i):.1f},{Y(lo[i]):.1f}"
                                     for i in range(len(lo) - 1, -1, -1)) + " Z"
    segs, cur, dots = [], [], ""
    for i, v in enumerate(vals):
        if v is None:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append((i, v))
            dots += f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.2" fill="{ink}"/>'
    if cur:
        segs.append(cur)
    pp = " ".join(" ".join(("L" if k else "M") + f"{X(i):.1f},{Y(v):.1f}"
                           for k, (i, v) in enumerate(s)) for s in segs)
    ticks = "".join(f'<text x="{X(i):.1f}" y="{H-8}" font-size="10" '
                    f'fill="currentColor" opacity=".6" text-anchor="middle">'
                    f'{int(t)}h</text>' for i, t in enumerate(cps))
    ylab = "".join(f'<text x="{L-6}" y="{Y(v)+3:.1f}" font-size="10" '
                   f'fill="currentColor" opacity=".6" text-anchor="end">'
                   f'{v:.3g}</text>' for v in (mn + (mx - mn) * .94,
                                               mn + (mx - mn) * .06))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="{prm} trajectory for {cid}">'
            f'<path d="{band}" fill="{ink}" opacity=".11"/>'
            f'<path d="{line(med)}" fill="none" stroke="currentColor" '
            f'opacity=".55" stroke-width="1.3" stroke-dasharray="4,3"/>'
            f'<path d="{pp}" fill="none" stroke="{ink}" stroke-width="2.1"/>'
            f'{dots}{ticks}{ylab}</svg>')


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

e, pol, pool, commit = load()
test_lots = list(pol.e.split.test)

st.title("Burn-In Screening \u2014 QA Inspector")
st.markdown(
    f'<div class="prov">{e.ds.cfg.get("version", "dataset-v1.1")} \u00b7 commit {commit} \u00b7 '
    f'{len(test_lots)} held-out test lots \u00b7 {len(test_lots)*500:,} components</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="provnote">Every tier, evidence line, score and forecast below is '
    'produced live by the committed pipeline \u2014 the same <code>explain/</code> code '
    'that writes the PDF disposition reports. Nothing here is mocked or '
    'pre-baked. No detector was fitted on any lot shown. Ground truth is never an '
    'input to a decision and stays hidden until you ask for it.</div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.header("Lot")
    lot = st.selectbox("Production batch", test_lots,
                       index=test_lots.index("LOT238") if "LOT238" in test_lots else 0,
                       help="All 48 lots here are held-out test lots.")
    st.caption("LOT238 carries a 64-part chamber fixture artifact \u2014 the clearest "
               "demonstration of the spatial layer. LOT192 has a chamber trip that "
               "invalidates readings.")
    df = lot_frame(lot)
    st.header("Filter")
    present = [t for t in TIER_ORDER if (df["tier"] == t).any()]
    choice = st.radio("Show", ["Needs action"] + present, index=0,
                      format_func=lambda s: s.replace("_", " ").title()
                      if s != "Needs action" else "Needs action (not PASS)")
    st.divider()
    st.caption(f"Thresholds: REVIEW at a {pol.review_budget*100:.0f}% yield-loss "
               f"budget, WATCH at {pol.watch_budget*100:.0f}%, both calibrated on "
               "validation lots the detectors never saw.")

counts = df["tier"].value_counts().to_dict()
strip = '<div class="tierstrip">'
for t in TIER_ORDER:
    if not counts.get(t):
        continue
    strip += (f'<div class="tiercell"><div class="n">{counts[t]}</div>'
              f'<div class="t" style="color:{TIER_COLOR[t]}">'
              f'{t.replace("_", " ")}</div></div>')
st.markdown(strip + "</div>", unsafe_allow_html=True)

view = df if choice == "Needs action" and False else df
if choice == "Needs action":
    view = df[df["tier"] != "PASS"]
elif choice in TIER_ORDER:
    view = df[df["tier"] == choice]

tab_part, tab_board = st.tabs(["Worklist & part detail", "Board map"])

with tab_part:
    left, right = st.columns([1.15, 2], gap="medium")
    with left:
        st.subheader(f"Worklist \u00b7 {len(view)}")
        st.caption("Sorted by severity, then by fused anomaly score. "
                   "Click a row to open it.")
        # The lot is already chosen in the sidebar, so repeating "LOT238-" in
        # every row only costs width. Show the component suffix; the full id is
        # kept in `view` for the lookup.
        disp = view.assign(
            part=view["component"].str.rsplit("-", n=1).str[-1],
            call=view["tier"].map(TIER_SHORT))[["part", "call", "score"]]
        sel = st.dataframe(
            disp, hide_index=True, width="stretch", height=560,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "part": st.column_config.TextColumn("Part", width="small"),
                "call": st.column_config.TextColumn("Call", width="small"),
                "score": st.column_config.NumberColumn(
                    "Score", format="%.2f", width="small",
                    help="Fused anomaly score (CUM_C4)")})
        rows = sel.selection.rows if sel and sel.selection else []
        cid = view.iloc[rows[0]]["component"] if rows else None

    with right:
        if cid is None:
            st.info("Select a component on the left to see why the system reached "
                    "its decision.")
        else:
            x = explain(cid)
            col = TIER_COLOR[x["tier"]]
            st.markdown(
                f'### `{cid}` &nbsp; <span class="chip" style="color:{col}">'
                f'{x["tier"].replace("_", " ")}</span>', unsafe_allow_html=True)
            st.caption(f'board {x["board"]} \u00b7 socket (r{x["row"]}, c{x["col"]})')
            st.markdown(
                f'**Primary reason:** {REASON_TEXT.get(x["primary"], x["primary"])}'
                + (f'  \n**Secondary:** '
                   f'{REASON_TEXT.get(x["secondary"], x["secondary"])}'
                   if x["secondary"] else ""))
            st.markdown(f'<div class="action" style="color:{col}">'
                        f'{ACTIONS[x["tier"]]}</div>', unsafe_allow_html=True)

            st.markdown("#### Evidence the decision rests on")
            body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                           for k, v in x["lines"])
            st.markdown(f'<table class="evid">{body}</table>',
                        unsafe_allow_html=True)
            if x["hypotheses"]:
                st.markdown(f'<div class="note"><b>Mechanism hypothesis:</b> '
                            f'{"; ".join(x["hypotheses"])}</div>',
                            unsafe_allow_html=True)

            st.markdown("#### Trajectory against the lot it was burned in with")
            st.markdown('<div class="note">Solid line is this part, dashed is the '
                        'lot median, shaded band is the AEC-Q001 dynamic PAT limit '
                        'at 6\u03c3 clipped to the datasheet limit. A gap means no '
                        'measurement at that checkpoint.</div>',
                        unsafe_allow_html=True)
            cs = st.columns(2)
            for i, prm in enumerate(e.params):
                with cs[i % 2]:
                    st.caption(f"{prm} ({UNITS[prm]})")
                    st.html(traj_svg(e, cid, lot, prm, col))

            st.markdown("#### Module B \u2014 168 h forecast from 0 h and 24 h only")
            mb = []
            for p in e.params:
                if cid not in e.mb_point.index:
                    continue
                pt = float(e.mb_point.loc[cid, p])
                ub = float(e.mb_upper.loc[cid, p])
                ls = float(e.l_safe[p].get(cid, np.nan))
                if not np.isfinite(pt):
                    continue
                breach = np.isfinite(ub) and np.isfinite(ls) and ub > ls
                mb.append(f"<tr><td>{p}</td><td>predicted {pt:.3f} {UNITS[p]} \u00b7 "
                          f"95% upper {ub:.3f} \u00b7 <b style='color:"
                          f"{TIER_COLOR['REJECT'] if breach else TIER_COLOR['PASS']}'>"
                          f"{'EXCEEDS' if breach else 'within'} safe limit "
                          f"{ls:.3f}</b></td></tr>")
            if mb:
                st.markdown(f'<table class="evid">{"".join(mb)}</table>',
                            unsafe_allow_html=True)
                st.markdown('<div class="note">The safe limit comes from PRIOR lots '
                            'only. This lot\'s own 168 h readings do not exist yet '
                            'when the 24 h decision is made.</div>',
                            unsafe_allow_html=True)

            st.markdown("#### Ground truth")
            with st.expander("Reveal what this component actually was", False):
                g = e.gt.loc[cid]
                truth = str(g["defect_type"])
                is_def = bool(e.y[cid])
                held = x["tier"] in ("REVIEW", "REJECT")
                ok = is_def == held
                c1, c2, c3 = st.columns(3)
                c1.metric("Injected type", truth)
                c2.metric("Truly defective", "YES" if is_def else "no")
                c3.metric("Severity", str(g.get("severity") or "\u2014"))
                if ok and is_def:
                    st.success("Caught. Held before it could ship.")
                elif ok:
                    st.success("Correctly released. A good part was not scrapped.")
                elif is_def:
                    st.error("MISSED \u2014 this defect escaped screening.")
                else:
                    st.warning("FALSE ALARM \u2014 a good part was held.")

with tab_board:
    st.subheader("Board map \u2014 is it the component, or the chamber?")
    st.markdown('<div class="note">Each square is a socket position. An anomaly '
                'that clusters by board position is the oven or the fixture, not '
                'the component. That is what the FIXTURE SUSPECT tier says, and why '
                'those parts are investigated rather than scrapped.</div>',
                unsafe_allow_html=True)
    leg = '<div class="legend">'
    for t in TIER_ORDER:
        if counts.get(t):
            leg += f'<i style="background:{TIER_COLOR[t]}"></i>{t.replace("_"," ")}'
    st.markdown(leg + "</div>", unsafe_allow_html=True)

    meta = e.ds.meas[e.ds.meas["lot_id"] == lot].drop_duplicates("component_id")
    meta = meta.set_index("component_id")[["board_id", "socket_row", "socket_col"]]
    tier_of = dict(zip(df["component"], df["tier"]))
    boards = sorted(meta["board_id"].unique())
    cols = st.columns(min(4, len(boards)))
    for i, b in enumerate(boards):
        sub = meta[meta["board_id"] == b]
        nr, nc = int(sub["socket_row"].max()) + 1, int(sub["socket_col"].max()) + 1
        at = {(int(r["socket_row"]), int(r["socket_col"])): cid
              for cid, r in sub.iterrows()}
        cells = ""
        for r in range(nr):
            for c in range(nc):
                cid2 = at.get((r, c))
                bg = (TIER_COLOR.get(tier_of.get(cid2, "PASS"), "#888")
                      if cid2 else "rgba(128,145,145,.18)")
                ttl = f"{cid2} \u00b7 {tier_of.get(cid2,'')}" if cid2 else ""
                cells += f'<div style="background:{bg}" title="{ttl}"></div>'
        n_flag = sum(1 for cid2 in at.values()
                     if tier_of.get(cid2, "PASS") != "PASS")
        with cols[i % len(cols)]:
            st.caption(f"{b} \u00b7 {n_flag}/{len(at)} flagged")
            st.markdown(f'<div class="cellgrid" style="grid-template-columns:'
                        f'repeat({nc},minmax(0,1fr));max-width:{nc*20}px">'
                        f'{cells}</div>', unsafe_allow_html=True)
