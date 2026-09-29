"""One-page QA Disposition Report, PDF.

The deliverable is not a SHAP plot. It is a document a QA inspector could act on
and a Group Head could sign, so it carries a sign-off block, a model version, and
every number with the comparison it was made against.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from explain.rules import UNITS
from explain.trajectory import trajectory_svg

TIER_COLOR = {"PASS": colors.HexColor("#1e7d32"), "WATCH": colors.HexColor("#b8860b"),
              "REVIEW": colors.HexColor("#c25e00"), "REJECT": colors.HexColor("#b00020"),
              "MEASUREMENT_INVALID": colors.HexColor("#5c5c5c"),
              "FIXTURE_SUSPECT": colors.HexColor("#00629b")}


def _svg_drawing(svg: Path, target_w_mm: float):
    """Convert the SVG into a native ReportLab drawing via svglib.

    Vector rather than raster: it stays sharp at any zoom, which matters for a
    document a Group Head may print. No external binary is required (neither
    rsvg-convert nor cairosvg is present in this environment), so the report
    builds anywhere the venv builds.
    """
    try:
        from svglib.svglib import svg2rlg
        d = svg2rlg(str(svg))
        if d is None:
            return None
        scale = (target_w_mm * mm) / d.width
        d.width *= scale
        d.height *= scale
        d.scale(scale, scale)
        return d
    except Exception:
        return None


def build_report(e, pol, x, cf_text, det_attr, shap_row, out_pdf: Path,
                 model_version: str, param_focus: str | None = None,
                 tmpdir: Path = Path("reports/_assets")):
    tmpdir.mkdir(parents=True, exist_ok=True)
    cid = x.cid
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=7.4, leading=9.2)
    small = ParagraphStyle("s", parent=body, fontSize=6.6, leading=8.0,
                           textColor=colors.HexColor("#555555"))
    h = ParagraphStyle("h", parent=ss["Heading4"], fontSize=8.6, leading=10,
                       spaceBefore=4, spaceAfter=2,
                       textColor=colors.HexColor("#222222"))
    mono = ParagraphStyle("m", parent=body, fontName="Courier", fontSize=6.5,
                          leading=8.0)

    # focus parameter: whichever drives the primary reason
    p = param_focus or x.signals.get("drift_param") or x.signals.get(
        "level_param") or e.params[0]
    end = max(e.checkpoints)
    t = np.array(e.checkpoints, float)
    vals = np.array([e.wide.loc[cid, (p, tt)] for tt in t], float)
    med, lo, hi = [], [], []
    for tt in t:
        st = e.lot_stats(e.lot[cid], tt).loc[p]
        med.append(st["median"])
        lo.append(st["median"] - 3 * st["sigma_mad"])
        hi.append(st["median"] + 3 * st["sigma_mad"])
    fc = float(e.mb_point.loc[cid, p]) if cid in e.mb_point.index else np.nan
    fu = float(e.mb_upper.loc[cid, p]) if cid in e.mb_upper.index else np.nan
    ls = float(e.l_safe[p].get(cid, np.nan))
    svg = tmpdir / f"{cid}_{p}.svg"
    trajectory_svg(svg, t, vals, np.array(med), np.array(lo), np.array(hi),
                   e.limits[p][1], forecast=fc, forecast_lo=fc - (fu - fc),
                   forecast_hi=fu, param=p, unit=UNITS[p], safe_limit=ls,
                   title=f"{p} trajectory vs lot {e.lot[cid]}",
                   subtitle="envelope = lot median +/- 3 robust sigma (MAD); "
                            "forecast from 0 h and 24 h only")
    drawing = _svg_drawing(svg, 74.0)      # sized so the report stays one page
    if drawing is None:
        # v1.1's PDFs were built without svglib and shipped with no chart at
        # all; the report's central figure must never vanish silently.
        raise RuntimeError("trajectory chart could not be rendered; install svglib "
                           "(pip install -r requirements.txt)")

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4,
                            leftMargin=13 * mm, rightMargin=13 * mm,
                            topMargin=8 * mm, bottomMargin=7 * mm,
                            title=f"QA Disposition {cid}")
    S = []
    S.append(Paragraph(
        "<b>QA DISPOSITION REPORT</b> &nbsp;&nbsp; burn-in parametric screening",
        ParagraphStyle("t", parent=ss["Heading2"], fontSize=12, spaceAfter=2)))
    meta = e.ds.meas[e.ds.meas["component_id"] == cid].iloc[0]
    # Model version gets a full-width row of its own: in a 28 mm cell it ran
    # into the timestamp beside it and both became unreadable.
    hdr = [["Component", cid, "Lot", e.lot[cid], "Board", meta["board_id"]],
           ["Socket", f"r{int(meta['socket_row'])} c{int(meta['socket_col'])}",
            "Generated",
            __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"), "", ""],
           ["Model version", model_version, "", "", "", ""]]
    tb = Table(hdr, colWidths=[22 * mm, 36 * mm, 20 * mm, 30 * mm, 18 * mm, 38 * mm])
    tb.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#666666")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#666666")),
        ("TEXTCOLOR", (4, 0), (4, -1), colors.HexColor("#666666")),
        ("SPAN", (1, 2), (-1, 2)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("TOPPADDING", (0, 0), (-1, -1), 1)]))
    S.append(tb)
    dt = Table([[Paragraph(f"<b>DECISION: {x.tier}</b>", ParagraphStyle(
        "d", parent=body, fontSize=11, textColor=colors.white))]],
        colWidths=[164 * mm])
    dt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TIER_COLOR[x.tier]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    S.append(Spacer(1, 3)); S.append(dt); S.append(Spacer(1, 3))

    S.append(Paragraph("LAYER 1 &mdash; deterministic evidence "
                       "(every value cites its comparison basis)", h))
    rows = [[Paragraph(f"<b>{k}</b>", body), Paragraph(v, body)] for k, v in x.lines]
    et = Table(rows, colWidths=[32 * mm, 132 * mm])
    et.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e8e8e8")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    S.append(et)

    from explain.rules import REASON_TEXT
    from explain.decision import ACTIONS
    rsn = [[Paragraph("<b>Primary reason</b>", body),
            Paragraph(REASON_TEXT[x.primary], body)]]
    if x.secondary:
        rsn.append([Paragraph("<b>Secondary</b>", body),
                    Paragraph(REASON_TEXT[x.secondary], body)])
    rsn.append([Paragraph("<b>Action</b>", body), Paragraph(ACTIONS[x.tier], body)])
    rt = Table(rsn, colWidths=[32 * mm, 132 * mm])
    rt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f8")),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    S.append(Spacer(1, 2)); S.append(rt)

    if drawing is not None:
        S.append(Spacer(1, 3))
        S.append(drawing)

    # Layer 2
    S.append(Paragraph("LAYER 2 &mdash; model attribution", h))
    d = det_attr.head(5)
    dr = [["detector", "normalised strength", "fired"]] + [
        [r.detector, f"{r.normalised_strength:.2f}", "YES" if r.fired else "no"]
        for r in d.itertuples()]
    dtb = Table(dr, colWidths=[42 * mm, 34 * mm, 14 * mm])
    dtb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 6.6),
                             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                             ("TOPPADDING", (0, 0), (-1, -1), 0.8),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8)]))
    sh = shap_row.drop(labels=["__base__"], errors="ignore")
    sh = sh.reindex(sh.abs().sort_values(ascending=False).index).head(5)
    sr = [["feature (Module B)", f"SHAP ({UNITS[p]})"]] + [
        [k.replace(f"{p}__", "").replace("z__", "z:"), f"{v:+.3f}"] for k, v in sh.items()]
    stb = Table(sr, colWidths=[44 * mm, 26 * mm])
    stb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 6.6),
                             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                             ("TOPPADDING", (0, 0), (-1, -1), 0.8),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8)]))
    S.append(Table([[dtb, stb]], colWidths=[92 * mm, 72 * mm],
                   style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])))
    S.append(Paragraph(
        "Detector strength is distance past that detector's own operating "
        "threshold (1.0 = exactly at it). SHAP values are exact TreeSHAP from the "
        "LightGBM drift model, in target units. <b>Caveat:</b> v0 and v24 are "
        "0.99 correlated, so SHAP credit between them is not identifiable &mdash; "
        "a Huber model of equal accuracy attributes it the other way round. "
        "Treat Layer 2 as supporting evidence; Layer 1 is the auditable record.",
        small))

    S.append(Paragraph("COUNTERFACTUAL", h))
    S.append(Paragraph(cf_text, body))

    S.append(Paragraph("LAYER 3 &mdash; failure-mechanism HYPOTHESIS "
                       "(for the FA engineer; not a diagnosis)", h))
    if x.hypotheses:
        hy = x.hypotheses[0]
        S.append(Paragraph(
            f"<b>{hy['hypothesis']}</b> &nbsp; [confidence: {hy['confidence']}]<br/>"
            f"<i>Because:</i> {hy['because']}<br/>"
            f"<i>Caveat:</i> {hy['caveat']}<br/>"
            f"<i>Suggested next test:</i> {hy['next_test']}", body))
    else:
        S.append(Paragraph("No mechanism signature matched; no hypothesis offered.", body))

    S.append(Spacer(1, 4))
    sign = [["Inspector", "", "Date", ""], ["Disposition", "", "Signature", ""],
            ["Notes", "", "", ""]]
    sg = Table(sign, colWidths=[22 * mm, 60 * mm, 20 * mm, 62 * mm], rowHeights=[6.5 * mm] * 3)
    sg.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
                            ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#555555")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    S.append(KeepTogether([Paragraph("SIGN-OFF", h), sg]))
    S.append(Paragraph(
        "Decision support only. Final rejection remains under approved engineering "
        "rules and QA control; this system detects, ranks, explains and forecasts. "
        "The model never overrides a hard engineering limit in either direction.",
        small))
    doc.build(S)
    if doc.page > 1:
        print(f"  WARNING: {out_pdf.name} runs to {doc.page} pages, not one")
    return out_pdf
