"""
Sensitivity of Type III separability to the leakage-family correlation.

corr(iddq, leakage) = 0.93 was originally chosen because it was the value that
made the centre-hider construction work, which is reverse-engineering from the
requirement rather than from physics. This sweep reports what actually happens
at weaker, more conservative correlations, so the assumption can be defended or
abandoned on evidence.

Run:  python3 sweep_correlation.py
"""
from __future__ import annotations

import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet

TARGETS = [0.70, 0.80, 0.85, 0.93]
LOTS, PARTS = 60, 500          # smaller than the shipped set; this is a sweep


def maha_by_part(meas: pd.DataFrame, params: list[str]) -> pd.Series:
    m = meas[meas["measurement_status"] == "MEASURED"]
    acc: dict[str, list[float]] = {}
    for _, sub in m.groupby(["lot_id", "checkpoint_h"]):
        X = sub[params].to_numpy()
        if len(X) < 30:
            continue
        d2 = MinCovDet(support_fraction=0.9, random_state=0).fit(X).mahalanobis(X)
        for cid, v in zip(sub["component_id"].to_numpy(), d2):
            acc.setdefault(cid, []).append(float(v))
    return pd.Series({k: float(np.median(v)) for k, v in acc.items()})


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for r in TARGETS:
            out = Path(td) / f"r{int(r * 100)}"
            subprocess.run([sys.executable, "generate_data.py", "--leakage-corr", str(r),
                            "--lots", str(LOTS), "--parts-per-lot", str(PARTS),
                            "--out", str(out)], check=True, capture_output=True)
            meas = pd.read_csv(out / "burnin_measurements.csv")
            gt = pd.read_csv(out / "ground_truth.csv").set_index("component_id")
            import json
            cfg = json.loads((out / "config.json").read_text())
            params = [q["name"] for q in cfg["parameters"]]

            # box-constrained ceiling from the empirical within-lot correlation
            z = meas[(meas["checkpoint_h"] == 0.0)
                     & (meas["measurement_status"] == "MEASURED")]
            R = np.mean([np.corrcoef(s[params].dropna().to_numpy(), rowvar=False)
                         for _, s in z.groupby("lot_id")], axis=0)
            inv = np.linalg.inv(R)
            c = stats.norm.ppf(0.5 + cfg["config"]["type3_band_pct"] / 100.0)
            ceiling = max(float(np.array(sv) @ inv @ np.array(sv))
                          for sv in itertools.product([-1.0, 1.0], repeat=len(params))) * c * c

            d2 = maha_by_part(meas, params)
            ty = gt["defect_type"].reindex(d2.index)
            good = d2[ty == "GOOD"]
            t3 = d2[ty == "III_CENTRE_HIDER"]
            sev = gt["severity"].reindex(d2.index)
            row = {
                "target_r": r,
                "empirical_r": float(R[0][1]),
                "box_ceiling_D2": ceiling,
                "good_p99_D2": float(good.quantile(0.99)),
                "typeIII_median_D2": float(t3.median()) if len(t3) else np.nan,
                "n_typeIII": len(t3),
                "catch@1%YL_%": (100 * float((t3 >= good.quantile(0.99)).mean())
                                 if len(t3) else np.nan),
                "AUROC": (float(stats.mannwhitneyu(t3, good, alternative="greater")
                                .statistic / (len(t3) * len(good))) if len(t3) else np.nan),
            }
            # The band and the correlation trade off directly: ceiling scales
            # with c(band)^2. So rather than just reporting "it collapses at
            # r<0.93", report the exchange rate -- the band you would need at
            # this correlation for a centre-hider to clear the good
            # population's upper tail with a 1.5x margin.
            best = ceiling / (c * c)
            need_c = np.sqrt(1.5 * row["good_p99_D2"] / best)
            row["min_band_pp"] = (float((stats.norm.cdf(need_c) - 0.5) * 100)
                                  if need_c < 4 else np.nan)
            for tier in ("severe", "moderate", "mild"):
                sel = d2[(ty == "III_CENTRE_HIDER") & (sev == tier)]
                row[f"catch_{tier}_%"] = (100 * float((sel >= good.quantile(0.99)).mean())
                                          if len(sel) else np.nan)
            rows.append(row)
            print(f"  done r={r}", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("TYPE III SEPARABILITY vs corr(iddq, leakage)")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nbox_ceiling_D2 is the most anomalous a part can be while staying inside")
    print("the configured univariate band. Compare it against good_p99_D2: when the")
    print("ceiling drops toward the good population's own upper tail, the centre-hider")
    print("stops being constructible at all, whatever the detector.")
    df.to_csv("data/correlation_sweep.csv", index=False)
    print("\nwritten to data/correlation_sweep.csv")


if __name__ == "__main__":
    main()
