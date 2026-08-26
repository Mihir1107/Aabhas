# What changed between dataset v1.0 and v1.1

For the presentation team. Every figure that moved, with old value, new value
and delta. Machine-readable version: `changed_numbers.csv` (129 changed cells).

**Tags:** `dataset-v1.0` (commit `8ca44bc`) → `dataset-v1.1` (commit `1ac9e59`).

---

## Why the re-baseline happened

Type III (centre-hider) parts in v1.0 had a construction artifact. Each
checkpoint was placed at a deterministic lot quantile, and lot quantiles move
smoothly with time, so the parts inherited a smooth curve and carried **no
independent per-checkpoint measurement noise**. Their trajectories were 3–5×
smoother than a real part's, and a trivial "too smooth" rule recovered **45.8%
of Type III at 1% yield loss** using no multivariate reasoning at all.

Two root causes were found and fixed. The second was not visible until the
first was fixed:

1. **No measurement noise.** Type III now carries its own already-drawn `eps` —
   the same array, the same draw, that every good part uses — so the noise
   distribution is identical by construction. Nothing is resampled or
   truncated; the band is honoured by shrinking the *target* given the realised
   noise, never the noise itself.
2. **The anomaly direction flipped between checkpoints.** The box-optimal sign
   vertex was re-solved at every checkpoint from that checkpoint's own noisy
   empirical covariance, and it could flip: a part sat at the 80th percentile
   of `prop_delay_ns` at 96 h and the 20th at 168 h. That is an estimation
   artifact, not a latent defect, and once real noise was added it became the
   *dominant* source of roughness (Type III went from 3–5× too smooth to 1.7–2×
   too rough). The direction is now solved once against the average within-lot
   correlation and held fixed.

**One parameter changed:** `type3_band_pct` 30.0 → 33.0. Adding noise jitters
the realised percentile, so the design target must sit further inside the band.
33.0 was chosen because it *reproduces* v1.0's joint separation (median
Mahalanobis D² 15.9 vs 16.4; 98.2nd vs 98.0th percentile of the good
population) rather than improving on it. The band is now the 17th–83rd
percentile, still squarely centre-of-distribution.

Nothing else was touched: no other archetype, not the good-part model, not any
parameter that was already passing its assertions.

---

## 1. The smoothness leak is closed

| Measure | v1.0 | v1.1 |
|---|---|---|
| Type III trajectory roughness (good = 1.000) | 0.25 | **1.040** |
| Kolmogorov–Smirnov vs good parts | p ≈ 0 | **p = 0.896** |
| Mann–Whitney U vs good parts | — | p = 0.713 |
| "too smooth" rule, catch at 1% yield loss | **45.8%** | **1.7%** |
| "too rough" rule, catch at 1% yield loss | — | 0.0% |

Chance is 1.0% by construction. New assertion **A8** enforces this and is
two-sided. Validator: **23/23 assertions pass** (was 20/20).

---

## 2. THE NUMBER THAT MATTERS MOST — Type III per-rung catch collapsed

This is the headline change and it goes on a slide. Catch rate on Type III,
all lots, at 7% yield loss:

| Rung | v1.0 | v1.1 | Δ |
|---|---|---|---|
| L2 trajectory features | 53.3% | **0.0%** | **−53.3** |
| L3b PCA T² | 51.7% | 1.7% | −50.0 |
| L3c kNN distance | 51.7% | 1.7% | −50.0 |
| L4a Isolation Forest | 11.7% | 0.0% | −11.7 |
| L4b Autoencoder | 50.0% | 13.3% | −36.7 |
| L4b′ LOF | 51.7% | 3.3% | −48.3 |
| **L3a Mahalanobis + MCD** | 80.0% | **71.7%** | −8.3 |
| **L3b-Q PCA Q-residual** | 81.7% | **66.7%** | −15.0 |
| L4c Union ensemble | 71.7% | 48.3% | −23.3 |

**Read this as a strengthening, not a weakening.** In v1.0 roughly half of all
Type III detection was the smoothness artifact, including 53% credited to L2 —
a pure trajectory detector with no multivariate reasoning whatsoever. With the
artifact removed, **only the genuinely joint detectors still find
centre-hiders**: Mahalanobis+MCD at 71.7% and the PCA Q-residual at 66.7%,
while every univariate and trajectory rung drops to 0–3%.

That is exactly the claim the dataset exists to support, and in v1.0 it was not
cleanly true. The deck can now say "the centre-hider is invisible to every
layer except the multivariate one" and point at L2 = 0.0% as the proof.

---

## 3. Aggregate ablation — small moves

Recall at 93% yield, test lots. Deltas are ≤ 1.2 points, which is 5 parts of
427.

| Method | v1.0 | v1.1 | Δ recall | Δ PR-AUC |
|---|---|---|---|---|
| L1 DPAT dynamic, MAD | 54.33% | 54.33% | 0.00 | +0.000 |
| L2 trajectory | 80.33% | 79.16% | −1.17 | −0.010 |
| L3a Mahalanobis + MCD | 65.34% | 64.87% | −0.47 | −0.002 |
| L3b PCA T² | 90.16% | 88.99% | −1.17 | — |
| L3b-Q PCA Q-residual | 81.73% | 81.26% | −0.47 | −0.008 |
| L3c kNN | 87.59% | 86.42% | −1.17 | — |
| L4a Isolation Forest | 67.68% | 68.15% | +0.47 | +0.003 |
| L4b Autoencoder | 90.63% | 89.46% | −1.17 | −0.027 |
| L4b′ LOF | 91.10% | 90.16% | −0.94 | −0.011 |
| L4c Union ensemble | 93.68% | 92.97% | −0.70 | −0.003 |

**Every headline recall figure moved down slightly.** That is the honest
direction: v1.0 was borrowing a few points from the artifact. L1 is unchanged
because DPAT never used the leak.

Unchanged: L0 catches 0.0% of every injected type; contamination 1.75%;
lot-to-lot ICC; Arrhenius AF = 77.66; all Type I / II / IV / Va / Vb / VI / VII
figures.

---

## 4. Numbers that are new in v1.1

### Cumulative ablation (each rung contains everything below it)

| Rung | Contents | Recall @93% yield | PR-AUC | Escape rate | Cost |
|---|---|---|---|---|---|
| C0 | L0 alone | n/a | n/a | 100.00% | 427,000 |
| C1 | L0 + L1 | 54.33% | 0.264 | 45.67% | 196,651 |
| C2 | + L2 | 79.86% | 0.711 | 20.14% | 87,651 |
| C3 | + L3 | **92.74%** | 0.800 | 7.26% | 32,651 |
| C4 | + L4 | 92.27% | 0.805 | 7.73% | 34,651 |

**The cumulative table is NOT monotone: C4 sits 0.47 points below C3.** That is
2 defective parts out of 427, well inside sampling noise on n=427, so the
honest reading is that **the ladder plateaus at L3 rather than that L4 hurts**.
PR-AUC does improve slightly (0.800 → 0.805) and AUROC is flat. Diagnosis: the
rungs are fused by max robust-z, so every added member also raises the maximum
score of good parts; at a fixed yield-loss budget the threshold rises, and a
member whose unique catches do not exceed its own tail false positives is a
wash. This is consistent with the per-tier finding that L4 does not beat L3.

### Ensemble fusion: your diagnosis was half right

| Fusion | Recall @93% yield | PR-AUC | AUROC |
|---|---|---|---|
| binary OR (decision level) | 90.16% | n/a | n/a |
| max of percentile ranks | 92.97% | 0.709 | 0.980 |
| mean of percentile ranks | 82.90% | 0.577 | 0.938 |
| **max robust-z** | 92.27% | **0.794** | 0.982 |
| weighted mean robust-z | 90.40% | 0.796 | 0.966 |
| *best single member (LOF)* | *90.16%* | ***0.882*** | *0.957* |

**Granularity loss was real but it is not the main effect.** Moving from
rank-space to z-space recovers +0.085 of PR-AUC (0.709 → 0.794), which is the
part of your diagnosis that holds. But **score fusion does not recover PR-AUC
to the single-member level** — 0.794 against LOF's 0.882.

The real cause was found by inspection: of the 20 false positives in the top
200 fused scores, **all 20 come from `L3b_Q` alone**. Under a max rule the
combined ranking inherits *every* member's tail false positives, and no amount
of preserved granularity removes them.

Confirming this: greedy forward member selection maximising PR-AUC **on the
validation lots** selects **LOF alone** — adding any second member reduces
validation PR-AUC. Selecting members on the test set instead would have picked
a 3-member set scoring 0.864, which is exactly the post-hoc selection bias
worth avoiding.

**Practical consequence for the deck:** fusion and single-detector are
different products. Use fusion when you want maximum recall at a fixed
operating point (92.3–93.0% vs 90.2%). Use LOF or the autoencoder alone when
you want a ranked worklist for an inspector, because PR-AUC is what governs the
quality of a ranking.

---

## 5. Type VI test-set coverage — NOT fixed, and why

Requested: move the split so at least two trap lots land in test. **Not
achievable.** The six Type VI lots are `LOT001, 040, 049, 054, 084, 146` —
five of the six fall in the first 85 lots. Reaching two in test needs the test
set to start before LOT084, i.e. a 35/20/45 split; reaching even one (LOT146)
needs 40/20/40, which halves the training data and puts more defective parts in
test (880) than in train (824).

Per your instruction the split is unchanged and every Type VI figure is
labelled all-lots. **The static-vs-dynamic PAT result stands as an all-lots
number.** One mitigation is available and worth using: LOT146 falls in the
**validation** block, so Type VI has 500 parts in a non-training holdout, which
is a clean evaluation for the fitting-free L1 detectors.

Split boundaries unchanged, 60/20/20 chronological:

| Split | Lots | Parts | Defective | I | II | III | IV | Vb | Va | VI | VII |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 144 (LOT000–143) | 72,000 | 1,220 | 415 | 343 | 37 | 251 | 174 | 1,081 | 2,500 | 128 |
| val | 48 (LOT144–191) | 24,000 | 453 | 150 | 133 | 15 | 96 | 59 | 359 | 500 | 384 |
| test | 48 (LOT192–239) | 24,000 | 427 | 155 | 124 | 8 | 73 | 67 | 360 | **0** | 128 |

Also in `split_counts.csv`.

---

## 6. What the presentation team must change

1. Any Type III per-rung catch rate — see section 2. The **L2 = 0.0%** figure
   is now a strong claim and should be on the slide.
2. Any aggregate recall figure — subtract roughly 1 point, see section 3.
3. The union-ensemble PR-AUC story — see section 4; fusion does not recover it,
   and the honest framing is "two different products".
4. Any Type VI figure must be labelled **all lots**, not test.
5. If a slide says the ablation climbs monotonically through L4, it should say
   it plateaus at L3.
