"""
Layer 3: physical failure-mechanism hypotheses.

EVERY OUTPUT OF THIS MODULE IS A HYPOTHESIS FOR A FAILURE-ANALYSIS ENGINEER,
NOT A DIAGNOSIS. The system does not replace the FA engineer; it hands them a
prioritised, pre-argued case file. Nothing here is asserted as fact and the
report text says so on the page.

CONFIDENCE LABELS, because these mappings are not equally well founded:

  well-founded   the signature follows from device physics that is not in
                 serious dispute, and the discriminator is a sign or a
                 direction rather than a magnitude
  plausible      a standard textbook association, but several mechanisms
                 produce the same signature and this one is not uniquely
                 implied
  speculative    consistent with the data, but a reliability engineer could
                 reasonably propose two or three alternatives with equal
                 standing

Only the fixture-versus-process discriminator is marked well-founded, and it is
the one worth defending in a Q&A.
"""

from __future__ import annotations

MECHANISMS = {
    "ACCELERATING_LEAKAGE": {
        "signature": "leakage rising with an accelerating (convex) trajectory",
        "hypothesis": "gate-oxide or dielectric degradation; possible TDDB precursor",
        "confidence": "plausible",
        "caveat": ("Accelerating leakage is consistent with TDDB but also with "
                   "ionic contamination and with junction damage. The trajectory "
                   "shape alone does not separate them; a temperature-step or "
                   "HTOL follow-up would."),
        "next_test": "extended HTOL with interim Iddq, then decap and emission microscopy",
    },
    "DISCRETE_STEP": {
        "signature": "a step change between two adjacent checkpoints, flat either side",
        "hypothesis": ("intermittent resistive short, marginal wire bond, or "
                       "latent ESD damage"),
        "confidence": "plausible",
        "caveat": ("A step is strong evidence of a discrete physical event rather "
                   "than a wear-out process, but it does not identify which. It "
                   "is also the signature a tester glitch produces, so the "
                   "measurement should be repeated before any FA spend."),
        "next_test": "repeat measurement; if it persists, curve-trace the affected pin",
    },
    "DELAY_DRIFT_STABLE_CURRENT": {
        "signature": "propagation delay drifting while supply and leakage stay flat",
        "hypothesis": "NBTI or hot-carrier injection",
        "confidence": "plausible",
        "caveat": ("NBTI and HCI both slow a part without moving quiescent "
                   "current much. Separating them needs a recovery test: NBTI "
                   "partially recovers when the stress is removed, HCI does not."),
        "next_test": "unbiased bake and re-measure; NBTI should partially recover",
    },
    "JOINT_INCONSISTENCY": {
        "signature": ("every individual parameter within its normal range, but "
                      "the combination is one the process does not produce"),
        "hypothesis": ("localised defect not yet affecting timing paths; the "
                       "class that escapes univariate screening"),
        "confidence": "speculative",
        "caveat": ("This is the weakest mapping in the table and should be "
                   "labelled as such. A joint-distribution outlier says the part "
                   "does not resemble its peers; it does NOT say why. The value "
                   "is in flagging the part for attention, not in naming the "
                   "mechanism."),
        "next_test": "quarantine and prioritise for FA; treat the mechanism as unknown",
    },
    "CORRELATION_INVERSION": {
        "signature": "high leakage together with SLOW timing",
        "hypothesis": ("a defect-driven leakage path rather than a fast process "
                       "corner"),
        "confidence": "well-founded",
        "caveat": ("This is the one discriminator here that rests on a sign "
                   "rather than a magnitude. Process variation moves threshold "
                   "voltage, so a leaky part from the fast corner is also FAST. "
                   "A part that is leaky AND slow is not on the process locus at "
                   "all, so its leakage has some other origin. The direction of "
                   "the correlation shift is the evidence."),
        "next_test": "compare against the lot's leakage-vs-delay locus; then FA",
    },
    "FIXTURE_THERMAL": {
        "signature": "anomalies clustered by socket or board position",
        "hypothesis": "chamber thermal gradient or socket contact degradation",
        "confidence": "well-founded",
        "caveat": ("Same sign argument, used the other way. HEAT raises leakage "
                   "and SLOWS the part; the process corner makes leaky parts "
                   "FAST. So a fixture artifact and a process outlier push the "
                   "leakage/delay correlation in OPPOSITE directions and are "
                   "separable by the sign of the shift, before any spatial "
                   "evidence is considered. The board clustering is then "
                   "confirmatory rather than the sole basis."),
        "next_test": ("map the chamber with a calibrated thermocouple board; "
                      "re-test the affected positions in a different socket"),
    },
    "BENIGN_HIGH_CORNER": {
        "signature": "elevated level, self-consistent across parameters, minimal drift",
        "hypothesis": "an extreme but legitimate process corner, not a defect",
        "confidence": "plausible",
        "caveat": ("A part can sit high on leakage and still be entirely healthy "
                   "if it is high in the way the process makes parts high: leaky "
                   "AND fast, moving along the corner locus. Rejecting these is "
                   "pure yield loss."),
        "next_test": "none; release unless other evidence appears",
    },
    "STATIC_LEVEL_OUTLIER": {
        "signature": ("level far outside the lot distribution but inside the "
                      "datasheet limit, with normal drift"),
        "hypothesis": ("a defect present at incoming rather than one developing "
                       "during burn-in"),
        "confidence": "plausible",
        "caveat": ("This is the problem statement's own worked example. Stable "
                   "does not mean safe: the part is already an outlier at t=0, "
                   "and burn-in will not reveal more about it. The right action "
                   "is disposition on the level, not further stress."),
        "next_test": "compare against incoming inspection data; FA if available",
    },
}


def hypothesise(signals: dict) -> list[dict]:
    """Rank mechanism hypotheses from the observed signature.

    `signals` carries booleans and magnitudes already computed elsewhere; this
    function does no analysis of its own, it only maps a signature to physics.
    """
    out = []

    def add(key, weight, because):
        m = dict(MECHANISMS[key])
        m["mechanism_key"] = key
        m["weight"] = weight
        m["because"] = because
        out.append(m)

    if signals.get("spatial_clustered"):
        add("FIXTURE_THERMAL", 1.0,
            f"{signals.get('n_flagged_on_board', 0)} flagged components share "
            f"board {signals.get('board_id', '?')}, and the leakage/delay shift "
            "has the thermal sign (leaky and slow) rather than the process sign")
    if signals.get("step_detected"):
        add("DISCRETE_STEP", 0.9,
            f"largest single-interval jump is {signals.get('max_jump_z', 0):.1f} "
            "robust sigma, with the neighbouring intervals normal")
    if signals.get("correlation_inverted"):
        add("CORRELATION_INVERSION", 0.85,
            "leakage is elevated while timing is slow, which is off the lot's "
            "leakage-vs-delay locus")
    if signals.get("accelerating_leakage"):
        add("ACCELERATING_LEAKAGE", 0.8,
            f"leakage drift rate rising across intervals (curvature "
            f"{signals.get('curvature_z', 0):+.1f} sigma)")
    if signals.get("delay_drift_stable_current"):
        add("DELAY_DRIFT_STABLE_CURRENT", 0.7,
            "propagation delay is drifting while quiescent current is flat")
    if signals.get("joint_only"):
        add("JOINT_INCONSISTENCY", 0.6,
            f"every parameter is within its normal range, but the joint distance "
            f"is D^2={signals.get('maha_d2', 0):.1f} against a good-part 99th "
            f"percentile of {signals.get('maha_p99', 0):.1f}")
    if signals.get("static_level_outlier"):
        add("STATIC_LEVEL_OUTLIER", 0.8,
            f"level is {signals.get('level_sigma', 0):.1f} robust sigma above the "
            "lot median at 0 h and the drift rate is unremarkable")
    if signals.get("benign_corner"):
        add("BENIGN_HIGH_CORNER", 0.5,
            "elevated but self-consistent across parameters, with minimal drift")
    return sorted(out, key=lambda m: -m["weight"])
