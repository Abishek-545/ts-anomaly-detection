"""
Generate reports/improvement_backlog.pdf

A literature-grounded backlog of improvements for the CNN anomaly detector,
derived from recent (2024-2026) research on reconstruction-based and
contamination-robust time-series anomaly detection. Each item is mapped to the
project's actual code and rated by effort / payoff / risk.

NOTE: recommendations are derived from paper abstracts and method summaries,
not full re-implementation of each paper. Treat "expected payoff" as a
hypothesis to test, not a promise.

Run:  python scripts/generate_improvement_backlog.py
"""

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = "reports/improvement_backlog.pdf"
NAVY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#2E6DA4")
TEAL = colors.HexColor("#2A8C82")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EAF0F6")
GOLD = colors.HexColor("#E8EEDB")


def S():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("TitleBig", parent=s["Title"], fontSize=22, textColor=NAVY, spaceAfter=4))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontSize=10, textColor=GREY, spaceAfter=8))
    s.add(ParagraphStyle("H1", parent=s["Heading1"], fontSize=15, textColor=NAVY, spaceBefore=14, spaceAfter=5))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=12, textColor=ACCENT, spaceBefore=10, spaceAfter=3))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=9.7, leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
    s.add(ParagraphStyle("Cap", parent=s["Normal"], fontSize=8.3, textColor=GREY, alignment=TA_LEFT, spaceAfter=10))
    s.add(ParagraphStyle("Key", parent=s["Normal"], fontSize=9.7, leading=14, textColor=NAVY,
                         backColor=LIGHT, borderPadding=6, spaceBefore=4, spaceAfter=10))
    s.add(ParagraphStyle("Cell", parent=s["Normal"], fontSize=8.2, leading=10.5))
    s.add(ParagraphStyle("CellH", parent=s["Normal"], fontSize=8.4, leading=10.5, textColor=colors.white,
                         fontName="Helvetica-Bold"))
    return s


def bullets(items, st, style="Body"):
    return ListFlowable([ListItem(Paragraph(t, st[style]), leftIndent=10) for t in items],
                        bulletType="bullet", start="•", leftIndent=12, spaceAfter=6)


def P(txt, st):
    return Paragraph(txt, st["Cell"])


def PH(txt, st):
    return Paragraph(txt, st["CellH"])


def grid(data, widths):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C6D6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build():
    st = S()
    doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=1.7 * cm, rightMargin=1.7 * cm, title="Improvement Backlog")
    e = []
    e.append(Paragraph("Improvement Backlog, Grounded in Recent Research", st["TitleBig"]))
    e.append(Paragraph("Reconstruction-based multivariate time-series anomaly detection. Baseline: "
                       "validation F1 ≈ 0.60, portal F1 ≈ 0.63, ROC-AUC ≈ 0.76. Literature window 2024–2026.",
                       st["Sub"]))
    e.append(Paragraph("Recommendations are derived from paper abstracts and method summaries and mapped to this "
                       "project's code. 'Payoff' is a hypothesis to test, not a promise — benchmarks (TSB-AD, "
                       "TAB) show no method wins universally.", st["Body"]))

    # ---- Context ----
    e.append(Paragraph("1.  Three framing facts from the literature", st["H1"]))
    e.append(bullets([
        "<b>Reconstruction-based (your family) is still competitive</b>, but effectiveness depends on anomaly "
        "type; your anomalies are slow drifts. A fancier model is not guaranteed to help. [Surveys 2024–25]",
        "<b>Contaminated training data is a first-class research problem in 2024–26</b> — exactly your #1 "
        "bottleneck ('train is not anomaly-free'). This is where the literature helps you most. [CLEANet, RiAD, "
        "TSAD-C, Encode-then-Decompose]",
        "<b>Much published F1 (0.9+) is inflated by point-adjustment</b>; one study showed random noise can beat "
        "SOTA under it. Your un-adjusted 0.63 is not comparable to those numbers — and it means you should verify "
        "your portal's exact metric. [Towards Unbiased Evaluation 2024; PATE 2024]",
    ], st))

    # ---- Backlog A: contamination ----
    e.append(Paragraph("2.  Backlog A — Contamination-robust training (highest leverage)", st["H1"]))
    e.append(Paragraph("Your teammate's <font face='Courier'>robust_downweight</font> in "
                       "<font face='Courier'>train.py</font> is a crude version of this whole family. The papers "
                       "give principled upgrades.", st["Body"]))
    e.append(grid([
        [PH("#", st), PH("Improvement", st), PH("Source", st), PH("Maps to your code", st),
         PH("Effort", st), PH("Payoff", st)],
        [P("A1", st), P("<b>kNN local-density loss weighting.</b> Weight each window's reconstruction loss by a "
                        "contamination score = similarity to its k nearest neighbours; sigmoid-down-weight "
                        "outliers.", st),
         P("CLEANet (AWRL), 2025", st), P("Replace the top-percentile rule in train.py with a kNN-density weight",
                                          st), P("Med", st), P("High", st)],
        [P("A2", st), P("<b>Dynamic (iterative) re-weighting.</b> Recompute sample weights across epochs so "
                        "persistently high-loss (likely-anomalous) windows fade out of training.", st),
         P("RiAD, 2024", st), P("Wrap the training loop in train.py with an epoch-wise weight update", st),
         P("Med", st), P("High", st)],
        [P("A3", st), P("<b>Masking-based decontamination.</b> Randomly mask parts of each input and reconstruct "
                        "them; because normals dominate, masking dilutes anomaly influence.", st),
         P("TSAD-C, 2024 (+6.3% F1)", st), P("New masking augmentation in SensorDataset / train step", st),
         P("Med", st), P("Med-High", st)],
        [P("A4", st), P("<b>Encode-then-decompose.</b> Split the latent into a stable (normal) and auxiliary "
                        "(anomaly-absorbing) part; score with mutual information, not raw error.", st),
         P("Encode-then-Decompose, 2025", st), P("Architectural change to ConvAutoencoder1D + new score", st),
         P("High", st), P("Med", st)],
    ], [0.7 * cm, 5.5 * cm, 3.0 * cm, 4.4 * cm, 1.1 * cm, 1.5 * cm]))

    # ---- Backlog B: architecture ----
    e.append(Paragraph("3.  Backlog B — Architecture & representation", st["H1"]))
    e.append(grid([
        [PH("#", st), PH("Improvement", st), PH("Source", st), PH("Maps to your code", st),
         PH("Effort", st), PH("Payoff", st)],
        [P("B1", st), P("<b>Multi-scale convolutions.</b> Add parallel branches mixing random + dilated convs to "
                        "capture features at several temporal scales.", st),
         P("Hybrid conv-AE for robotics, 2024", st), P("Extend DilatedResidualBlock stack in model.py", st),
         P("Med", st), P("Med", st)],
        [P("B2", st), P("<b>Two-stream temporal + feature encoder.</b> One encoder over time, one over the "
                        "transposed input (cross-sensor correlations). Lightweight (CLEANet: -90% params vs "
                        "transformer).", st),
         P("CLEANet, 2025", st), P("Second encoder branch in ConvAutoencoder1D", st), P("Med", st), P("Med", st)],
        [P("B3", st), P("<b>Association-discrepancy scoring.</b> Score by how a point's learned attention/"
                        "association pattern differs from normal — complements reconstruction error.", st),
         P("Anomaly-Transformer / VAE+AssocDisc, 2025", st), P("New scoring head; larger change than Mahalanobis",
                                                               st), P("High", st), P("Uncertain", st)],
        [P("B4", st), P("<b>Explicit inter-sensor graph.</b> Learn a dynamic adjacency (self-attention) over the "
                        "18 sensors to model their relationships (time-then-graph).", st),
         P("TSAD-C, 2024", st), P("Graph module before/after the encoder", st), P("High", st), P("Uncertain", st)],
        [P("B5", st), P("<b>Variational (VAE) bottleneck.</b> Probabilistic latent; surveys report VAEs do well "
                        "on pattern-wise anomalies.", st),
         P("Surveys 2024–25; math13071209", st), P("Swap bottleneck + add KL term in train.py", st),
         P("High", st), P("Uncertain", st)],
    ], [0.7 * cm, 5.5 * cm, 3.4 * cm, 4.0 * cm, 1.1 * cm, 1.5 * cm]))

    # ---- Backlog C: scoring ----
    e.append(Paragraph("4.  Backlog C — Scoring (cheap, aligned with what you already do)", st["H1"]))
    e.append(grid([
        [PH("#", st), PH("Improvement", st), PH("Source / rationale", st), PH("Maps to your code", st),
         PH("Effort", st), PH("Payoff", st)],
        [P("C1", st), P("<b>Fit the Mahalanobis normal model on TRAIN</b> (abundant) rather than the few val "
                        "normal windows — better-conditioned covariance. Fit on lowest-error windows to dodge "
                        "contamination.", st), P("Covariance estimation best-practice", st),
         P("fit_mahalanobis_from_validation → train", st), P("Low", st), P("Med", st)],
        [P("C2", st), P("<b>Combine two scores</b> (e.g. masked-reconstruction + full-reconstruction, weighted).",
                        st), P("TSAD-C s = λ1·s1 + λ2·s2", st), P("Extend scoring in run_cnn.py", st),
         P("Low-Med", st), P("Med", st)],
        [P("C3", st), P("<b>Seed-ensemble scores.</b> Average per-timestep scores over 3–5 CNNs (different seeds) "
                        "— variance reduction typically lifts AUC.", st), P("Ensembling best-practice", st),
         P("New run_cnn_ensemble pipeline", st), P("Low", st), P("Med", st)],
    ], [0.7 * cm, 5.6 * cm, 3.4 * cm, 4.0 * cm, 1.1 * cm, 1.4 * cm]))

    # ---- Backlog D: evaluation/threshold ----
    e.append(Paragraph("5.  Backlog D — Evaluation & threshold (protect the F1 you have)", st["H1"]))
    e.append(grid([
        [PH("#", st), PH("Improvement", st), PH("Source", st), PH("Maps to your code", st),
         PH("Effort", st), PH("Payoff", st)],
        [P("D1", st), P("<b>Confirm the portal metric, then match the threshold objective to it.</b> You optimise "
                        "F0.5; if the portal grades F1, tune the threshold for F1.", st),
         P("your setup + eval papers", st), P("beta in find_best_threshold / config", st),
         P("Low", st), P("Med (free F1)", st)],
        [P("D2", st), P("<b>Report F1-BA / PATE internally</b> for honest self-comparison; do not compare your "
                        "0.63 to point-adjusted paper numbers.", st), P("Unbiased Eval 2024; PATE 2024", st),
         P("Add to metrics.py (reporting only)", st), P("Low", st), P("Rigor", st)],
        [P("D3", st), P("<b>Quantile-based threshold</b> as a robust alternative/cross-check to the PR-curve "
                        "sweep.", st), P("TSAD-C thresholding", st), P("Option in find_best_threshold", st),
         P("Low", st), P("Low-Med", st)],
    ], [0.7 * cm, 5.6 * cm, 3.2 * cm, 4.2 * cm, 1.1 * cm, 1.4 * cm]))

    # ---- Priority ----
    e.append(Paragraph("6.  Recommended sequence (effort-adjusted)", st["H1"]))
    e.append(Paragraph("Do the cheap, aligned, high-confidence items first; treat the big architectural bets as "
                       "later research, only if the ceiling is still binding.", st["Body"]))
    e.append(grid([
        [PH("Order", st), PH("Item", st), PH("Why first", st)],
        [P("0", st), P("Diagnostic plots (score-vs-label per run; normal/anomaly histogram)", st),
         P("See the failure mode before changing anything; confirms contamination is visible", st)],
        [P("1", st), P("D1 — confirm portal metric & align threshold", st),
         P("Possibly free F1, minutes of work, zero model risk", st)],
        [P("2", st), P("A1/A2 — kNN-density robust training (upgrade robust_downweight)", st),
         P("Highest-leverage lever; attacks the AUC ceiling at its source; code partly exists", st)],
        [P("3", st), P("C3 — seed-ensemble + C1 — Mahalanobis-on-train", st),
         P("Cheap, low-risk AUC lifts; also stabilises your noisy single-run F1", st)],
        [P("4", st), P("A3 — masking decontamination", st),
         P("Well-evidenced (+6.3% F1 in TSAD-C); moderate effort", st)],
        [P("5", st), P("B1/B2 — multi-scale / two-stream encoder", st),
         P("Architectural gains after the cheaper wins are exhausted", st)],
        [P("6", st), P("A4/B3/B4/B5 — decompose / attention / graph / VAE", st),
         P("Big research bets; high effort, uncertain payoff; last", st)],
    ], [1.1 * cm, 6.7 * cm, 6.9 * cm]))

    e.append(Paragraph("Every experiment must change ONE thing and be measured with run-grouped CV or seed "
                       "averaging — a single run varies by ±0.02 F1, so smaller 'gains' prove nothing.", st["Key"]))

    doc.build(e)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
