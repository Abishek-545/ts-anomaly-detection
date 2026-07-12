"""
Generate reports/cnn_deep_dive.pdf

A self-contained learning resource that explains the anomaly-detection system
end to end, with special focus on the CNN autoencoder and the current result
(validation F1 ~= 0.60, portal F1 ~= 0.63). Includes a methodology diagram and
a layer-by-layer CNN architecture diagram drawn from the actual code in
src/models/cnn/ and src/pipelines/run_cnn.py.

Run:  python scripts/generate_cnn_deep_dive.py
"""

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = "reports/cnn_deep_dive.pdf"

NAVY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#2E6DA4")
TEAL = colors.HexColor("#2A8C82")
GREEN = colors.HexColor("#3F7D3F")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EAF0F6")
LIGHTGREEN = colors.HexColor("#DCEBD6")
BAR_ENC = colors.HexColor("#3B6EA5")
BAR_BOTTLE = colors.HexColor("#C55A11")
BAR_DEC = colors.HexColor("#5B9BD5")
BAR_IO = colors.HexColor("#7F7F7F")


def S():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("TitleBig", parent=s["Title"], fontSize=23, textColor=NAVY, spaceAfter=4))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontSize=10.5, textColor=GREY, spaceAfter=6))
    s.add(ParagraphStyle("H1", parent=s["Heading1"], fontSize=16, textColor=NAVY, spaceBefore=16, spaceAfter=6))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=13, textColor=ACCENT, spaceBefore=12, spaceAfter=4))
    s.add(ParagraphStyle("H3", parent=s["Heading3"], fontSize=11, textColor=TEAL, spaceBefore=8, spaceAfter=2))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=7))
    s.add(ParagraphStyle("BodyL", parent=s["Normal"], fontSize=10, leading=15, alignment=TA_LEFT, spaceAfter=7))
    s.add(ParagraphStyle("Cap", parent=s["Normal"], fontSize=8.5, textColor=GREY, alignment=TA_LEFT, spaceBefore=2, spaceAfter=12))
    s.add(ParagraphStyle("Key", parent=s["Normal"], fontSize=10, leading=15, textColor=NAVY,
                         backColor=LIGHT, borderPadding=6, spaceBefore=4, spaceAfter=12))
    return s


def bullets(items, st, style="Body"):
    return ListFlowable(
        [ListItem(Paragraph(t, st[style]), leftIndent=10) for t in items],
        bulletType="bullet", start="•", leftIndent=12, spaceAfter=8,
    )


def num_list(items, st, style="Body"):
    return ListFlowable(
        [ListItem(Paragraph(t, st[style]), leftIndent=10) for t in items],
        bulletType="1", leftIndent=14, spaceAfter=8,
    )


def table(rows, header, widths):
    data = [header] + rows
    t = Table(data, hAlign="LEFT", colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C6D6")),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _arrow_down(d, cx, y_top, length=12):
    y2 = y_top - length
    d.add(Line(cx, y_top, cx, y2 + 3, strokeColor=GREY, strokeWidth=1))
    d.add(Polygon([cx - 3, y2 + 4, cx + 3, y2 + 4, cx, y2 - 1], fillColor=GREY, strokeColor=GREY))


def methodology_diagram():
    W, H = 470, 486
    d = Drawing(W, H)
    steps = [
        ("Raw sensor runs  —  28 train / 10 val / 53 test CSVs, 18 sensors", BAR_IO, colors.white),
        ("StandardScaler  (fit on TRAIN only, applied to all splits)", GREY, colors.white),
        ("Per-run sliding windows  (200 x 18, step 10; never cross run boundaries)", GREY, colors.white),
        ("Dilated-TCN CNN autoencoder  (trained to reconstruct NORMAL windows)", NAVY, colors.white),
        ("Per-sensor reconstruction error  (MSE over time, one value per sensor)", ACCENT, colors.white),
        ("Mahalanobis distance  (Ledoit-Wolf normal model → one score / window)", ACCENT, colors.white),
        ("Dense per-timestep scoring  (step 1, average overlapping windows)", TEAL, colors.white),
        ("F-beta=0.5 threshold  (chosen on validation labels)", TEAL, colors.white),
        ("Per-timestep 0/1  →  submission (run_id, timestep, prediction)", GREEN, colors.white),
    ]
    box_h, gap, bw = 34, 18, 430
    x = (W - bw) / 2
    y = H - box_h
    for i, (txt, fill, tc) in enumerate(steps):
        d.add(Rect(x, y, bw, box_h, fillColor=fill, strokeColor=colors.white, strokeWidth=0.5, rx=4, ry=4))
        d.add(String(x + bw / 2, y + box_h / 2 - 3, txt, textAnchor="middle", fontSize=8.2, fillColor=tc))
        if i < len(steps) - 1:
            _arrow_down(d, W / 2, y, gap)
        y -= (box_h + gap)
    return d


def cnn_arch_diagram():
    """Vertical hourglass: bar width proportional to the time dimension (200 or
    100), colour groups encoder / bottleneck / decoder."""
    W, H = 470, 470
    d = Drawing(W, H)
    # (op label, C x T label, time_dim, colour)
    layers = [
        ("Input", "18 x 200", 200, BAR_IO),
        ("DilatedResidualBlock  d=1", "32 x 200", 200, BAR_ENC),
        ("DilatedResidualBlock  d=2", "32 x 200", 200, BAR_ENC),
        ("DilatedResidualBlock  d=4", "32 x 200", 200, BAR_ENC),
        ("DilatedResidualBlock  d=8", "64 x 200", 200, BAR_ENC),
        ("MaxPool1d(2)  — bottleneck", "64 x 100", 100, BAR_BOTTLE),
        ("ConvTranspose1d  stride 2", "32 x 200", 200, BAR_DEC),
        ("DilatedResidualBlock  d=4", "32 x 200", 200, BAR_DEC),
        ("DilatedResidualBlock  d=2", "32 x 200", 200, BAR_DEC),
        ("Conv1d k=3  → reconstruction", "18 x 200", 200, BAR_IO),
    ]
    cx = 268
    max_w = 250
    bh, gap = 26, 16
    y = H - bh
    for i, (op, shape, tdim, col) in enumerate(layers):
        bw = max_w * (tdim / 200.0)
        d.add(Rect(cx - bw / 2, y, bw, bh, fillColor=col, strokeColor=colors.white, strokeWidth=0.6, rx=3, ry=3))
        d.add(String(cx, y + bh / 2 - 3, shape, textAnchor="middle", fontSize=8, fillColor=colors.white))
        d.add(String(cx - max_w / 2 - 8, y + bh / 2 - 3, op, textAnchor="end", fontSize=7.6, fillColor=colors.HexColor("#333333")))
        if i < len(layers) - 1:
            _arrow_down(d, cx, y, gap)
        y -= (bh + gap)
    # group brackets on the right
    d.add(String(cx + max_w / 2 + 12, H - 5 * (bh + gap) + 30, "ENCODER", textAnchor="start", fontSize=8, fillColor=BAR_ENC))
    d.add(String(cx + max_w / 2 + 12, H - 9 * (bh + gap) + 20, "DECODER", textAnchor="start", fontSize=8, fillColor=BAR_DEC))
    return d


def build():
    st = S()
    doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=2 * cm, rightMargin=2 * cm, title="CNN Autoencoder: Deep Dive")
    e = []

    # ============ TITLE ============
    e.append(Paragraph("The CNN Anomaly Detector, End to End", st["TitleBig"]))
    e.append(Paragraph("A learning resource for multivariate time-series anomaly detection on "
                       "batch-distillation sensor data. Current result: validation F1 ≈ 0.60, "
                       "competition-portal F1 ≈ 0.63.", st["Sub"]))
    e.append(Paragraph("This document explains how the system works so you can reason about its "
                       "behaviour and improvements yourself — not just raise a number. Every claim is "
                       "tied to the actual code in <font face='Courier'>src/</font>.", st["Body"]))

    # ============ 1. PROBLEM & DATA ============
    e.append(Paragraph("1.  The problem and the data", st["H1"]))
    e.append(Paragraph(
        "You are detecting anomalies in an industrial batch-distillation process instrumented with 18 "
        "sensors (temperatures, pressures, flows, levels — tags like T701, PDI702, FT703). The defining "
        "constraint is that <b>the training data has no anomaly labels.</b> You get 28 unlabeled training "
        "runs, 10 labeled validation runs (per-timestep 0/1 labels), and 53 unlabeled test runs scored by "
        "an external portal.", st["Body"]))
    e.append(Paragraph(
        "That constraint forces the entire design. You cannot train a classifier that learns 'anomaly vs "
        "normal', because you have no anomaly examples to learn from. So you take the "
        "<b>reconstruction / novelty</b> route: teach a model what NORMAL looks like, then flag whatever "
        "departs from it. This is why an <i>autoencoder</i> is the natural choice — it is trained only to "
        "copy normal data through a bottleneck, and it reconstructs unfamiliar (anomalous) patterns badly. "
        "The reconstruction error becomes the anomaly score.", st["Key"]))

    # ============ 2. METHODOLOGY DIAGRAM ============
    e.append(Paragraph("2.  The pipeline at a glance", st["H1"]))
    e.append(Paragraph("Every one of the three pipelines (<font face='Courier'>run_cnn.py</font>, "
                       "<font face='Courier'>run_hybrid.py</font>, <font face='Courier'>run_final_hybrid.py</font>) "
                       "follows the same spine. The CNN pipeline is the model of record:", st["Body"]))
    e.append(methodology_diagram())
    e.append(Paragraph("Figure 1. End-to-end methodology. Grey = data handling, navy = the model, blue = "
                       "scoring, green = output. The validation labels are used at two points only: to pick "
                       "the best training checkpoint and to choose the threshold — never to train the network.",
                       st["Cap"]))

    # ============ 3. STEP BY STEP ============
    e.append(Paragraph("3.  The pipeline, step by step (with the 'why' and the 'what if')", st["H1"]))

    e.append(Paragraph("3.1  Ingestion and per-run loading", st["H2"]))
    e.append(Paragraph(
        "<font face='Courier'>load_split_per_file</font> loads each run as its own DataFrame and keeps them "
        "<i>separate</i>. This matters for the very next steps: runs are independent experiments, and a window "
        "must never span two of them.", st["Body"]))

    e.append(Paragraph("3.2  Scaling  —  <font face='Courier'>StandardScaler</font>, fit on train only", st["H2"]))
    e.append(Paragraph(
        "Sensors live on wildly different scales (a temperature in the hundreds, a pressure difference near "
        "zero). A neural network trained on raw values would let the large-magnitude sensors dominate the loss. "
        "StandardScaler converts every sensor to mean 0 / std 1. <b>Why fit on train only?</b> The scaler's "
        "mean and std are parameters learned from data; fitting them on validation or test would leak "
        "information about the evaluation set into preprocessing. <b>What if you skipped it?</b> The MSE loss "
        "would be dominated by a few high-variance sensors and the autoencoder would essentially ignore the "
        "rest — many real anomalies would become invisible in the score.", st["Body"]))

    e.append(Paragraph("3.3  Windowing  —  200 timesteps, per run", st["H2"]))
    e.append(Paragraph(
        "The CNN does not see one timestep at a time; it sees a <b>window</b> of 200 consecutive timesteps "
        "across all 18 sensors (<font face='Courier'>create_windows_per_file</font>, "
        "<font face='Courier'>WINDOW_SIZE=200</font>). Training uses stride "
        "<font face='Courier'>STEP=10</font> (windows every 10 steps — enough coverage without 20x "
        "redundancy); validation/test use <font face='Courier'>EVAL_STEP=1</font> for a dense, per-timestep "
        "score.", st["Body"]))
    e.append(bullets([
        "<b>Why 200?</b> The project's window-size tuning (notebook 04) found F1 kept rising as the window "
        "grew from 20 to 200 — the anomalies here are slow, sustained drifts, not sharp one-step spikes, so "
        "the model needs a wide temporal context to recognise 'this pattern is abnormal'.",
        "<b>Why per-run (never splicing runs together)?</b> An earlier bug concatenated all runs into one "
        "array before windowing, so a window could stitch the end of one run onto the start of an unrelated "
        "one — a fake transient the model wasted capacity 'explaining'. Fixing this was one of the six "
        "original bug fixes.",
        "<b>What if you made it 400?</b> More context, but two costs: fewer training windows, and short runs "
        "(one train run is only 743 timesteps) would yield very few or zero windows — you would silently "
        "drop data. 200 is a sweet spot given the run lengths.",
    ], st))

    e.append(Paragraph("3.4  The model  —  a dilated-TCN autoencoder", st["H2"]))
    e.append(Paragraph(
        "The network (<font face='Courier'>ConvAutoencoder1D</font>) is a 1-D convolutional autoencoder built "
        "from dilated residual blocks. Section 4 dissects it layer by layer. The one-line intuition: it "
        "<i>compresses</i> a 200-step window down through a bottleneck and then <i>rebuilds</i> it, and because "
        "it was only ever trained on normal windows, it can only rebuild normal-looking patterns well.", st["Body"]))

    e.append(Paragraph("3.5  Training  —  reconstruct, and keep the best checkpoint", st["H2"]))
    e.append(Paragraph(
        "Training minimises mean-squared reconstruction error (<font face='Courier'>nn.MSELoss</font>) with "
        "Adam (<font face='Courier'>lr=1e-3</font>) for 50 epochs. Crucially, every 5 epochs it scores the "
        "validation set and <b>keeps the best-scoring checkpoint</b>, not the final epoch:", st["Body"]))
    e.append(Paragraph(
        "This exists because the training loss falls smoothly for all 50 epochs (0.187 → 0.007) while "
        "validation F-beta peaks early and then <i>drifts down</i>. In your last GPU run the best checkpoint "
        "was <b>epoch 45 (0.601)</b> and epoch 50 scored only 0.558. Without checkpoint selection you would "
        "ship the worse, over-trained epoch-50 model. This is the autoencoder slowly learning to reconstruct "
        "anomalies too (because the unlabeled train set is not guaranteed anomaly-free), which shrinks the very "
        "normal-vs-anomalous error gap the score depends on.", st["Key"]))
    e.append(Paragraph(
        "<b>What if you trained to convergence with no checkpointing?</b> You saw it: ~0.05 F1 lower. "
        "<b>What if you evaluated every epoch instead of every 5?</b> Finer selection, marginally better, "
        "but 5x the validation-scoring cost during training.", st["Body"]))

    e.append(Paragraph("3.6  Scoring  —  per-sensor error, then Mahalanobis distance", st["H2"]))
    e.append(Paragraph(
        "This is the single most important scoring decision in the project. A naive score averages the "
        "reconstruction error over every sensor AND timestep into one number. That is exactly wrong here: a "
        "genuine alarming spike on one normally-precise sensor gets diluted by another sensor's ordinary noise.",
        st["Body"]))
    e.append(num_list([
        "<b>Per-sensor error</b> (<font face='Courier'>get_reconstruction_errors(per_sensor=True)</font>): "
        "average the error over time only, giving one error value <i>per sensor</i> per window — an 18-dim "
        "error vector, not a scalar.",
        "<b>Fit a normal model</b> (<font face='Courier'>fit_mahalanobis</font>): on windows known to be normal, "
        "estimate the mean error vector <font face='Courier'>mu</font> and the inverse covariance "
        "<font face='Courier'>precision</font> using a Ledoit-Wolf shrinkage estimator (well-conditioned even "
        "with few samples).",
        "<b>Mahalanobis distance</b> (<font face='Courier'>mahalanobis_scores</font>): score each window by how "
        "far its error vector sits from the normal cloud, weighting each sensor by how much it <i>normally</i> "
        "varies and accounting for how sensors' errors co-vary. Formula: (x−μ)ᵀ Σ⁻¹ (x−μ).",
    ], st))
    e.append(Paragraph(
        "<b>Why this matters:</b> switching from flat MSE to Mahalanobis was the change that finally moved F1 "
        "from ~0.48 to ~0.60. <b>What if you reverted to flat MSE?</b> You would lose the ability to say "
        "'sensor 7 is off by 4 of its own standard deviations' and fall back to 'the average error is a bit "
        "high' — far less discriminative.", st["Body"]))

    e.append(Paragraph("3.7  From window scores to per-timestep scores", st["H2"]))
    e.append(Paragraph(
        "The submission is per-timestep, but the model scores per-window. "
        "<font face='Courier'>expand_window_scores_to_timesteps</font> scores densely (stride 1) and averages "
        "every overlapping window covering a timestep. A single timestep near the middle of a run is covered by "
        "up to 200 windows; averaging them yields a smooth, robust per-timestep score instead of a coarse "
        "blindly-broadcast block label (the pre-fix behaviour).", st["Body"]))

    e.append(Paragraph("3.8  Thresholding  —  F-beta = 0.5", st["H2"]))
    e.append(Paragraph(
        "A score is continuous; a prediction is 0/1. <font face='Courier'>find_best_threshold</font> sweeps the "
        "precision-recall curve and picks the cutoff maximising F-beta with "
        "<font face='Courier'>beta=0.5</font>. Beta &lt; 1 weights <i>precision</i> above recall. Why: the "
        "normal and anomalous score distributions overlap a lot, and plain F1 (beta=1) kept choosing an "
        "aggressive high-recall threshold that flagged 65–88% of all timesteps — the project's chronic "
        "over-prediction symptom. Beta=0.5 trades some recall for a much saner false-positive rate.", st["Body"]))
    e.append(Paragraph(
        "IMPORTANT NUANCE (a live improvement question): the portal appears to score plain <b>F1</b>. You are "
        "optimising the threshold for F0.5. If the portal metric really is F1, a threshold tuned for F1 could "
        "score higher on the portal. This is worth testing — see the roadmap.", st["Key"]))

    e.append(Paragraph("3.9  Evaluation", st["H2"]))
    e.append(Paragraph(
        "<font face='Courier'>compute_metrics</font> reports precision, recall, F1, ROC-AUC and the predicted "
        "anomaly rate. Note ROC-AUC is computed on the <i>raw scores</i> (threshold-free), so it measures the "
        "quality of the model's <b>ranking</b> — remember that number; Section 6 shows it is your true "
        "ceiling.", st["Body"]))

    e.append(PageBreak())

    # ============ 4. CNN ARCHITECTURE ============
    e.append(Paragraph("4.  The CNN autoencoder, layer by layer", st["H1"]))
    e.append(Paragraph(
        "The input to the network is one window shaped <b>18 channels x 200 timesteps</b> (sensors are the "
        "convolution 'channels', time is the length). The encoder compresses; the decoder rebuilds. The width "
        "of each bar below is proportional to the time dimension, so you can literally see the 200 → 100 → 200 "
        "hourglass.", st["Body"]))
    e.append(cnn_arch_diagram())
    e.append(Paragraph("Figure 2. ConvAutoencoder1D. Only MaxPool1d halves time (200→100); the dilated blocks "
                       "use 'same' padding so they change channels, not length. ConvTranspose1d restores "
                       "200. The final Conv1d maps back to 18 sensors.", st["Cap"]))

    e.append(Paragraph("4.1  What a DilatedResidualBlock does", st["H2"]))
    e.append(bullets([
        "<b>Two Conv1d layers</b> (kernel 3) with BatchNorm and ReLU. A convolution slides a small learned "
        "filter along time, detecting local temporal patterns shared across positions.",
        "<b>Dilation</b> inserts gaps between the filter taps. Dilation 8 means the filter looks at timesteps "
        "t, t±8, t±16 — covering a wide span with few parameters. Stacking dilations 1, 2, 4, 8 grows the "
        "<b>receptive field</b> to ~63 timesteps, so a neuron near the bottleneck 'sees' a long stretch of "
        "context. The old architecture was a single kernel-3 conv with a receptive field of just 3 timesteps — "
        "blind to the slow drifts that are the actual anomalies.",
        "<b>Residual (skip) connection</b>: the block computes <i>output = F(x) + x</i> (a 1x1 conv projects x "
        "when the channel count changes). This lets gradients flow through deep stacks and lets each block "
        "learn a refinement rather than a full transform — the core idea behind ResNets/TCNs.",
        "<b>'Same' padding</b> keeps the time length fixed inside a block because this is reconstruction, not "
        "forecasting — the model may use both past and future context within the window (non-causal).",
    ], st))

    e.append(Paragraph("4.2  Why an hourglass (the bottleneck) is the whole point", st["H2"]))
    e.append(Paragraph(
        "MaxPool1d(2) squeezes time 200→100 at the narrowest point. This <b>bottleneck</b> forces the network "
        "to throw information away and keep only what is needed to rebuild normal windows. An autoencoder with "
        "no bottleneck could learn the identity function (copy input to output) and reconstruct <i>everything</i> "
        "perfectly — including anomalies — making the error score useless. The compression is what makes "
        "anomalies reconstruct badly. <b>What if the bottleneck were larger?</b> Easier reconstruction, smaller "
        "normal-vs-anomalous gap, weaker score. Too small? It cannot even rebuild normal data and everything "
        "looks anomalous. It is a capacity dial.", st["Body"]))

    e.append(Paragraph("4.3  Data-flow table", st["H2"]))
    e.append(table(
        [["Input window", "—", "18 x 200", "one scaled window, sensors as channels"],
         ["Enc block d=1", "conv+BN+ReLU x2, +skip", "32 x 200", "local temporal features"],
         ["Enc block d=2", "dilation 2", "32 x 200", "wider context"],
         ["Enc block d=4", "dilation 4", "32 x 200", "wider context"],
         ["Enc block d=8", "dilation 8, 32→64 ch", "64 x 200", "long-range context, richer channels"],
         ["MaxPool1d(2)", "downsample time", "64 x 100", "bottleneck — forces compression"],
         ["ConvTranspose1d", "upsample stride 2", "32 x 200", "begin reconstruction"],
         ["Dec block d=4", "dilation 4", "32 x 200", "refine"],
         ["Dec block d=2", "dilation 2", "32 x 200", "refine"],
         ["Conv1d k=3", "project to sensors", "18 x 200", "reconstructed window"]],
        ["Stage", "Operation", "Output (C x T)", "Role"],
        [3.0 * cm, 3.6 * cm, 2.3 * cm, 5.0 * cm]))
    e.append(Paragraph("The anomaly score is built from the gap between the Input row and the final row: "
                       "per-sensor MSE between them, then Mahalanobis distance (Section 3.6).", st["Cap"]))

    e.append(PageBreak())

    # ============ 5. WHY THESE CHOICES ============
    e.append(Paragraph("5.  Design choices, at a glance", st["H1"]))
    e.append(table(
        [["Autoencoder (not a classifier)", "No anomaly labels exist to train a classifier",
          "Only way to learn from unlabeled normal data"],
         ["Dilated TCN blocks", "Anomalies are slow drifts; need wide receptive field",
          "RF ~3 → ~63; F1 0.458 → 0.484"],
         ["Bottleneck (MaxPool)", "Prevent identity learning; force compression",
          "Makes anomalies reconstruct badly"],
         ["Per-run windowing", "Runs are independent experiments",
          "Removes fake cross-run transients"],
         ["Per-sensor + Mahalanobis", "Don't dilute one sensor's spike with another's noise",
          "F1 ~0.48 → ~0.60 (biggest single jump)"],
         ["Checkpoint selection", "Val score peaks before training loss does",
          "+~0.05 F1; avoids over-training"],
         ["F-beta = 0.5 threshold", "Overlapped distributions; plain F1 over-predicts",
          "Fixes 65–88% over-flagging"],
         ["Dense scoring + averaging", "Submission is per-timestep",
          "Smooth, honest per-timestep signal"]],
        ["Choice", "Why", "Effect / evidence"],
        [3.9 * cm, 5.1 * cm, 4.9 * cm]))

    # ============ 6. BOTTLENECKS ============
    e.append(Paragraph("6.  Where the F1 is actually lost (bottlenecks & failure modes)", st["H1"]))
    e.append(Paragraph(
        "Your best validation F1 is ~0.60 (portal 0.63) with ROC-AUC ~0.76. The gap between a perfect 1.0 and "
        "your 0.60 has a small number of concrete causes, in rough order of impact:", st["Body"]))
    e.append(num_list([
        "<b>The score-separability ceiling (the big one).</b> ROC-AUC ~0.76 means the model's ranking of "
        "anomalous vs normal timesteps genuinely overlaps. F1 at the best threshold is bounded by that overlap. "
        "No amount of threshold tuning breaks this ceiling — only a <i>better model / better score</i> raises "
        "AUC. This is why score improvements dominate threshold improvements.",
        "<b>Training-set contamination.</b> The 'normal' training data almost certainly contains some "
        "undetected anomalies. The autoencoder slowly learns to reconstruct them, narrowing the normal-vs-"
        "anomalous error gap. Your own training curves (loss keeps dropping while val peaks at epoch ~45) are "
        "the fingerprint of this.",
        "<b>Everything is tuned on the same 10-run validation set.</b> Window size, stride, architecture, "
        "checkpoint, Mahalanobis model, and threshold are all chosen by watching the same 10 runs. Your "
        "validation F1 is therefore an optimistic, 'seen' estimate. (Reassuringly, the portal 0.63 &gt; val "
        "0.60, so you are not badly over-fit — but you have no clean held-out estimate.)",
        "<b>Threshold objective may not match the portal metric.</b> You optimise F0.5; the portal appears to "
        "score F1. That mismatch can leave portal F1 on the table (Section 3.8).",
        "<b>Run-to-run variance.</b> Identical configs scored 0.568–0.611 purely from training stochasticity. "
        "A single run is a noisy estimate of the architecture's true quality.",
    ], st))

    # ============ 7. HOW TO THINK ABOUT IMPROVING ============
    e.append(Paragraph("7.  How to think about improving this model", st["H1"]))
    e.append(Paragraph(
        "This section is about the <i>reasoning process</i>, not a shopping list. Internalise these five habits "
        "and you can generate and triage your own ideas.", st["Body"]))
    e.append(Paragraph("7.1  Diagnose before you prescribe", st["H2"]))
    e.append(Paragraph(
        "Never change the model before you have <i>looked</i> at how it fails. Plot the Mahalanobis score over "
        "time for each validation run with the true anomaly regions shaded. Ask: are we <b>missing</b> "
        "anomalies (false negatives) or <b>over-flagging</b> normal stretches (false positives)? Histogram the "
        "scores for normal vs anomalous timesteps and look at the overlap. These two plots tell you whether "
        "your problem is the threshold, the score, or a specific anomaly type — and each implies a different "
        "fix.", st["Body"]))
    e.append(Paragraph("7.2  Know your ceiling: separate ranking from thresholding", st["H2"]))
    e.append(Paragraph(
        "Two different quantities: ROC-AUC measures how well the model <i>ranks</i> anomalies (threshold-free); "
        "F1 measures a single operating point. If AUC is low, tuning the threshold is rearranging deck chairs — "
        "go improve the score. If AUC is high but F1 is low, <i>then</i> the threshold is your problem. Always "
        "ask which one you are actually fighting.", st["Body"]))
    e.append(Paragraph("7.3  Match the objective to the metric you are graded on", st["H2"]))
    e.append(Paragraph(
        "If the portal scores F1, tuning for F0.5 is optimising the wrong thing. More generally: every place you "
        "'pick the best X on validation' hides an objective (beta, the checkpoint metric, the fusion weight). "
        "Make sure that objective is the thing you ultimately care about.", st["Body"]))
    e.append(Paragraph("7.4  Change one thing, measure honestly, respect variance", st["H2"]))
    e.append(Paragraph(
        "Because a single run varies by ±0.02 F1, a change that 'improves' F1 by 0.01 has proven nothing. "
        "Change one variable at a time; where you can, average over seeds or use run-grouped cross-validation "
        "(you already built this in <font face='Courier'>run_final_hybrid.py</font>) to get an estimate that is "
        "not fooled by noise or by peeking at validation.", st["Body"]))
    e.append(Paragraph("7.5  Attack the dominant bottleneck, not the convenient one", st["H2"]))
    e.append(Paragraph(
        "It is tempting to tweak the threshold because it is easy. But if the ceiling is score separability "
        "(Section 6.1), the high-leverage work is raising AUC: cleaner training signal, better normal model, "
        "better architecture. Spend effort where the bottleneck is, even when it is harder.", st["Body"]))

    # ============ 8. ROADMAP ============
    e.append(Paragraph("8.  Prioritised improvement roadmap", st["H1"]))
    e.append(Paragraph("Each item lists the intuition, the expected benefit, the risk, and what to study first. "
                       "Ordered by expected payoff per unit effort.", st["Body"]))

    def roadmap(title, intuition, benefit, risk, study):
        e.append(Paragraph(title, st["H3"]))
        e.append(bullets([
            f"<b>Intuition:</b> {intuition}",
            f"<b>Expected benefit:</b> {benefit}",
            f"<b>Risk:</b> {risk}",
            f"<b>Study first:</b> {study}",
        ], st))

    roadmap(
        "1. Contamination-robust training (attacks bottleneck 6.2)",
        "Down-weight the highest-loss training windows each batch — they are the likeliest undetected "
        "anomalies polluting your 'normal' set. Cleaner normal signal → bigger error gap → higher AUC.",
        "Directly targets the dominant ceiling; already coded in your teammate's train.py (weight decay + early "
        "stopping + robust down-weighting).",
        "Down-weighting too aggressively discards hard-but-normal windows; needs an A/B against the current CNN.",
        "Robust statistics / trimmed loss; the M-ELBO idea from Xu et al. 2018 (Donut).")
    roadmap(
        "2. Seed-ensemble the CNN",
        "Train 3–5 CNNs with different seeds and average their per-timestep scores. Ensembling cancels each "
        "model's idiosyncratic errors and typically raises AUC; it also removes the 0.568–0.611 run-to-run "
        "lottery.",
        "Low-risk, reliable small gain and a much more stable number to report. Cheap on the V100.",
        "5x training cost; scores must be normalised consistently before averaging.",
        "Ensemble methods; why variance reduction improves AUC.")
    roadmap(
        "3. Fit the Mahalanobis model on train, not validation",
        "Currently mu/precision come from the few normal windows in the 10-run val set. Train has ~18k windows "
        "of (assumed) normal data — a far better-conditioned covariance estimate.",
        "Better normal model → better Mahalanobis scores; also frees validation to be used only for the "
        "threshold.",
        "Train contamination biases mu; mitigate by fitting on the lowest-error training windows only.",
        "Covariance estimation, Ledoit-Wolf shrinkage, Mahalanobis distance geometry.")
    roadmap(
        "4. Align the threshold to the portal metric",
        "If the portal grades F1, choose the threshold that maximises validation F1 (beta=1), or sweep beta and "
        "submit the best. You may already be leaving portal F1 on the table.",
        "Potentially free F1 with zero model change; a few minutes of work.",
        "You only get portal feedback per submission; verify the portal metric first to avoid guessing.",
        "Precision-recall trade-offs; the F-beta family and what beta encodes.")
    roadmap(
        "5. Honest tuning split / nested CV",
        "Reuse the run-grouped GroupKFold you already wrote so window size, threshold, and architecture are "
        "chosen without peeking at the data you report on.",
        "A trustworthy generalisation estimate — stops silent over-fitting to 10 runs.",
        "Fewer runs per fold makes each estimate noisier; needs care with such a small val set.",
        "Cross-validation, data leakage, nested CV.")
    roadmap(
        "6. Architecture / window sweeps (last, not first)",
        "Try window 300, a smaller/larger bottleneck, or LayerNorm instead of BatchNorm.",
        "Possible AUC gains, but diminishing returns and easy to over-fit the search to validation.",
        "Highest risk-per-gain; only worthwhile after 1–4 and with an honest tuning split.",
        "Receptive-field arithmetic, normalization layers, the bias-variance trade-off.")

    e.append(Paragraph(
        "Recommended order of attack: <b>diagnose (7.1) → robust training (1) → seed ensemble (2) → "
        "Mahalanobis-on-train (3) → threshold/metric alignment (4)</b>. Do 5 (honest CV) alongside so every "
        "result you trust is measured without peeking. Leave 6 for last.", st["Key"]))

    e.append(Paragraph("A closing principle", st["H2"]))
    e.append(Paragraph(
        "The strongest version of this project is not the one with the most models — it is the one you "
        "understand well enough to explain every number. You have already shown this discipline: you tried "
        "fusion three ways and rejected it with evidence. Keep that habit. Raise AUC at the source, measure "
        "without fooling yourself, and let the threshold be the last small polish on a score you have earned.",
        st["Body"]))

    doc.build(e)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
