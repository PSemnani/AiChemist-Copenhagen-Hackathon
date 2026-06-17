"""
Generate PDF presentation — Academic style (Nature/Science paper aesthetic).
Design: deep navy #003049, warm amber #F4A261, Georgia serif, clean white.
Step-by-step: icon grid overview + one deep-dive slide per step.
"""
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY    = "#003049"
AMBER   = "#F4A261"
WHITE   = "#FFFFFF"
OFFWHT  = "#FAFAFA"
LIGHT   = "#EDF2F4"
BLUEGR  = "#8D99AE"
DARK    = "#2B2D42"
GREEN   = "#2D6A4F"
RED     = "#E84855"
CODE_BG = "#1E2A38"
CODE_FG = "#E8EFF5"
AMBERLG = "#FEF0E4"

W, H = 16, 9


# ── Helpers ──────────────────────────────────────────────────────────────────
def fig():
    f = plt.figure(figsize=(W, H))
    f.patch.set_facecolor(WHITE)
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    return f, ax


def header(ax, title, sub=None):
    ax.add_patch(plt.Rectangle((0, H - 1.55), W, 1.55, facecolor=NAVY, zorder=2))
    ax.add_patch(plt.Rectangle((0, H - 1.57), 0.55, 1.57, facecolor=AMBER, zorder=3))
    ax.text(0.85, H - 0.72, title, fontsize=25, fontweight="bold",
            color=WHITE, va="center", fontfamily="serif", zorder=4)
    if sub:
        ax.text(0.85, H - 1.28, sub, fontsize=12, color=AMBER,
                va="center", zorder=4)


def card(ax, x, y, w, h, title=None, border=NAVY, bg=LIGHT, title_bg=None, radius=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.06",
                                facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2))
    if title:
        tb = title_bg or border
        ax.add_patch(FancyBboxPatch((x, y + h - 0.62), w, 0.62,
                                    boxstyle="round,pad=0.06",
                                    facecolor=tb, edgecolor="none", zorder=3))
        ax.text(x + w / 2, y + h - 0.32, title, fontsize=11,
                fontweight="bold", color=WHITE, ha="center", va="center",
                fontfamily="serif", zorder=4)


def code(ax, x, y, w, h, lines, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                facecolor=CODE_BG, edgecolor=NAVY, linewidth=1.5, zorder=2))
    for i, ln in enumerate(lines):
        ax.text(x + 0.22, y + h - 0.28 - i * (h - 0.3) / max(len(lines), 1),
                ln, fontsize=fs, family="monospace", color=CODE_FG,
                va="top", zorder=3)


def arrow(ax, x1, y1, x2, y2, c=AMBER, lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=20),
                zorder=5)


def bullet(ax, x, y, items, fs=13, gap=0.55, col=DARK):
    for i, it in enumerate(items):
        ax.text(x, y - i * gap, "▸", fontsize=fs, color=AMBER, va="center")
        ax.text(x + 0.38, y - i * gap, it, fontsize=fs, color=col, va="center")


def divider(ax, y=H - 1.56):
    ax.axhline(y, color=LIGHT, lw=1, zorder=1)


def tag(ax, x, y, label, bg=AMBER, fg=NAVY, fs=10):
    tw = len(label) * 0.085 + 0.3
    ax.add_patch(FancyBboxPatch((x, y - 0.2), tw, 0.38,
                                boxstyle="round,pad=0.07",
                                facecolor=bg, edgecolor="none", zorder=3))
    ax.text(x + tw / 2, y, label, fontsize=fs, color=fg,
            ha="center", va="center", fontweight="bold", zorder=4)
    return tw


# ── Slide 1 — Title ───────────────────────────────────────────────────────────
def s_title():
    f, ax = fig()
    # Left navy panel
    ax.add_patch(plt.Rectangle((0, 0), 5.8, H, facecolor=NAVY))
    ax.add_patch(plt.Rectangle((5.8, 0), W - 5.8, H, facecolor=OFFWHT))
    ax.add_patch(plt.Rectangle((5.8, 0), 0.06, H, facecolor=AMBER))
    # Dot grid on right
    for xi in np.linspace(6.5, 15.5, 12):
        for yi in np.linspace(0.5, 8.5, 7):
            ax.plot(xi, yi, "o", ms=1.8, color=NAVY, alpha=0.1)
    # Molecule schematic
    cx, cy, r = 2.9, 5.4, 0.55
    for angle, dr in [(0, 1.2), (72, 1.2), (144, 1.2), (216, 1.2), (288, 1.2)]:
        a = np.radians(angle)
        ax.plot([cx, cx + dr * np.cos(a)], [cy, cy + dr * np.sin(a)],
                color=AMBER, lw=2.5, alpha=0.5, zorder=2)
        ax.plot(cx + dr * np.cos(a), cy + dr * np.sin(a),
                "o", ms=10, color=NAVY, mec=AMBER, mew=2, zorder=3)
    ax.plot(cx, cy, "o", ms=14, color=AMBER, zorder=4)
    # Title text
    ax.text(6.4, 7.0,
            "Training Data\nReconstruction",
            fontsize=34, fontweight="bold", color=NAVY,
            va="center", fontfamily="serif", linespacing=1.15)
    ax.text(6.4, 5.35, "from Molecular Classifiers",
            fontsize=20, color=BLUEGR, va="center", fontfamily="serif")
    ax.plot([6.4, 15.2], [4.7, 4.7], color=AMBER, lw=1.5)
    ax.text(6.4, 4.3,
            "If a model memorized its training data —\ncan you make it confess?",
            fontsize=13, color=DARK, va="center", style="italic",
            fontfamily="serif", linespacing=1.4)
    ax.text(6.4, 2.9, "AiChemist Team  ·  Copenhagen Hackathon 2026",
            fontsize=12, color=BLUEGR)
    # Left panel labels
    for i, t in enumerate(["Membership Inference", "ECFP4 Fingerprints", "Gradient Norm"]):
        y = 2.6 - i * 0.65
        ax.text(0.5, y, "→", fontsize=11, color=AMBER)
        ax.text(0.85, y, t, fontsize=11, color=WHITE)
    return f


# ── Slide 2 — The Challenge ───────────────────────────────────────────────────
def s_challenge():
    f, ax = fig()
    header(ax, "The Challenge")
    ax.text(0.8, 6.65,
            "Two trained neural networks classify molecules by biological activity. "
            "The training data is hidden.",
            fontsize=13, color=DARK, fontfamily="serif")
    ax.text(0.8, 6.1,
            "Goal: submit 1000 molecules per task that exactly match the original training set.",
            fontsize=13, color=NAVY, fontweight="bold", fontfamily="serif")

    specs = [
        ("Task 1 — Binary Classifier", RED,
         ["Architecture: 2048 → 512 → 128 → 2",
          "No regularisation  (no dropout, no batchnorm)",
          "Train accuracy:  ~100%   |   Test accuracy:  ~65%",
          "Deliberately overfit — memorised its training data"]),
        ("Task 2 — 4-Class Classifier", GREEN,
         ["Architecture: 2048 → [BN + ReLU + Dropout] → 512 → 256 → 4",
          "AdamW optimiser  +  early stopping at epoch 4",
          "Train accuracy:  ~90%   |   Test accuracy:  ~86%",
          "Well regularised — generalises to new molecules"]),
    ]
    for i, (title, col, items) in enumerate(specs):
        x = 0.5 + i * 8.0
        card(ax, x, 1.5, 7.3, 4.2, border=col, bg=WHITE)
        ax.add_patch(FancyBboxPatch((x, 5.1), 7.3, 0.62,
                                    boxstyle="round,pad=0.06",
                                    facecolor=col, edgecolor="none", zorder=3))
        ax.text(x + 3.65, 5.4, title, fontsize=13, fontweight="bold",
                color=WHITE, ha="center", fontfamily="serif", zorder=4)
        for j, it in enumerate(items):
            ax.text(x + 0.3, 4.7 - j * 0.62, f"• {it}",
                    fontsize=11, color=DARK)

    ax.text(8.0, 1.05,
            "Both models take an ECFP4 fingerprint as input  (Morgan algorithm · radius 2 · 2048 bits)",
            fontsize=11, color=BLUEGR, ha="center", style="italic", fontfamily="serif")
    return f


# ── Slide 3 — ECFP4 ──────────────────────────────────────────────────────────
def s_ecfp4():
    f, ax = fig()
    header(ax, "ECFP4 Fingerprints", "How a molecule becomes a row of 0s and 1s")

    steps = [
        ("SMILES\nCC(=O)Oc1ccccc1C(=O)O", NAVY),
        ("Circular atom\nenvironments\n(radius 2)", BLUEGR),
        ("Hash each\nsubstructure\nto a bit index", BLUEGR),
        ("2048-bit\nbinary vector\n[0,1,0,1,0,…]", AMBER),
    ]
    xs = [0.6, 3.9, 7.2, 10.5]
    for (label, col), x in zip(steps, xs):
        ax.add_patch(FancyBboxPatch((x, 4.4), 2.9, 2.1,
                                    boxstyle="round,pad=0.1",
                                    facecolor=col, edgecolor="none", alpha=0.12))
        ax.add_patch(FancyBboxPatch((x, 4.4), 2.9, 2.1,
                                    boxstyle="round,pad=0.1",
                                    facecolor="none", edgecolor=col, linewidth=1.8))
        ax.text(x + 1.45, 5.5, label, fontsize=11, color=DARK,
                ha="center", va="center", fontfamily="serif", fontweight="bold", linespacing=1.3)
    for x in [3.5, 6.8, 10.1]:
        arrow(ax, x, 5.5, x + 0.35, 5.5)

    # Bit strip
    bits = [0,1,0,0,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0]
    for j, b in enumerate(bits):
        c = AMBER if b else LIGHT
        ec = NAVY if b else BLUEGR
        ax.add_patch(plt.Rectangle((0.55 + j * 0.43, 3.05), 0.36, 0.65,
                                   facecolor=c, edgecolor=ec, lw=0.8, zorder=3))
    ax.text(0.55 + len(bits) * 0.43 / 2, 2.78,
            "← 35 of 2048 bits shown   ·   amber = substructure present   ·   grey = absent →",
            fontsize=10, color=BLUEGR, ha="center", style="italic")

    bullet(ax, 0.7, 2.2, [
        "Same molecule always gives the same fingerprint",
        "Similar molecules share many bits  (measured by Tanimoto similarity)",
        "The mapping is one-way — you cannot reconstruct a molecule from bits alone",
    ], fs=12, gap=0.52)
    return f


# ── Slide 4 — Approach 1: REINVENT ───────────────────────────────────────────
def s_rl():
    f, ax = fig()
    header(ax, "Approach 1 — Generative Pipeline", "REINVENT RL  ·  attempted first")

    boxes = [
        (0.5,  "Pre-trained\nSMILES LSTM\n(prior)", NAVY),
        (3.7,  "RL Agent\n(trainable\ncopy)", AMBER),
        (6.9,  "Task Model\n(scoring\nfunction)", NAVY),
        (10.1, "REINFORCE\nLoss", RED),
    ]
    for x, lbl, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 3.6), 2.8, 2.5,
                                    boxstyle="round,pad=0.12",
                                    facecolor=col, edgecolor="none", alpha=0.1))
        ax.add_patch(FancyBboxPatch((x, 3.6), 2.8, 2.5,
                                    boxstyle="round,pad=0.12",
                                    facecolor="none", edgecolor=col, linewidth=2))
        ax.text(x + 1.4, 4.85, lbl, fontsize=11, color=DARK,
                ha="center", va="center", fontfamily="serif", fontweight="bold",
                linespacing=1.3)
    for x in [3.3, 6.5, 9.7]:
        arrow(ax, x, 4.85, x + 0.35, 4.85)

    code(ax, 0.5, 1.5, 12.5, 1.75, [
        "augmented_ll  =  prior_log_prob(smiles)  +  σ × confidence(smiles)",
        "loss          =  mean( ( augmented_ll  −  agent_log_prob(smiles) )² )",
        "#  agent steers toward molecules the task model scores highly",
    ])

    # Result badge
    ax.add_patch(FancyBboxPatch((13.1, 3.5), 2.6, 2.8,
                                boxstyle="round,pad=0.12",
                                facecolor=RED, edgecolor="none", alpha=0.1))
    ax.add_patch(FancyBboxPatch((13.1, 3.5), 2.6, 2.8,
                                boxstyle="round,pad=0.12",
                                facecolor="none", edgecolor=RED, linewidth=2))
    ax.text(14.4, 5.45, "Result", fontsize=13, color=RED, ha="center",
            fontfamily="serif", fontweight="bold")
    ax.text(14.4, 4.75, "0 / 1000", fontsize=26, color=RED,
            ha="center", fontweight="bold")
    ax.text(14.4, 4.15, "correct", fontsize=12, color=RED, ha="center")
    ax.text(14.4, 3.7, "submissions", fontsize=10, color=BLUEGR, ha="center")

    ax.text(7.0, 1.2,
            "Generates plausible drug-like molecules — but not the specific molecules in the training set.",
            fontsize=12, color=DARK, ha="center", style="italic", fontfamily="serif")
    return f


# ── Slide 5 — Logit Insight ───────────────────────────────────────────────────
def s_logit():
    f, ax = fig()
    header(ax, "The Key Insight — Raw Logits",
           "Softmax hides what logits reveal")

    # Left panel
    card(ax, 0.5, 1.1, 7.0, 5.9, border=RED, bg=WHITE)
    ax.add_patch(FancyBboxPatch((0.5, 6.38), 7.0, 0.62,
                                boxstyle="round,pad=0.06",
                                facecolor=RED, edgecolor="none", zorder=3))
    ax.text(4.0, 6.69, "Softmax confidence  —  saturates",
            fontsize=12, fontweight="bold", color=WHITE,
            ha="center", fontfamily="serif", zorder=4)
    for i, (logit, prob) in enumerate([(10, "0.9999"), (27, "0.9999"), (66, "1.0000")]):
        y = 5.55 - i * 0.88
        ax.text(1.1, y, f"logit = {logit}", fontsize=12, color=DARK,
                family="monospace", va="center")
        arrow(ax, 3.3, y, 4.1, y, c=BLUEGR, lw=1.5)
        ax.text(4.4, y, f"p = {prob}", fontsize=12, color=DARK,
                family="monospace", va="center")
        ax.text(6.5, y, "≈ 1.00", fontsize=12, color=RED,
                fontweight="bold", va="center")
    ax.text(4.0, 2.45, "All look identical.",
            fontsize=13, color=RED, ha="center",
            fontweight="bold", fontfamily="serif")
    ax.text(4.0, 1.85, "Cannot distinguish training\nfrom non-training molecules.",
            fontsize=11, color=BLUEGR, ha="center", fontfamily="serif")

    # Right panel
    card(ax, 8.5, 1.1, 7.0, 5.9, border=GREEN, bg=WHITE)
    ax.add_patch(FancyBboxPatch((8.5, 6.38), 7.0, 0.62,
                                boxstyle="round,pad=0.06",
                                facecolor=GREEN, edgecolor="none", zorder=3))
    ax.text(12.0, 6.69, "Raw logit  —  full dynamic range",
            fontsize=12, fontweight="bold", color=WHITE,
            ha="center", fontfamily="serif", zorder=4)

    rows = [("Wrong dataset", 10, 10/66), ("Closer dataset", 27, 27/66), ("Training source", 66, 1.0)]
    colors = [RED, AMBER, GREEN]
    for i, ((name, val, frac), col) in enumerate(zip(rows, colors)):
        y = 5.55 - i * 1.1
        ax.text(8.9, y + 0.12, name, fontsize=11, color=DARK, fontfamily="serif")
        ax.text(8.9, y - 0.28, f"max logit ≈ {val}", fontsize=9.5,
                color=BLUEGR, family="monospace")
        bar = frac * 4.8
        ax.add_patch(FancyBboxPatch((12.0, y - 0.22), bar, 0.48,
                                    boxstyle="round,pad=0.04",
                                    facecolor=col, edgecolor="none", alpha=0.85, zorder=3))
        ax.text(12.1 + bar, y, f"  {val}", fontsize=11, color=col,
                fontweight="bold", va="center", zorder=4)

    ax.text(12.0, 2.45, "6× logit gap reveals the training source.",
            fontsize=13, color=GREEN, ha="center",
            fontweight="bold", fontfamily="serif")
    ax.text(12.0, 1.85, "The model's internal certainty\nbetray which molecules it memorised.",
            fontsize=11, color=BLUEGR, ha="center", fontfamily="serif")
    return f


# ── Slide 6 — Submission 1 Overview (Icon Grid) ───────────────────────────────
def s_overview():
    f, ax = fig()
    header(ax, "Submission 1 — At a Glance", "Six steps · fold-based · gradient norm")

    steps = [
        ("01", "Load\nDataset",     "CARBIDE data_splits.tsv\n1988 molecules total",  NAVY),
        ("02", "Filter\nFolds",     "Keep Fold != 0\n1588 training molecules",        AMBER),
        ("03", "Fingerprint",       "ECFP4 · 2048 bits\nRDKit Morgan radius 2",       NAVY),
        ("04", "Run\nModel",        "Forward pass\nGet logit scores",                 AMBER),
        ("05", "Gradient\nNorm",    "Backprop loss\nSmall norm = memorised",          NAVY),
        ("06", "Export\nTop-1000",  "Task 1: top-1000\nTask 2: 250 per class",        GREEN),
    ]

    cols, rows = 3, 2
    cw, ch = 4.8, 2.85
    xs = [0.5 + c * (cw + 0.28) for c in range(cols)]
    ys = [1.2 + r * (ch + 0.2) for r in range(rows - 1, -1, -1)]

    for i, (icon, title, desc, col) in enumerate(steps):
        c, r = i % cols, i // cols
        x, y = xs[c], ys[r]
        ax.add_patch(FancyBboxPatch((x, y), cw, ch,
                                    boxstyle="round,pad=0.12",
                                    facecolor=col, edgecolor="none", alpha=0.08, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y), cw, ch,
                                    boxstyle="round,pad=0.12",
                                    facecolor="none", edgecolor=col, linewidth=2, zorder=3))
        ax.add_patch(plt.Circle((x + 0.62, y + ch - 0.65), 0.38,
                               color=col, zorder=4))
        ax.text(x + 0.62, y + ch - 0.65, icon, fontsize=13, color=WHITE,
                ha="center", va="center", fontweight="bold", zorder=5)
        ax.text(x + 1.35, y + ch - 0.60, title, fontsize=12,
                fontweight="bold", color=DARK, va="center",
                fontfamily="serif", linespacing=1.2, zorder=4)
        for j, line in enumerate(desc.split("\n")):
            ax.text(x + 0.3, y + ch - 1.45 - j * 0.52, line,
                    fontsize=11, color=DARK, zorder=4)
        tag(ax, x + cw - 1.0, y + 0.3, f"Step {i+1}", bg=col, fg=WHITE, fs=9)

    return f


# ── Slide 7 — Step 1+2: Data ──────────────────────────────────────────────────
def s_step12():
    f, ax = fig()
    header(ax, "Steps 1 & 2 — Load and Filter", "Finding the training molecules")

    # Step 1
    card(ax, 0.5, 1.2, 7.3, 5.7, border=NAVY, bg=WHITE)
    ax.text(0.85, 6.53, "Step 1 — Load Dataset", fontsize=13,
            fontweight="bold", color=NAVY, fontfamily="serif")
    ax.text(0.85, 6.05,
            "CARBIDE's data_splits.tsv contains every molecule\n"
            "used in the train/test cross-validation splits.",
            fontsize=11.5, color=DARK, linespacing=1.4)

    # Mini table
    headers = ["SMILES", "Cluster_ID", "Fold"]
    col_xs = [0.9, 4.5, 6.3]
    ax.add_patch(plt.Rectangle((0.7, 4.55), 6.8, 0.45, facecolor=NAVY, zorder=3))
    for hx, h in zip(col_xs, headers):
        ax.text(hx, 4.77, h, fontsize=10, color=WHITE, fontweight="bold",
                va="center", zorder=4)
    rows_data = [("CC1=Nc2cc(C(F)…", "26", "0"), ("C#CC(O)(C=CCl)…", "232", "1"),
                 ("C#CC1(O)CCC2C…", "1", "1"), ("Brc1c(NC2=NCCN…", "14", "3")]
    for ri, (s, c, fo) in enumerate(rows_data):
        y = 4.25 - ri * 0.45
        if ri % 2 == 0:
            ax.add_patch(plt.Rectangle((0.7, y - 0.18), 6.8, 0.4, facecolor=LIGHT, zorder=2))
        for hx, val in zip(col_xs, [s, c, fo]):
            ax.text(hx, y, val, fontsize=9.5, color=DARK, va="center",
                    family="monospace")

    ax.text(0.85, 1.7, "Total: 1988 molecules across 5 folds (Fold 0–4)",
            fontsize=11, color=BLUEGR, style="italic", fontfamily="serif")

    # Step 2
    card(ax, 8.2, 1.2, 7.3, 5.7, border=AMBER, bg=WHITE)
    ax.text(8.55, 6.53, "Step 2 — Filter to Training Folds",
            fontsize=13, fontweight="bold", color=NAVY, fontfamily="serif")
    ax.text(8.55, 6.05,
            "With seed = 0, fold 0 was held out as the test set.\n"
            "We keep only folds 1, 2, 3, 4 — the training set.",
            fontsize=11.5, color=DARK, linespacing=1.4)

    # Fold diagram
    fold_cols = [RED, GREEN, GREEN, GREEN, GREEN]
    fold_labels = ["Fold 0\n(test)", "Fold 1", "Fold 2", "Fold 3", "Fold 4"]
    for j, (fc, fl) in enumerate(zip(fold_cols, fold_labels)):
        x = 8.5 + j * 1.38
        ax.add_patch(FancyBboxPatch((x, 4.0), 1.18, 1.65,
                                    boxstyle="round,pad=0.08",
                                    facecolor=fc, edgecolor="none", alpha=0.15))
        ax.add_patch(FancyBboxPatch((x, 4.0), 1.18, 1.65,
                                    boxstyle="round,pad=0.08",
                                    facecolor="none", edgecolor=fc, linewidth=2))
        ax.text(x + 0.59, 4.82, fl, fontsize=10, ha="center", color=DARK,
                fontfamily="serif", fontweight="bold", linespacing=1.3)
        if j > 0:
            ax.text(x + 0.59, 4.15, "~397", fontsize=9, ha="center", color=fc)

    ax.text(9.3, 3.68, "→  kept (training)", fontsize=11, color=GREEN)
    ax.text(8.56, 3.68, "✗  excluded", fontsize=11, color=RED)

    code(ax, 8.4, 1.6, 6.9, 1.6, [
        "splits = pd.read_csv('data_splits.tsv', sep='\\t')",
        "train  = splits[splits['Fold'] != 0]['SMILES']",
        "# → 1588 training molecules",
    ], fs=9.5)
    return f


# ── Slide 8 — Steps 3+4: Model ────────────────────────────────────────────────
def s_step34():
    f, ax = fig()
    header(ax, "Steps 3 & 4 — Fingerprint & Forward Pass",
           "Converting molecules to numbers, then asking the model")

    # Step 3
    card(ax, 0.5, 1.2, 7.3, 5.7, border=NAVY, bg=WHITE)
    ax.text(0.85, 6.53, "Step 3 — ECFP4 Fingerprint",
            fontsize=13, fontweight="bold", color=NAVY, fontfamily="serif")
    ax.text(0.85, 6.0,
            "Each molecule is converted to a 2048-bit binary vector.\n"
            "This is the language the model understands.",
            fontsize=11.5, color=DARK, linespacing=1.4)
    # Bit strip
    bits = [0,1,0,0,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1]
    for j, b in enumerate(bits):
        bx = 0.75 + j * 0.33
        ax.add_patch(plt.Rectangle((bx, 4.5), 0.27, 0.6,
                                   facecolor=AMBER if b else LIGHT,
                                   edgecolor=NAVY if b else BLUEGR, lw=0.8, zorder=3))
    ax.text(0.75 + 20 * 0.33 / 2, 4.25, "… 2048 bits total",
            fontsize=9.5, color=BLUEGR, ha="center", style="italic")
    code(ax, 0.7, 1.6, 6.8, 2.4, [
        "from rdkit.Chem import rdFingerprintGenerator",
        "gen = GetMorganGenerator(radius=2, fpSize=2048)",
        "fp  = gen.GetFingerprintAsNumPy(mol)",
        "# → array([0,1,0,0,1,1,…], dtype=float32)",
    ], fs=9.5)

    # Step 4
    card(ax, 8.2, 1.2, 7.3, 5.7, border=AMBER, bg=WHITE)
    ax.text(8.55, 6.53, "Step 4 — Run the Model",
            fontsize=13, fontweight="bold", color=NAVY, fontfamily="serif")
    ax.text(8.55, 6.0,
            "Feed each fingerprint into the task model.\n"
            "The model outputs one raw score (logit) per class.",
            fontsize=11.5, color=DARK, linespacing=1.4)

    # Network diagram
    layer_xs = [9.0, 10.6, 12.0, 13.4]
    layer_ns = [5, 4, 3, 2]
    layer_lbls = ["2048", "512", "128", "2"]
    for lx, ln, ll in zip(layer_xs, layer_ns, layer_lbls):
        for ni in range(ln):
            ny = 4.8 - ni * 0.52 + (ln - 1) * 0.26
            ax.plot(lx, ny, "o", ms=11, color=NAVY, zorder=4)
        ax.text(lx, 3.38, ll, fontsize=9, color=BLUEGR, ha="center")
    for i in range(len(layer_xs) - 1):
        for n1 in range(layer_ns[i]):
            y1 = 4.8 - n1 * 0.52 + (layer_ns[i] - 1) * 0.26
            for n2 in range(layer_ns[i + 1]):
                y2 = 4.8 - n2 * 0.52 + (layer_ns[i + 1] - 1) * 0.26
                ax.plot([layer_xs[i], layer_xs[i + 1]], [y1, y2],
                        color=LIGHT, lw=0.7, zorder=2)

    # Output logits
    ax.add_patch(FancyBboxPatch((13.8, 4.15), 1.5, 1.3,
                                boxstyle="round,pad=0.08",
                                facecolor=AMBER, edgecolor="none", alpha=0.15, zorder=3))
    ax.text(14.55, 5.1, "logit₀", fontsize=11, color=NAVY, ha="center",
            fontfamily="serif")
    ax.text(14.55, 4.55, "logit₁", fontsize=11, color=NAVY, ha="center",
            fontfamily="serif")
    arrow(ax, 13.4, 4.8, 13.75, 4.8, c=AMBER)

    code(ax, 8.4, 1.6, 6.9, 1.7, [
        "X = torch.from_numpy(np.stack(fingerprints))",
        "logits = model(X)   # shape: (n_molecules, n_classes)",
        "# → tensor([[-3.1, 8.7], [-12.4, 22.1], …])",
    ], fs=9.5)
    return f


# ── Slide 9 — Step 5: Gradient Norm ──────────────────────────────────────────
def s_step5():
    f, ax = fig()
    header(ax, "Step 5 — Gradient Norm",
           "The membership signal: how much does the model need to change its mind?")

    # Analogy
    ax.add_patch(FancyBboxPatch((0.5, 5.6), 7.0, 1.8,
                                boxstyle="round,pad=0.12",
                                facecolor=AMBERLG, edgecolor=AMBER, linewidth=2))
    tag(ax, 0.65, 7.25, "Analogy", bg=AMBER, fg=NAVY)
    ax.text(0.85, 6.85,
            "Think of a student taking an exam.",
            fontsize=12.5, color=NAVY, fontfamily="serif", fontweight="bold")
    ax.text(0.85, 6.38,
            "A student who memorised the answer → no hesitation → gradient ≈ 0\n"
            "A student who is guessing → unsure → needs to adjust → gradient > 0",
            fontsize=11.5, color=DARK, linespacing=1.45)

    # Left: math
    card(ax, 0.5, 1.1, 7.0, 4.3, border=NAVY, bg=WHITE)
    ax.text(0.85, 5.05, "The Mathematics", fontsize=13, color=NAVY,
            fontweight="bold", fontfamily="serif")
    ax.text(0.85, 4.55,
            "We backpropagate the loss through the model to the input:",
            fontsize=11.5, color=DARK)
    code(ax, 0.65, 3.15, 6.7, 1.15, [
        "x.requires_grad_(True)",
        "F.cross_entropy(model(x), pred).backward()",
    ], fs=10)
    ax.text(0.85, 2.78,
            "Then measure how large that gradient is:",
            fontsize=11.5, color=DARK)
    code(ax, 0.65, 1.85, 6.7, 0.72, [
        "gradient_norm = x.grad.norm()",
    ], fs=10.5)

    # Right: signal chart
    card(ax, 8.2, 1.1, 7.3, 4.3, border=GREEN, bg=WHITE)
    ax.text(8.55, 5.05, "What the signal looks like", fontsize=13,
            color=NAVY, fontweight="bold", fontfamily="serif")

    examples = [
        ("Training molecule\n(memorised)", 0.002, GREEN),
        ("Training molecule\n(memorised)", 0.009, GREEN),
        ("Non-training\n(unknown)",        0.18,  RED),
        ("Non-training\n(unknown)",        0.31,  RED),
    ]
    for i, (lbl, val, col) in enumerate(examples):
        y = 4.35 - i * 0.82
        ax.text(8.55, y, lbl, fontsize=10, color=DARK, va="center",
                fontfamily="serif", linespacing=1.2)
        bar = val / 0.35 * 4.5
        ax.add_patch(FancyBboxPatch((11.2, y - 0.22), bar, 0.4,
                                    boxstyle="round,pad=0.04",
                                    facecolor=col, edgecolor="none",
                                    alpha=0.8, zorder=3))
        ax.text(11.25 + bar, y, f"  {val:.3f}", fontsize=10,
                color=col, va="center", fontweight="bold")

    ax.text(11.0, 1.55, "← smaller gradient norm", fontsize=10, color=GREEN)
    ax.text(11.0, 1.15, "= model is already at a loss minimum", fontsize=10, color=GREEN)
    ax.text(11.0, 0.75, "= this molecule was memorised", fontsize=10, color=GREEN,
            fontweight="bold")
    return f


# ── Slide 10 — Step 6: Export ─────────────────────────────────────────────────
def s_step6():
    f, ax = fig()
    header(ax, "Step 6 — Rank and Export",
           "Sort by gradient norm, take the top molecules")

    ax.text(0.8, 6.65,
            "Sort all 1588 training-fold molecules by gradient norm (ascending).\n"
            "Smallest norm first = most memorised = most likely in training set.",
            fontsize=13, color=DARK, fontfamily="serif", linespacing=1.4)

    # Task 1
    card(ax, 0.5, 1.2, 7.3, 5.0, border=RED, bg=WHITE)
    ax.text(0.85, 5.85, "Task 1 — Binary Classifier", fontsize=13,
            fontweight="bold", color=RED, fontfamily="serif")
    ax.text(0.85, 5.35, "Take the 1000 molecules with the smallest gradient norm.",
            fontsize=11.5, color=DARK)
    code(ax, 0.65, 3.6, 7.1, 1.55, [
        "norms  = gradient_norms(model1, fingerprints)",
        "top    = np.argsort(norms)[:1000]",
        "result = [smiles[i] for i in top]",
    ], fs=9.5)
    ax.text(0.85, 3.25, "norm range in top-1000:  0.0000 – 0.0363", fontsize=11,
            color=BLUEGR, style="italic", family="monospace")
    ax.add_patch(FancyBboxPatch((0.65, 1.35), 7.1, 1.65,
                                boxstyle="round,pad=0.08",
                                facecolor=LIGHT, edgecolor=RED, linewidth=1.5))
    ax.text(4.2, 2.8, "Output", fontsize=11, color=RED, ha="center",
            fontweight="bold", fontfamily="serif")
    ax.text(4.2, 2.35, "task1_submission.csv", fontsize=11, color=DARK,
            ha="center", family="monospace")
    ax.text(4.2, 1.75, "1000 molecules", fontsize=12, color=DARK,
            ha="center", fontweight="bold")

    # Task 2
    card(ax, 8.2, 1.2, 7.3, 5.0, border=GREEN, bg=WHITE)
    ax.text(8.55, 5.85, "Task 2 — 4-Class Classifier", fontsize=13,
            fontweight="bold", color=GREEN, fontfamily="serif")
    ax.text(8.55, 5.35,
            "For each of the 4 classes, compute the gradient norm\n"
            "using that class as the target. Assign each molecule\n"
            "to its most 'natural' class, take top 250 per class.",
            fontsize=11.5, color=DARK, linespacing=1.4)
    code(ax, 8.4, 3.2, 7.0, 1.55, [
        "for c in range(4):",
        "    norms[:,c] = gradient_norms(model2, fps, class_idx=c)",
        "assigned = np.argmin(norms, axis=1)",
    ], fs=9.5)
    ax.add_patch(FancyBboxPatch((8.4, 1.35), 7.0, 1.65,
                                boxstyle="round,pad=0.08",
                                facecolor=LIGHT, edgecolor=GREEN, linewidth=1.5))
    ax.text(11.9, 2.8, "Output", fontsize=11, color=GREEN, ha="center",
            fontweight="bold", fontfamily="serif")
    ax.text(11.9, 2.35, "task2_submission.csv", fontsize=11, color=DARK,
            ha="center", family="monospace")
    ax.text(11.9, 1.75, "250 × 4 classes  =  1000 molecules",
            fontsize=12, color=DARK, ha="center", fontweight="bold")
    return f


# ── Slide 11 — Results ────────────────────────────────────────────────────────
def s_results():
    f, ax = fig()
    header(ax, "Results — The Journey")

    ax.plot([0.8, 15.5], [4.5, 4.5], color=LIGHT, lw=3, zorder=1)

    events = [
        (1.4,  "Random\nGeneration",     "0 correct",  RED,   "Prior generates valid\nmolecules, not specific ones"),
        (4.4,  "Large Database\nScreening",  "0 correct",  RED,   "Wrong dataset —\nlogits all similar (~10)"),
        (7.8,  "Logit Signal\nDiscovery",  "60+ correct", AMBER, "Right domain found\nvia logit magnitude jump"),
        (11.4, "Gradient\nNorm",          "Best result", GREEN, "Principled approach:\nfold-based + ∂L/∂x ≈ 0"),
    ]
    for x, label, result, col, note in events:
        ax.plot(x, 4.5, "o", ms=20, color=col, zorder=3)
        ax.text(x, 4.5, "→", fontsize=11, color=WHITE,
                ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(x, 5.5, label, fontsize=11, color=DARK, ha="center",
                fontfamily="serif", fontweight="bold", linespacing=1.3)

        ax.add_patch(FancyBboxPatch((x - 1.35, 2.3), 2.7, 1.85,
                                    boxstyle="round,pad=0.1",
                                    facecolor=col, edgecolor="none", alpha=0.1))
        ax.add_patch(FancyBboxPatch((x - 1.35, 2.3), 2.7, 1.85,
                                    boxstyle="round,pad=0.1",
                                    facecolor="none", edgecolor=col, linewidth=1.8))
        ax.text(x, 3.68, result, fontsize=13, color=col,
                ha="center", fontweight="bold", fontfamily="serif")
        ax.text(x, 2.88, note, fontsize=9.5, color=DARK,
                ha="center", linespacing=1.3)

    ax.add_patch(FancyBboxPatch((0.5, 0.5), 15.0, 1.5,
                                boxstyle="round,pad=0.12",
                                facecolor=LIGHT, edgecolor=NAVY, linewidth=1.5))
    ax.text(8.0, 1.55, "Key finding", fontsize=12, color=NAVY, ha="center",
            fontweight="bold", fontfamily="serif")
    ax.text(8.0, 0.95,
            "Raw logit magnitude — not softmax confidence — is the discriminative membership signal. "
            "A 6× logit gap between the wrong and right dataset is unmistakable.",
            fontsize=11, color=DARK, ha="center")
    return f


# ── Slide 12 — Takeaways ─────────────────────────────────────────────────────
def s_takeaways():
    f, ax = fig()
    f.patch.set_facecolor(NAVY)
    ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor=NAVY))
    ax.add_patch(plt.Rectangle((0, 0), 0.55, H, facecolor=AMBER))

    ax.text(W / 2 + 0.28, 8.15, "Takeaways",
            fontsize=34, color=WHITE, ha="center",
            fontweight="bold", fontfamily="serif")
    ax.plot([2.0, W - 1.0], [7.55, 7.55], color=AMBER, lw=1.5, alpha=0.5)

    items = [
        (AMBER, "Dataset identification matters more than algorithm sophistication"),
        (GREEN, "Use raw logits — not softmax — for membership inference in overfit models"),
        (WHITE, "Gradient norm is the principled signal: ∂L/∂x ≈ 0 at training data points"),
        (BLUEGR,"A generative model cannot reconstruct specific training molecules"),
        (AMBER, "Cross-validation fold structure reveals the train/test split boundary"),
    ]
    for i, (col, text) in enumerate(items):
        y = 6.7 - i * 1.08
        ax.add_patch(plt.Circle((2.3, y), 0.3, color=col, zorder=3))
        ax.text(2.3, y, str(i + 1), fontsize=13, color=NAVY if col == AMBER else DARK,
                ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(2.85, y, text, fontsize=13, color=WHITE, va="center")

    ax.plot([2.0, W - 1.0], [0.95, 0.95], color=AMBER, lw=1, alpha=0.3)
    ax.text(W / 2 + 0.28, 0.6,
            "github.com/PSemnani/AiChemist-Copenhagen-Hackathon",
            fontsize=11.5, color=BLUEGR, ha="center", style="italic")
    return f


# ── RENDER ────────────────────────────────────────────────────────────────────
slides = [
    s_title, s_challenge, s_ecfp4, s_rl, s_logit,
    s_overview, s_step12, s_step34, s_step5, s_step6,
    s_results, s_takeaways,
]

out = "docs/presentation.pdf"
with PdfPages(out) as pdf:
    for fn in slides:
        fig_obj = fn()
        pdf.savefig(fig_obj, bbox_inches="tight", dpi=150)
        plt.close(fig_obj)
        print(f"  ✓  {fn.__name__}")

print(f"\nSaved → {out}  ({len(slides)} slides)")
