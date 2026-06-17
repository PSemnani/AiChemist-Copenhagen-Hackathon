"""Generate a PDF presentation for the AiChemist Copenhagen Hackathon project."""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY   = "#1B3264"
TEAL   = "#2AB7CA"
WHITE  = "#FFFFFF"
LIGHT  = "#F4F6F9"
DARK   = "#1E1E2E"
CORAL  = "#E84855"
GREEN  = "#2ECC71"
GRAY   = "#6C757D"
LGRAY  = "#DEE2E6"
CODE_BG = "#2D2D2D"
CODE_FG = "#F8F8F2"
YELLOW = "#F9C74F"

W, H = 16, 9  # slide dimensions (inches)


def new_slide(bg=WHITE):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def header_bar(ax, title, subtitle=None):
    ax.add_patch(FancyBboxPatch((0, H - 1.6), W, 1.6, boxstyle="square,pad=0",
                                facecolor=NAVY, edgecolor="none", zorder=2))
    ax.add_patch(plt.Rectangle((0, H - 1.62), 0.6, 1.62,
                                facecolor=TEAL, zorder=3))
    ax.text(0.85, H - 0.78, title, fontsize=26, fontweight="bold",
            color=WHITE, va="center", ha="left", zorder=4)
    if subtitle:
        ax.text(0.85, H - 1.35, subtitle, fontsize=13, color=TEAL,
                va="center", ha="left", zorder=4)


def code_box(ax, x, y, w, h, code_lines, fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=CODE_BG, edgecolor=TEAL, linewidth=1.5))
    for i, line in enumerate(code_lines):
        ax.text(x + 0.2, y + h - 0.28 - i * 0.3, line,
                fontsize=fontsize, family="monospace", color=CODE_FG, va="top")


def pill(ax, x, y, label, color=TEAL, text_color=WHITE, fontsize=11, w=None):
    tw = w or (len(label) * 0.095 + 0.35)
    ax.add_patch(FancyBboxPatch((x, y - 0.22), tw, 0.42,
                                boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor="none", alpha=0.9))
    ax.text(x + tw / 2, y, label, fontsize=fontsize, color=text_color,
            ha="center", va="center", fontweight="bold")
    return tw


def arrow(ax, x1, y1, x2, y2, color=TEAL, lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=18))


def bullet(ax, x, y, items, fontsize=13, spacing=0.52, color=DARK,
           marker="▸", marker_color=TEAL):
    for i, item in enumerate(items):
        yi = y - i * spacing
        ax.text(x, yi, marker, fontsize=fontsize, color=marker_color, va="center")
        ax.text(x + 0.35, yi, item, fontsize=fontsize, color=color, va="center")


# ── SLIDE 1 — Title ───────────────────────────────────────────────────────────
def slide_title():
    fig, ax = new_slide(NAVY)
    # Accent strip
    ax.add_patch(plt.Rectangle((0, 0), 0.8, H, facecolor=TEAL))
    ax.add_patch(plt.Rectangle((0.8, 0), W - 0.8, H, facecolor=NAVY))
    # Molecule grid decoration
    for xi in np.linspace(1.5, 15.5, 8):
        for yi in np.linspace(0.5, 8.5, 5):
            ax.plot(xi, yi, "o", ms=2, color=TEAL, alpha=0.08)
    # Title
    ax.text(W / 2 + 0.4, 5.8,
            "Training Data Reconstruction",
            fontsize=36, fontweight="bold", color=WHITE,
            ha="center", va="center")
    ax.text(W / 2 + 0.4, 4.95,
            "from Molecular Classifiers",
            fontsize=36, fontweight="bold", color=TEAL,
            ha="center", va="center")
    ax.text(W / 2 + 0.4, 3.9,
            "If a model memorized its training data — can you make it confess?",
            fontsize=15, color=LGRAY, ha="center", va="center",
            style="italic")
    # Divider
    ax.plot([3, W - 1], [3.3, 3.3], color=TEAL, lw=1.5, alpha=0.4)
    ax.text(W / 2 + 0.4, 2.75, "AiChemist Team  ·  Copenhagen Hackathon 2026",
            fontsize=13, color=LGRAY, ha="center", va="center")
    # Badge
    ax.add_patch(FancyBboxPatch((6.5, 1.5), 4.0, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor=TEAL, edgecolor="none", alpha=0.25))
    ax.text(W / 2 + 0.4, 1.85, "Membership Inference  ·  ECFP4  ·  Neural Networks",
            fontsize=11, color=TEAL, ha="center", va="center")
    return fig


# ── SLIDE 2 — The Challenge ───────────────────────────────────────────────────
def slide_challenge():
    fig, ax = new_slide()
    header_bar(ax, "The Challenge")

    ax.text(0.8, 6.3, "Two black-box neural networks classify molecules by biological activity.",
            fontsize=14, color=DARK, va="center")
    ax.text(0.8, 5.85, "Goal: submit 1000 molecules per task that match the original training set.",
            fontsize=14, color=DARK, va="center", fontweight="bold")

    # Task cards
    for i, (title, items, color) in enumerate([
        ("Task 1 — Binary Classifier",
         ["Architecture: 2048 → 512 → 128 → 2",
          "No dropout, no batch norm",
          "Train acc: ~100%   Test acc: ~65%",
          "Deliberately overfit — memorized training data"],
         CORAL),
        ("Task 2 — 4-Class Classifier",
         ["Architecture: 2048 → [BN+ReLU+Dropout] → 512 → 256 → 4",
          "AdamW + early stopping (epoch 4)",
          "Train acc: ~90%   Test acc: ~86%",
          "Regularized — generalizes well"],
         TEAL),
    ]):
        x = 0.6 + i * 7.8
        ax.add_patch(FancyBboxPatch((x, 1.4), 7.0, 3.9,
                                    boxstyle="round,pad=0.15",
                                    facecolor=LIGHT, edgecolor=color, linewidth=2))
        ax.add_patch(FancyBboxPatch((x, 4.7), 7.0, 0.65,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + 3.5, 5.02, title, fontsize=14, fontweight="bold",
                color=WHITE, ha="center", va="center")
        for j, item in enumerate(items):
            ax.text(x + 0.45, 4.15 - j * 0.62, f"• {item}",
                    fontsize=11.5, color=DARK, va="center")

    ax.text(8.0, 1.0, "Both models take ECFP4 fingerprints as input  (Morgan, radius 2, 2048 bits)",
            fontsize=12, color=GRAY, ha="center", style="italic")
    return fig


# ── SLIDE 3 — ECFP4 Fingerprints ─────────────────────────────────────────────
def slide_ecfp4():
    fig, ax = new_slide()
    header_bar(ax, "ECFP4 Fingerprints", "How molecules become numbers")

    ax.text(0.8, 6.9,
            "Each molecule is encoded as a 2048-bit binary vector — its 'digital barcode'.",
            fontsize=13, color=DARK)

    # SMILES → FP diagram
    boxes = [
        (1.0, 5.1, 3.2, "SMILES\nCC(=O)Oc1ccccc1C(=O)O", TEAL),
        (5.2, 5.1, 3.2, "Circular atom\nenvironments", NAVY),
        (9.4, 5.1, 3.2, "Hash to bit\nposition (mod 2048)", NAVY),
        (11.8, 3.6, 4.8, "2048-bit vector\n[0,1,0,0,1,1,0,...]", CORAL),
    ]
    for (x, y, w, label, color) in boxes[:3]:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.4,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor="none", alpha=0.15))
        ax.add_patch(FancyBboxPatch((x, y), w, 1.4,
                                    boxstyle="round,pad=0.1",
                                    facecolor="none", edgecolor=color, linewidth=1.5))
        ax.text(x + w / 2, y + 0.7, label, fontsize=11, color=DARK,
                ha="center", va="center", fontweight="bold")

    arrow(ax, 4.2, 5.8, 5.1, 5.8)
    arrow(ax, 8.4, 5.8, 9.3, 5.8)
    arrow(ax, 12.6, 5.1, 12.6, 4.9)

    x, y, w, label, color = boxes[3]
    ax.add_patch(FancyBboxPatch((x, y), w, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor="none", alpha=0.15))
    ax.add_patch(FancyBboxPatch((x, y), w, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor="none", edgecolor=color, linewidth=1.5))
    ax.text(x + w / 2, y + 0.65, label, fontsize=11, color=DARK,
            ha="center", va="center", fontweight="bold")

    # Bit vector visual
    bits = [0,1,0,0,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,1]
    for j, b in enumerate(bits):
        color = TEAL if b else LGRAY
        ax.add_patch(plt.Rectangle((0.6 + j * 0.46, 2.0), 0.38, 0.55,
                                   facecolor=color, edgecolor=WHITE, lw=0.5))
    ax.text(0.6 + len(bits) * 0.46 / 2, 1.65,
            "← 32 of 2048 bits shown  —  each '1' means a specific chemical substructure is present",
            fontsize=10, color=GRAY, ha="center")

    bullet(ax, 0.8, 2.9, [
        "Same molecule always produces the same fingerprint",
        "Two similar molecules have similar fingerprints (measurable by Tanimoto similarity)",
        "Fingerprinting is one-way — you cannot reconstruct a molecule from its bits alone",
    ], fontsize=12, spacing=0.48)
    return fig


# ── SLIDE 4 — Approach 1: REINVENT ───────────────────────────────────────────
def slide_approach1():
    fig, ax = new_slide()
    header_bar(ax, "Approach 1 — Generative Pipeline", "REINVENT RL (attempted first)")

    # Pipeline boxes
    steps = [
        ("Pre-trained\nSMILES LSTM\n(prior)", TEAL, 1.2),
        ("RL Agent\n(trainable\ncopy)", NAVY, 4.2),
        ("Scoring\nFunction\n(task model)", TEAL, 7.2),
        ("REINFORCE\nLoss", CORAL, 10.2),
    ]
    for (label, color, x) in steps:
        ax.add_patch(FancyBboxPatch((x, 3.7), 2.5, 2.0,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="none", alpha=0.15))
        ax.add_patch(FancyBboxPatch((x, 3.7), 2.5, 2.0,
                                    boxstyle="round,pad=0.15",
                                    facecolor="none", edgecolor=color, linewidth=2))
        ax.text(x + 1.25, 4.7, label, fontsize=11, color=DARK,
                ha="center", va="center", fontweight="bold")

    for x in [3.7, 6.7, 9.7]:
        arrow(ax, x, 4.7, x + 0.45, 4.7)

    # RL formula
    code_box(ax, 1.1, 2.0, 11.5, 1.4, [
        "augmented_ll  =  prior_log_prob(smiles)  +  σ × confidence(smiles)",
        "loss          =  mean( ( augmented_ll  −  agent_log_prob(smiles) )² )",
        "→  agent learns to generate molecules the task model is confident about",
    ], fontsize=10)

    # Result badge
    ax.add_patch(FancyBboxPatch((11.8, 3.5), 3.8, 2.3,
                                boxstyle="round,pad=0.15",
                                facecolor=CORAL, edgecolor="none", alpha=0.12))
    ax.add_patch(FancyBboxPatch((11.8, 3.5), 3.8, 2.3,
                                boxstyle="round,pad=0.15",
                                facecolor="none", edgecolor=CORAL, linewidth=2))
    ax.text(13.7, 5.15, "Result", fontsize=13, color=CORAL,
            ha="center", fontweight="bold")
    ax.text(13.7, 4.6, "0 / 1000", fontsize=22, color=CORAL,
            ha="center", fontweight="bold")
    ax.text(13.7, 4.05, "correct", fontsize=13, color=CORAL, ha="center")

    ax.text(8.0, 1.35,
            "The model generates valid, drug-like molecules — but not the specific training molecules.",
            fontsize=12, color=DARK, ha="center", style="italic")
    ax.text(8.0, 0.85,
            "A generative model produces plausible molecules from its learned distribution, not from the task model's memory.",
            fontsize=11, color=GRAY, ha="center")
    return fig


# ── SLIDE 5 — The Logit Insight ───────────────────────────────────────────────
def slide_logit():
    fig, ax = new_slide()
    header_bar(ax, "The Key Insight — Raw Logits", "Why softmax fails as a membership signal")

    # Left: softmax saturation
    ax.add_patch(FancyBboxPatch((0.5, 1.2), 7.0, 5.5,
                                boxstyle="round,pad=0.15",
                                facecolor=LIGHT, edgecolor=CORAL, linewidth=1.5))
    ax.text(4.0, 6.4, "Softmax Confidence", fontsize=14, color=CORAL,
            ha="center", fontweight="bold")
    ax.text(4.0, 6.0, "saturates — loses information", fontsize=11,
            color=GRAY, ha="center")

    rows = [
        ("logit = 10", "→", "0.9999", "≈ 1.00"),
        ("logit = 27", "→", "0.9999", "≈ 1.00"),
        ("logit = 66", "→", "1.0000", "≈ 1.00"),
    ]
    for i, (l, arr, val, rnd) in enumerate(rows):
        y = 5.2 - i * 0.9
        ax.text(0.9, y, l, fontsize=12, color=DARK, family="monospace")
        ax.text(2.9, y, arr, fontsize=12, color=CORAL)
        ax.text(3.4, y, f"softmax = {val}", fontsize=12, color=DARK, family="monospace")
        ax.text(5.9, y, rnd, fontsize=12, color=CORAL, fontweight="bold")
    ax.text(4.0, 2.0, "All look identical!", fontsize=13,
            color=CORAL, ha="center", fontweight="bold")
    ax.text(4.0, 1.55,
            "You cannot tell training members from outsiders.",
            fontsize=11, color=GRAY, ha="center")

    # Right: raw logit signal
    ax.add_patch(FancyBboxPatch((8.2, 1.2), 7.3, 5.5,
                                boxstyle="round,pad=0.15",
                                facecolor=LIGHT, edgecolor=GREEN, linewidth=1.5))
    ax.text(11.85, 6.4, "Raw Logit", fontsize=14, color=GREEN,
            ha="center", fontweight="bold")
    ax.text(11.85, 6.0, "keeps full dynamic range", fontsize=11,
            color=GRAY, ha="center")

    datasets = [
        ("Wrong dataset", 10, 0.0, CORAL),
        ("Closer dataset", 27, 0.53, YELLOW),
        ("Training source", 66, 1.0, GREEN),
    ]
    for i, (name, logit, frac, color) in enumerate(datasets):
        y = 5.1 - i * 1.1
        ax.text(8.5, y + 0.05, name, fontsize=11, color=DARK)
        ax.text(8.5, y - 0.3, f"max logit ≈ {logit}", fontsize=10,
                color=GRAY, family="monospace")
        bar_w = frac * 5.5
        ax.add_patch(FancyBboxPatch((11.2, y - 0.22), bar_w, 0.42,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="none", alpha=0.85))
        ax.text(11.2 + bar_w + 0.15, y, f"{logit}",
                fontsize=11, color=color, va="center", fontweight="bold")

    ax.text(11.85, 2.1,
            "6× logit gap reveals training source",
            fontsize=13, color=GREEN, ha="center", fontweight="bold")
    ax.text(11.85, 1.6,
            "The model 'remembers' — and logits betray it.",
            fontsize=11, color=GRAY, ha="center")
    return fig


# ── SLIDE 6 — Membership Inference Pipeline ───────────────────────────────────
def slide_pipeline():
    fig, ax = new_slide()
    header_bar(ax, "Membership Inference Pipeline", "The winning approach")

    steps = [
        ("1. Load\nDataset", "CARBIDE data_splits.tsv\nTraining folds 1–4\n~1588 molecules", TEAL),
        ("2. Filter\nFolds", "Keep Fold ≠ 0\n(seed=0 → fold 0\nwas held-out test)", NAVY),
        ("3. ECFP4\nFingerprint", "RDKit Morgan\nradius=2, 2048 bits\nfloat32 0/1 vector", TEAL),
        ("4. Gradient\nNorm", "Backprop loss\nthrough task model\nto input fingerprint", NAVY),
        ("5. Rank &\nExport", "Sort ascending\nTop-1000 for Task 1\n250/class for Task 2", CORAL),
    ]
    for i, (title, desc, color) in enumerate(steps):
        x = 0.5 + i * 3.1
        ax.add_patch(FancyBboxPatch((x, 3.3), 2.7, 3.0,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="none", alpha=0.12))
        ax.add_patch(FancyBboxPatch((x, 3.3), 2.7, 3.0,
                                    boxstyle="round,pad=0.15",
                                    facecolor="none", edgecolor=color, linewidth=2))
        ax.add_patch(FancyBboxPatch((x, 5.7), 2.7, 0.65,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="none", alpha=0.9))
        ax.text(x + 1.35, 6.02, title, fontsize=11, color=WHITE,
                ha="center", va="center", fontweight="bold")
        for j, line in enumerate(desc.split("\n")):
            ax.text(x + 1.35, 5.15 - j * 0.52, line, fontsize=10,
                    color=DARK, ha="center", va="center")
        if i < 4:
            arrow(ax, x + 2.7, 4.8, x + 3.05, 4.8)

    # Formula box
    code_box(ax, 0.5, 1.2, 15.0, 1.8, [
        "x.requires_grad_(True)",
        "loss = cross_entropy( model(x),  predicted_class )",
        "loss.backward()",
        "gradient_norm = x.grad.norm()     # small norm  →  loss minimum  →  memorized",
    ], fontsize=10.5)
    return fig


# ── SLIDE 7 — Why Gradient Norm Works ────────────────────────────────────────
def slide_gradnorm():
    fig, ax = new_slide()
    header_bar(ax, "Why Gradient Norm Works", "Mathematical intuition")

    # Left column: math intuition
    ax.text(0.7, 6.8, "The Mathematics", fontsize=15, color=NAVY, fontweight="bold")

    bullet(ax, 0.7, 6.3, [
        "Cross-entropy loss is minimized at training data points",
        "At a minimum, the gradient = 0  (definition of minimum)",
        "Non-training molecules are not at any minimum → gradient > 0",
    ], fontsize=12, spacing=0.58)

    code_box(ax, 0.6, 3.3, 7.2, 1.7, [
        "∂Loss/∂x = (softmax(logits) − one_hot(y)) · ∂logits/∂x",
        "         ≈  0   for training molecules   (loss ≈ 0)",
        "         >  0   for non-training molecules",
    ], fontsize=10)

    ax.text(0.7, 3.0, "Therefore:", fontsize=13, color=DARK, fontweight="bold")
    for i, (label, color) in enumerate([
        ("small gradient norm  →  loss minimum  →  memorized (training)", GREEN),
        ("large gradient norm  →  not a minimum  →  unknown molecule", CORAL),
    ]):
        ax.add_patch(FancyBboxPatch((0.6, 2.35 - i * 0.75), 7.2, 0.58,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor="none", alpha=0.12))
        ax.add_patch(FancyBboxPatch((0.6, 2.35 - i * 0.75), 7.2, 0.58,
                                    boxstyle="round,pad=0.08",
                                    facecolor="none", edgecolor=color, linewidth=1.5))
        ax.text(0.85, 2.64 - i * 0.75, label, fontsize=11, color=DARK, va="center")

    # Right column: visual
    ax.text(9.0, 6.8, "Loss Landscape (schematic)", fontsize=15, color=NAVY, fontweight="bold")

    x_vals = np.linspace(8.5, 15.5, 300)
    # loss landscape: valleys at training points
    y_base = 4.5
    y_loss = y_base + 1.2 * (
        0.5 + 0.4 * np.sin(2.0 * (x_vals - 8.5)) +
        0.6 * np.sin(3.3 * (x_vals - 8.5)) +
        0.3 * np.sin(5.5 * (x_vals - 8.5))
    )
    ax.plot(x_vals, y_loss, color=NAVY, lw=2.5, zorder=3)
    ax.fill_between(x_vals, y_base - 0.1, y_loss, color=NAVY, alpha=0.06)

    # Training points (minima)
    for xp, label in [(10.2, "Training\nmolecule"), (13.1, "Training\nmolecule")]:
        yp = y_base + 1.2 * (0.5 + 0.4 * np.sin(2.0 * (xp - 8.5)) +
                              0.6 * np.sin(3.3 * (xp - 8.5)) +
                              0.3 * np.sin(5.5 * (xp - 8.5)))
        ax.plot(xp, yp, "o", ms=10, color=GREEN, zorder=5)
        ax.text(xp, yp - 0.45, label, fontsize=9, color=GREEN,
                ha="center", va="top", fontweight="bold")
        ax.annotate("", xy=(xp, yp), xytext=(xp, yp - 0.1),
                    arrowprops=dict(arrowstyle="-", color=GREEN,
                                   lw=1.5, linestyle="dashed"))

    # Non-training point (on slope)
    xnt = 11.6
    ynt = y_base + 1.2 * (0.5 + 0.4 * np.sin(2.0 * (xnt - 8.5)) +
                           0.6 * np.sin(3.3 * (xnt - 8.5)) +
                           0.3 * np.sin(5.5 * (xnt - 8.5)))
    ax.plot(xnt, ynt, "o", ms=10, color=CORAL, zorder=5)
    ax.text(xnt, ynt + 0.25, "Non-training\nmolecule", fontsize=9,
            color=CORAL, ha="center", va="bottom", fontweight="bold")
    ax.annotate("", xy=(xnt + 0.4, ynt - 0.15),
                xytext=(xnt, ynt),
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=2, mutation_scale=14))
    ax.text(xnt + 0.55, ynt - 0.2, "gradient ≠ 0", fontsize=9, color=CORAL, va="center")

    ax.text(12.0, 1.6, "gradient ≈ 0", fontsize=10, color=GREEN,
            ha="center", fontweight="bold")
    ax.text(12.0, 1.2, "at training points", fontsize=10, color=GREEN, ha="center")
    ax.plot([10.2, 13.1], [1.5, 1.5], color=GREEN, lw=1.5, linestyle="--", alpha=0.5)

    ax.text(8.5, 1.0, "x  (fingerprint space)", fontsize=10, color=GRAY)
    ax.text(8.1, 3.0, "L\no\ns\ns", fontsize=10, color=GRAY, ha="center")
    return fig


# ── SLIDE 8 — Results ─────────────────────────────────────────────────────────
def slide_results():
    fig, ax = new_slide()
    header_bar(ax, "Results & Journey")

    # Timeline
    events = [
        (1.2, "Random\nGeneration", "0 correct", CORAL, "Prior generates valid\nmolecules — not training ones"),
        (4.2, "Large\nDatabase\nScreening", "0 correct", CORAL, "Wrong dataset — logits\nall similar (~10)"),
        (7.5, "Logit Signal\nDiscovery", "60+ correct", YELLOW, "Right dataset found via\nlogit magnitude jump"),
        (11.0, "Gradient\nNorm", "Best result", GREEN, "Principled membership\ninference on training folds"),
    ]
    ax.plot([0.9, 14.8], [4.5, 4.5], color=LGRAY, lw=2.5, zorder=1)

    for (x, label, result, color, note) in events:
        ax.plot(x, 4.5, "o", ms=18, color=color, zorder=3)
        ax.text(x, 4.5, "→", fontsize=10, color=WHITE,
                ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(x, 5.35, label, fontsize=10, color=DARK,
                ha="center", va="center", fontweight="bold", ma="center")
        ax.add_patch(FancyBboxPatch((x - 1.2, 2.5), 2.4, 1.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor="none", alpha=0.12))
        ax.add_patch(FancyBboxPatch((x - 1.2, 2.5), 2.4, 1.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor="none", edgecolor=color, linewidth=1.5))
        ax.text(x, 3.55, result, fontsize=13, color=color,
                ha="center", va="center", fontweight="bold")
        ax.text(x, 2.85, note, fontsize=8.5, color=DARK,
                ha="center", va="center", ma="center")

    # Bottom key lesson
    ax.add_patch(FancyBboxPatch((0.5, 0.55), 15.0, 1.55,
                                boxstyle="round,pad=0.15",
                                facecolor=NAVY, edgecolor="none", alpha=0.08))
    ax.add_patch(FancyBboxPatch((0.5, 0.55), 15.0, 1.55,
                                boxstyle="round,pad=0.15",
                                facecolor="none", edgecolor=NAVY, linewidth=1.5))
    ax.text(8.0, 1.6, "Key lesson", fontsize=12, color=NAVY,
            ha="center", fontweight="bold")
    ax.text(8.0, 1.05,
            "Raw logit magnitude (not softmax confidence) is the discriminative membership signal in overfit models. "
            "A 6× logit gap between the wrong and right dataset is unmistakable.",
            fontsize=11, color=DARK, ha="center", va="center")
    return fig


# ── SLIDE 9 — Takeaways ───────────────────────────────────────────────────────
def slide_takeaways():
    fig, ax = new_slide(NAVY)
    ax.add_patch(plt.Rectangle((0, 0), 0.8, H, facecolor=TEAL))

    ax.text(W / 2 + 0.4, 8.0, "Takeaways", fontsize=30,
            color=WHITE, ha="center", fontweight="bold")
    ax.plot([2, W - 1], [7.5, 7.5], color=TEAL, lw=1.5, alpha=0.4)

    takeaways = [
        (TEAL,  "1",  "Dataset identification matters more than algorithm sophistication"),
        (GREEN, "2",  "Use raw logits — not softmax — for membership inference in overfit models"),
        (YELLOW,"3",  "Gradient norm is the principled signal: training points sit at loss minima"),
        (CORAL, "4",  "A generative model cannot reconstruct specific training molecules"),
        (TEAL,  "5",  "Fold structure in the dataset reveals the train/test split"),
    ]
    for i, (color, num, text) in enumerate(takeaways):
        y = 6.6 - i * 1.1
        ax.add_patch(plt.Circle((2.1, y), 0.3, color=color, zorder=3))
        ax.text(2.1, y, num, fontsize=14, color=DARK if color == YELLOW else WHITE,
                ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(2.65, y, text, fontsize=13.5, color=WHITE, va="center")

    ax.text(W / 2 + 0.4, 0.7,
            "github.com/PSemnani/AiChemist-Copenhagen-Hackathon",
            fontsize=12, color=TEAL, ha="center", style="italic")
    return fig


# ── RENDER ────────────────────────────────────────────────────────────────────
slides = [
    slide_title,
    slide_challenge,
    slide_ecfp4,
    slide_approach1,
    slide_logit,
    slide_pipeline,
    slide_gradnorm,
    slide_results,
    slide_takeaways,
]

out = "docs/presentation.pdf"
with PdfPages(out) as pdf:
    for fn in slides:
        fig = fn()
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  ✓  {fn.__name__}")

print(f"\nSaved → {out}")
