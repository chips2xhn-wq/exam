"""Architecture diagram for com-ply12.py (PhysicsResNetNN + Optuna pipeline).

Drawing conventions follow top-tier CS conference schematics
(NeurIPS/ICML/CVPR): pastel functional-group blocks, dashed residual skip
connection (ResNet, CVPR 2016), stacked encoder blocks with dimension
annotations and a dashed internals inset (Transformer, NeurIPS 2017).

Outputs (SVG primary, PDF/PNG secondary) are saved under ./results/.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ── MANDATORY: editable SVG text ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
})

# ── Palette: muted functional families (NMI-pastel style) ────────────────────
C = {
    'ink':       '#1a1a1a',
    'arrow':     '#4D4D4D',
    'data_fill': '#DCE6F2', 'data_edge': '#3775BA',   # data / input  (blue)
    'enc_fill':  '#C9D6EE', 'enc_edge':  '#0F4D92',   # angle encoder blocks
    'enc_bg':    '#EFF4FB',                            # encoder container tint
    'res_fill':  '#F0E0D0', 'res_edge':  '#B26A3B',   # residual block (peach)
    'res_bg':    '#FBF3EC',
    'mat_fill':  '#DDF3DE', 'mat_edge':  '#3E8E46',   # material branch (green)
    'fus_fill':  '#DCD8F0', 'fus_edge':  '#7C6CCF',   # fusion MLP (violet)
    'fus_bg':    '#F0EFF8',
    'out_fill':  '#AADCA9', 'out_edge':  '#2E7D32',   # output
    'opt_fill':  '#E4CCD8', 'opt_edge':  '#9A4D8E',   # Optuna HPO (rose)
    'trn_fill':  '#F0E0D0', 'trn_edge':  '#B26A3B',   # final training
    'art_fill':  '#EAEAEA', 'art_edge':  '#767676',   # artifacts (neutral)
    'dim':       '#767676',                            # dimension annotations
}

FS_TITLE, FS_BOX, FS_SMALL, FS_DIM = 8.5, 7.0, 6.3, 6.0


def box(ax, x, y, w, h, text, fc, ec, fs=FS_BOX, dashed=False, lw=1.0,
        color=C['ink'], weight='normal', zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0,rounding_size=0.6',
                       fc=fc, ec=ec, lw=lw,
                       linestyle=(0, (4, 2.4)) if dashed else '-',
                       zorder=zorder)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, color=color, fontweight=weight,
                linespacing=1.3, zorder=zorder + 1)


def arrow(ax, x1, y1, x2, y2, dashed=False, lw=1.1, color=C['arrow'],
          ms=8, zorder=3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=ms, lw=lw, color=color,
                                 linestyle=(0, (4, 2.4)) if dashed else '-',
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def dim_tag(ax, x, y, text):
    """Small gray dimension annotation next to a data-flow arrow."""
    ax.text(x, y, text, fontsize=FS_DIM, color=C['dim'], ha='left',
            va='center', style='italic', zorder=4)


# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 1.0], wspace=0.02,
                      left=0.01, right=0.99, top=0.98, bottom=0.01)
axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])
for ax in (axa, axb):
    ax.set_xlim(0, 100); ax.set_ylim(0, 108)
    ax.axis('off')

# ═══ Panel a — network architecture (vertical flow, Transformer-style) ═══════
axa.text(-2, 103.5, 'a', fontsize=11, fontweight='bold', ha='left', va='center')
axa.text(50, 103.5, 'PhysicsResNetNN architecture', fontsize=FS_TITLE,
         fontweight='semibold', ha='center', va='center')

CX = 30          # main flow centre-x

# Input: stacked-laminate icon + label
box(axa, 10, 91.5, 40, 7.0, '', C['data_fill'], C['data_edge'])
for i in range(6):                      # mini ply-stack icon
    y = 97.0 - i * 0.85
    dx = (i % 3 - 1) * 1.6
    axa.plot([14.5 + dx, 21.5 - dx], [y, y], lw=1.0, color=C['data_edge'],
             solid_capstyle='round', zorder=3)
axa.text(24.5, 95.0, '12 ply angles\n$\\theta_1\\ \\dots\\ \\theta_{12}$',
         fontsize=7.5, ha='left', va='center', color=C['ink'],
         linespacing=1.3)

# Periodic encoding
arrow(axa, CX, 91.5, CX, 88.9)
box(axa, 10, 82.2, 40, 6.7, 'Periodic encoding\n$\\sin(2\\theta),\\ \\cos(2\\theta)$',
    C['data_fill'], C['data_edge'])
dim_tag(axa, 51.5, 85.5, '12 → 24')

# ── Angle-encoder container ──
axa.add_patch(FancyBboxPatch((4, 33.5), 52, 46.5,
              boxstyle='round,pad=0,rounding_size=0.8',
              fc=C['enc_bg'], ec=C['enc_edge'], lw=1.0, zorder=1))
axa.text(7, 77.7, 'Angle encoder', fontsize=FS_BOX, fontweight='semibold',
         color=C['enc_edge'], ha='left', va='center')

arrow(axa, CX, 82.2, CX, 74.9)
box(axa, 10, 69.0, 40, 5.9, 'Linear 24 → 256\nLayerNorm · GELU',
    C['enc_fill'], C['enc_edge'])

arrow(axa, CX, 69.0, CX, 66.6)
# Residual block with dashed identity skip (ResNet convention)
box(axa, 13, 55.5, 34, 11.1, 'Residual block ×1\n(dim = 256)',
    C['res_fill'], C['res_edge'], weight='semibold')
sx = 9.0                                # skip-line x
axa.plot([CX, sx], [68.0, 68.0], lw=1.0, color=C['res_edge'],
         linestyle=(0, (4, 2.4)), zorder=3)
axa.plot([sx, sx], [68.0, 53.3], lw=1.0, color=C['res_edge'],
         linestyle=(0, (4, 2.4)), zorder=3)
axa.plot([sx, CX - 2.0], [53.3, 53.3], lw=1.0, color=C['res_edge'],
         linestyle=(0, (4, 2.4)), zorder=3)
circ = Circle((CX, 53.3), 2.0, fc='white', ec=C['res_edge'], lw=1.1, zorder=4)
axa.add_patch(circ)
axa.text(CX, 53.3, '+', fontsize=8.5, ha='center', va='center',
         color=C['res_edge'], zorder=5)
axa.text(CX + 2.8, 51.8, 'GELU$(x + F(x))$', fontsize=FS_DIM,
         color=C['res_edge'], ha='left', va='center', style='italic')

# Flow continues below the + node
arrow(axa, CX, 51.3, CX, 49.9)
box(axa, 10, 44.0, 40, 5.9, 'Linear 256 → 256\nLayerNorm · GELU',
    C['enc_fill'], C['enc_edge'])
arrow(axa, CX, 44.0, CX, 41.4)
box(axa, 10, 35.5, 40, 5.9, 'Linear 256 → $h$ = 42\nLayerNorm · GELU',
    C['enc_fill'], C['enc_edge'])

# ── Residual-block internals inset (right, dashed) ──
ixc = 82                                # inset flow centre-x
axa.add_patch(FancyBboxPatch((59, 39.5), 40, 39.5,
              boxstyle='round,pad=0,rounding_size=0.8',
              fc=C['res_bg'], ec=C['res_edge'], lw=0.9,
              linestyle=(0, (4, 2.4)), zorder=1))
axa.text(61, 76.2, 'Residual block internals', fontsize=FS_SMALL,
         color=C['res_edge'], fontweight='semibold', ha='left', va='center')
steps = ['Linear 256 → 256', 'LayerNorm · GELU', 'Linear 256 → 256', 'LayerNorm']
sy, sh, sgap = 70.0, 4.6, 1.6
for i, s in enumerate(steps):
    box(axa, ixc - 13, sy - sh, 26, sh, s, 'white', C['res_edge'],
        fs=FS_SMALL, lw=0.9)
    if i < len(steps) - 1:
        arrow(axa, ixc, sy - sh, ixc, sy - sh - sgap, lw=0.9, ms=6)
    sy -= (sh + sgap)
isx = 65.5                              # inset skip-line x
top_y, bot_y = 71.2, 44.3
axa.plot([ixc, isx], [top_y, top_y], lw=0.9, color=C['res_edge'],
         linestyle=(0, (4, 2.4)), zorder=3)
axa.plot([isx, isx], [top_y, bot_y], lw=0.9, color=C['res_edge'],
         linestyle=(0, (4, 2.4)), zorder=3)
axa.plot([isx, ixc - 1.8], [bot_y, bot_y], lw=0.9, color=C['res_edge'],
         linestyle=(0, (4, 2.4)), zorder=3)
axa.text(isx - 1.0, (top_y + bot_y) / 2, 'skip', fontsize=FS_DIM,
         color=C['res_edge'], ha='right', va='center', style='italic')
circ2 = Circle((ixc, bot_y), 1.8, fc='white', ec=C['res_edge'], lw=1.0, zorder=4)
axa.add_patch(circ2)
axa.text(ixc, bot_y, '+', fontsize=8, ha='center', va='center',
         color=C['res_edge'], zorder=5)
axa.text(ixc, bot_y - 3.0, 'GELU', fontsize=FS_DIM, color=C['res_edge'],
         ha='center', va='center', style='italic')

# ── Concatenation ──
arrow(axa, CX, 35.5, CX, 31.5)
axa.add_patch(Circle((CX, 29.4), 2.1, fc='white', ec=C['fus_edge'], lw=1.2,
                     zorder=4))
axa.text(CX, 29.4, 'cat', fontsize=FS_SMALL, ha='center', va='center',
         color=C['fus_edge'], zorder=5)
dim_tag(axa, CX + 3.0, 31.8, '$h$ + $m$')

# Material branch (optional interface, dashed)
box(axa, 62, 26.0, 35, 6.5, 'Material params (20 props)\ngroup-wise scaling',
    C['mat_fill'], C['mat_edge'], fs=FS_SMALL, dashed=True)
arrow(axa, 62, 29.4, CX + 2.1, 29.4, dashed=True, color=C['mat_edge'], lw=1.0)
axa.text(79.5, 24.0, 'optional ($m$ = 0 for this dataset)', fontsize=FS_DIM,
         color=C['mat_edge'], ha='center', va='center', style='italic')

# ── Fusion network container ──
axa.add_patch(FancyBboxPatch((4, 6.5), 52, 19.0,
              boxstyle='round,pad=0,rounding_size=0.8',
              fc=C['fus_bg'], ec=C['fus_edge'], lw=1.0, zorder=1))
axa.text(7, 23.2, 'Fusion network', fontsize=FS_BOX, fontweight='semibold',
         color=C['fus_edge'], ha='left', va='center')
arrow(axa, CX, 27.3, CX, 21.8)
box(axa, 8, 16.4, 44, 5.4, 'Linear $(h{+}m)$ → 256\nLayerNorm · GELU · Dropout',
    C['fus_fill'], C['fus_edge'], fs=FS_SMALL)
arrow(axa, CX, 16.4, CX, 15.0)
box(axa, 8, 11.6, 44, 3.4, 'Linear 256 → 32 · LayerNorm · GELU',
    C['fus_fill'], C['fus_edge'], fs=FS_SMALL)
arrow(axa, CX, 11.6, CX, 10.2)
box(axa, 20, 7.2, 20, 3.0, 'Linear 32 → 1', C['fus_fill'], C['fus_edge'],
    fs=FS_SMALL)

# Output
arrow(axa, CX, 7.2, CX, 5.4)
box(axa, 4, 0.2, 52, 5.2, 'Prediction $\\hat{y}$ (log force)\n$\\exp$ · inverse scale → critical force (N)',
    C['out_fill'], C['out_edge'], fs=FS_SMALL, weight='semibold')

# ═══ Panel b — Optuna HPO + training pipeline ════════════════════════════════
axb.text(-2, 103.5, 'b', fontsize=11, fontweight='bold', ha='left', va='center')
axb.text(50, 103.5, 'Optimization & training pipeline', fontsize=FS_TITLE,
         fontweight='semibold', ha='center', va='center')

BX, BW2 = 14, 72                        # panel-b flow: x 14–86
BC = BX + BW2 / 2                       # centre 50

# Dataset
box(axb, BX, 89.0, BW2, 9.0,
    'LaRC05 compression dataset (3,000 samples)\ntrain/val/test = 2,100/600/300 (seed 42)',
    C['data_fill'], C['data_edge'], fs=FS_SMALL)
arrow(axb, BC, 89.0, BC, 86.5)

# Optuna HPO
box(axb, BX, 57.0, BW2, 29.5, '', C['opt_fill'], C['opt_edge'])
axb.text(BC, 83.6, 'Bayesian HPO (Optuna)', fontsize=FS_SMALL,
         fontweight='semibold', ha='center', va='center', color=C['ink'])
axb.text(BC, 80.4, 'TPE sampler · MedianPruner', fontsize=FS_SMALL,
         ha='center', va='center')
axb.text(BC, 77.4, '150 trials × 60 epochs', fontsize=FS_SMALL,
         ha='center', va='center')
axb.text(BC, 73.4, 'search space:', fontsize=FS_DIM, ha='center', va='center',
         color='#333333', style='italic')
axb.text(BC, 68.6,
         '$h$$\\in${6,…,72} · dropout$\\in$[0, 0.4]\n'
         'lr$\\in$[8e-4, 2e-3] · wd$\\in$[8e-5, 2e-4]\nbatch$\\in${16, 32, 64}',
         fontsize=FS_DIM, ha='center', va='center', color='#333333',
         linespacing=1.5)
# pruning loop-back arrow
axb.annotate('', xy=(87.0, 75.0), xytext=(87.0, 81.0),
             arrowprops=dict(arrowstyle='-|>', mutation_scale=7, lw=0.9,
                             color=C['opt_edge'],
                             connectionstyle='arc3,rad=-1.6'))
axb.text(91.5, 71.8, 'prune', fontsize=FS_DIM, color=C['opt_edge'],
         ha='center', va='center', style='italic')

arrow(axb, BC, 57.0, BC, 54.5)
# Best hyperparameters (from run log)
box(axb, BX, 42.5, BW2, 12.0, '', '#F0C0CC', C['opt_edge'])
axb.text(BC, 51.5, 'Best hyperparameters', fontsize=FS_SMALL,
         fontweight='semibold', ha='center', va='center')
axb.text(BC, 47.2, '$h$ = 42 · dropout 0.064\nlr 1.5e-3 · wd 1.1e-4 · batch 32',
         fontsize=FS_SMALL, ha='center', va='center', linespacing=1.4)

arrow(axb, BC, 42.5, BC, 40.0)
# Final training
box(axb, BX, 23.0, BW2, 17.0, '', C['trn_fill'], C['trn_edge'])
axb.text(BC, 36.6, 'Final training (best configuration)', fontsize=FS_SMALL,
         fontweight='semibold', ha='center', va='center')
axb.text(BC, 30.4,
         'AdamW · ReduceLROnPlateau (×0.5)\n'
         '≤ 1,000 epochs · early stopping (60)\n'
         'MSE on scaled log force',
         fontsize=FS_SMALL, ha='center', va='center', linespacing=1.45)

arrow(axb, BC, 23.0, BC, 20.5)
# Test evaluation
box(axb, BX, 11.5, BW2, 9.0,
    'Held-out test evaluation (300 samples)\nRMSE · R² · relative-error audit (>10%)',
    C['out_fill'], C['out_edge'], fs=FS_SMALL)

arrow(axb, BC, 11.5, BC, 10.0)
# Artifacts
box(axb, BX, 0.5, BW2, 9.5,
    'Artifacts: result figure (.png/.pdf)\nmodel (.pth) · scalers (.pkl)',
    C['art_fill'], C['art_edge'], fs=FS_SMALL)

# ── Export ────────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
stem = 'results/architecture_ply12'
fig.savefig(f'{stem}.svg', bbox_inches='tight')
fig.savefig(f'{stem}.pdf', bbox_inches='tight')
fig.savefig(f'{stem}.png', dpi=600, bbox_inches='tight')
plt.close(fig)
print('saved:', stem + '.svg/.pdf/.png')
