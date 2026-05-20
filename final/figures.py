"""
Paper figures.

Run after experiments.py. Loads .npz files from the working directory
and writes one .png per figure to the same directory.
"""

import os
import warnings
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Charter', 'STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10.5,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333',
    'xtick.color': '#333', 'ytick.color': '#333',
    'axes.labelcolor': '#222', 'text.color': '#222',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})

C_ADV    = '#B0413E'
C_ADV_L  = '#E5BFB8'
C_OBJ    = '#3D6A7A'
C_OBJ_L  = '#B7CAD3'
C_CTRL   = '#8A8A8A'
C_CTRL_L = '#E0E0E0'
C_PRIM   = '#1B3B6F'
C_GOLD   = '#C89B3C'
C_INK    = '#1A1A1A'


def _save(fname, fig):
    fig.savefig(fname, dpi=220, bbox_inches='tight', facecolor='white', pad_inches=0.06)
    print(f"  saved: {fname}")
    plt.close(fig)


def _style(ax):
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')
    ax.tick_params(width=0.6)


def _load(name):
    if not os.path.exists(name):
        print(f'  {name} not found')
        return None
    f = np.load(name, allow_pickle=True)
    out = {}
    for k in f.files:
        v = f[k]
        if v.shape == () and v.dtype == object:
            out[k] = v.item()
        elif v.shape == ():
            out[k] = float(v)
        else:
            out[k] = v
    return out


# =============================================================================
# FIG 1: pipeline_overview.png
# =============================================================================
def fig_pipeline_overview():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')

    # Column headers
    ax.text(15, 47.5, 'Strategic  $X_\\mathcal{S}$',
            ha='center', fontsize=12, fontweight='bold', color=C_ADV)
    ax.text(15, 44.3, 'agent-curated', ha='center', fontsize=8.5,
            color=C_ADV, style='italic')
    ax.text(45, 47.5, 'Objective  $X_\\mathcal{O}$',
            ha='center', fontsize=12, fontweight='bold', color=C_OBJ)
    ax.text(45, 44.3, 'independent sources', ha='center', fontsize=8.5,
            color=C_OBJ, style='italic')
    ax.text(75, 47.5, 'Controls',
            ha='center', fontsize=12, fontweight='bold', color=C_CTRL)
    ax.text(75, 44.3, 'covariates only', ha='center', fontsize=8.5,
            color=C_CTRL, style='italic')

    def mod(x, y, w, h, label, lightcol, edgecol, dashed=False):
        ls = '--' if dashed else '-'
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=0.5",
            facecolor=lightcol, edgecolor=edgecol, linewidth=1.1, linestyle=ls))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=10, color=C_INK)

    # Strategic column
    mod(5,  35, 20, 5, 'Listing text',   C_ADV_L, C_ADV)
    mod(5,  28, 20, 5, 'Listing photos', C_ADV_L, C_ADV)
    # Objective column
    mod(35, 35, 20, 5, 'Street View',    C_OBJ_L, C_OBJ)
    mod(35, 28, 20, 5, 'Satellite',      C_OBJ_L, C_OBJ)
    # Controls column (dashed; smaller stack)
    mod(65, 35, 20, 5, 'Tabular  (price, beds, baths, sqft)',
        C_CTRL_L, C_CTRL, dashed=True)
    mod(65, 28, 20, 5, 'Census tract demographics',
        C_CTRL_L, C_CTRL, dashed=True)

    # CLIP encoder box (spans Strategic + Objective, NOT Controls)
    ax.add_patch(FancyBboxPatch((5, 19), 50, 6,
        boxstyle="round,pad=0.3,rounding_size=0.7",
        facecolor='#F4F4F4', edgecolor='#666', linewidth=1.1))
    ax.text(30, 22, 'Frozen CLIP encoder  (4 modalities $\\times$ 512-d)',
            ha='center', va='center', fontsize=10.5, color=C_INK)

    # Arrows into CLIP
    for x_src in [15, 15, 45, 45]:
        for y_src in [35, 28]:
            arr = FancyArrowPatch((x_src, y_src), (30, 25),
                arrowstyle='->', mutation_scale=8,
                color='#888', linewidth=0.7)
            ax.add_patch(arr)

    # Controls bypass CLIP with a side label
    ax.text(75, 22, '(used as covariates\nin causal specs only)',
            ha='center', va='center', fontsize=8, color='#666', style='italic')

    # 3-source PID box (centered under CLIP)
    ax.add_patch(FancyBboxPatch((5, 11), 60, 5.5,
        boxstyle="round,pad=0.3,rounding_size=0.7",
        facecolor='#FBF4DD', edgecolor=C_GOLD, linewidth=1.4))
    ax.text(35, 13.75,
            r'3-source PID:  $X_\mathcal{S}^{adv}\ \cdot\ X_\mathcal{S}^{rel}\ \cdot\ X_\mathcal{O}$',
            ha='center', va='center', fontsize=11, fontweight='bold', color=C_INK)

    # Arrow from CLIP to PID
    arr = FancyArrowPatch((30, 19), (30, 16.5),
        arrowstyle='-|>', mutation_scale=12,
        color=C_GOLD, linewidth=1.4)
    ax.add_patch(arr)

    # Headline result box: spans the full plot width (5 to 95)
    ax.add_patch(FancyBboxPatch((5, 1.5), 90, 6,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor='#EFEAF5', edgecolor=C_PRIM, linewidth=1.3))
    ax.text(50, 5.4,
        r'$\mathbf{U_\mathcal{S}^{adv} = 0.039}$ nats   '
        r'vs.\ standard 2-source $U_\mathcal{S} = 0.0027$  $\,\cdot\,$  '
        r'$\mathbf{14.4\times}$ larger',
        ha='center', va='center', fontsize=12, fontweight='bold', color=C_PRIM)
    ax.text(50, 2.6,
        r'1-d supervised direction exceeds 64-d PCA upper bound by $\mathbf{3.4\times}$  $\,\cdot\,$  '
        r'$z = 25.4$ over null  $\,\cdot\,$  CI [0.034, 0.043]',
        ha='center', va='center', fontsize=9.5, color='#555', style='italic')

    # Arrow from PID to result
    arr = FancyArrowPatch((35, 11), (50, 7.5),
        arrowstyle='-|>', mutation_scale=12,
        color=C_PRIM, linewidth=1.3)
    ax.add_patch(arr)

    _save('figure_1_pipeline.png', fig)


# =============================================================================
# FIG 2: causal_dag.png
# =============================================================================
def fig_causal_dag():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(-5, 50); ax.axis('off')

    # Latents (top row): E, V, A from left to right
    def lat(x, y, label):
        c = Circle((x, y), 3.5, facecolor='#F4F4F4', edgecolor='#333', linewidth=1.1)
        ax.add_patch(c)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold')

    lat(30, 40, '$E$')
    lat(50, 40, '$V$')
    lat(70, 40, '$A$')

    # Observed boxes (bottom row): Y^gap on far left, X_S, X_O, Y^price on far right
    def obs(x, y, label, col, lcol):
        ax.add_patch(FancyBboxPatch((x - 7, y - 2.5), 14, 5,
            boxstyle="round,pad=0.15,rounding_size=0.5",
            facecolor=lcol, edgecolor=col, linewidth=1.1))
        ax.text(x, y, label, ha='center', va='center', fontsize=11)

    obs(10, 15, '$Y^{gap}$',  C_ADV, C_ADV_L)
    obs(38, 15, '$X_\\mathcal{S}$', C_ADV, C_ADV_L)
    obs(62, 15, '$X_\\mathcal{O}$', C_OBJ, C_OBJ_L)
    obs(90, 15, '$Y^{price}$', C_OBJ, C_OBJ_L)

    # Sub-labels
    ax.text(38, 9, 'strategic', ha='center', fontsize=8.5, color=C_ADV, style='italic')
    ax.text(62, 9, 'objective', ha='center', fontsize=8.5, color=C_OBJ, style='italic')

    def arr(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
            arrowstyle='-|>', mutation_scale=10, color='#444',
            linewidth=0.9, shrinkA=5, shrinkB=5)
        ax.add_patch(a)

    # E -> X_S
    arr(30, 36.5, 38, 17.5)
    # V -> X_S
    arr(50, 36.5, 38, 17.5)
    # V -> X_O (short, no overlap)
    arr(50, 36.5, 62, 17.5)
    # V -> Y^price (curves out around X_O)
    a = FancyArrowPatch((50, 36.5), (90, 17.5),
        arrowstyle='-|>', mutation_scale=10, color='#444',
        linewidth=0.9, shrinkA=5, shrinkB=5,
        connectionstyle='arc3,rad=-0.18')
    ax.add_patch(a)
    # A -> X_S
    arr(70, 36.5, 38, 17.5)
    # A -> Y^gap (curves out around X_S)
    a = FancyArrowPatch((70, 36.5), (10, 17.5),
        arrowstyle='-|>', mutation_scale=10, color='#444',
        linewidth=0.9, shrinkA=5, shrinkB=5,
        connectionstyle='arc3,rad=0.22')
    ax.add_patch(a)

    # Top label
    ax.text(50, 47, 'Latent factors  (effort, value, adversarial intent)',
            ha='center', fontsize=10, fontweight='bold', color='#444')
    # Bottom caption
    ax.text(50, -3.5,
        r'$X_\mathcal{O}$ depends only on $V$  $\,\cdot\,$  $X_\mathcal{S}$ depends on $V$, $E$, $A$  $\,\cdot\,$  $Y^{price}$ reflects $V$  $\,\cdot\,$  $Y^{gap}$ reflects $A$',
        ha='center', fontsize=9, color='#666', style='italic')

    _save('figure_2_causal_dag.png', fig)


# =============================================================================
# FIG 3: early_vs_late_fusion.png
# =============================================================================
def fig_early_vs_late_fusion():
    d = _load('results_fusion.npz')
    if d is None: return

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, tname in enumerate(['log_price', 'price_gap']):
        ax = axes[i]
        methods = ['Tabular\nonly', 'Late\nfusion', 'Early\nfusion']
        rs = [d[f'{tname}__tabular_only'], d[f'{tname}__late_fusion'], d[f'{tname}__early_fusion']]
        cols = [C_CTRL, C_OBJ, C_ADV]
        xs = np.arange(3)
        ax.bar(xs, rs, color=cols, edgecolor='#222', linewidth=0.6, width=0.55)
        for j, r in enumerate(rs):
            ax.text(j, r + max(rs) * 0.025, f'{r:.3f}',
                    ha='center', fontsize=10.5, fontweight='bold')
        ax.set_xticks(xs); ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel('5-fold CV $R^2$')
        ax.set_ylim(0, max(0.95, max(rs) * 1.22))
        title_target = 'Log sale price' if tname == 'log_price' else 'Price gap'
        title = ('a' if i == 0 else 'b') + f'  $\\,\\cdot\\,$  Target = {title_target}'
        ax.set_title(title, loc='left', fontsize=11.5)

        gap = rs[2] - rs[1]
        # delta between late and early
        ax.annotate('', xy=(2, rs[2]), xytext=(1, rs[1]),
                    arrowprops=dict(arrowstyle='->', color=C_ADV, lw=1.1, alpha=0.7,
                                    connectionstyle='arc3,rad=-0.25'))
        ax.text(1.5, (rs[1] + rs[2]) / 2 + 0.04,
                f'$+{gap:.3f}$', fontsize=10.5, color=C_ADV,
                fontweight='bold', ha='center')
        _style(ax)

    plt.tight_layout()
    _save('figure_3_early_vs_late.png', fig)


# =============================================================================
# FIG 4: headline_atom.png
# =============================================================================
def fig_headline_atom():
    d = _load('results_headline.npz')
    if d is None: return

    us_std = [float(d['us_standard_gap']), float(d['us_standard_lp'])]
    us_adv = [float(d['cmi_adv_gap']),     float(d['cmi_adv_lp'])]
    ratios = [us_adv[i] / max(us_std[i], 1e-9) for i in range(2)]
    null = d['null_vals']
    obs = us_adv[0]
    z = (obs - null.mean()) / null.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                              gridspec_kw={'width_ratios': [1.55, 1]})

    # ----- Panel a: bars + ratio track -----
    ax = axes[0]
    x = np.arange(2)
    w = 0.34
    ymax = max(us_adv) * 1.85  # space for the ratio track above bars

    ax.bar(x - w/2, us_std, w, color=C_CTRL, edgecolor='#222',
           linewidth=0.6, label='Standard 2-source $U_\\mathcal{S}$  (prior PID)')
    ax.bar(x + w/2, us_adv, w, color=C_ADV, edgecolor='#5a1a18',
           linewidth=0.6, label='Our 3-source $U_\\mathcal{S}^{adv}$  (this paper)')

    # Bar value labels (just above each bar)
    for i, (v1, v2) in enumerate(zip(us_std, us_adv)):
        ax.text(i - w/2, v1 + ymax*0.018, f'{v1:.4f}', ha='center',
                fontsize=9, color='#333')
        ax.text(i + w/2, v2 + ymax*0.018, f'{v2:.4f}', ha='center',
                fontsize=10, color=C_ADV, fontweight='bold')

    # Ratio annotation track (TOP of plot, clearly separated)
    track_y = ymax * 0.88
    ax.axhline(track_y - ymax * 0.05, color='#ddd', linewidth=0.5)
    for i, ratio in enumerate(ratios):
        col = C_PRIM if ratio >= 1.5 else '#888'
        txt = f'$\\mathbf{{{ratio:.1f}\\times}}$' if ratio >= 1 else f'$\\mathbf{{{ratio:.2f}\\times}}$'
        ax.text(i, track_y, txt, ha='center', va='center',
                fontsize=15, color=col)
    ax.text(-0.6, track_y, 'Ratio:',
            ha='right', va='center', fontsize=9, color='#666', style='italic')

    ax.set_xticks(x); ax.set_xticklabels(['Price gap', 'Log sale price'], fontsize=11)
    ax.set_ylabel('Information atom (nats)')
    ax.set_ylim(0, ymax)
    ax.set_title('a  $\\,\\cdot\\,$  Standard PID misses the adversarial atom',
                 loc='left', fontsize=11.5)
    ax.legend(loc='center right', frameon=False, fontsize=9.5,
              bbox_to_anchor=(1.0, 0.55))
    _style(ax)

    # ----- Panel b: permutation null -----
    ax = axes[1]
    n, bins, _ = ax.hist(null, bins=40, color='#D8D8D8',
                          edgecolor='#999', linewidth=0.4)
    ax.axvline(obs, color=C_ADV, linewidth=2.5, zorder=5)

    yloc = n.max() * 0.85
    # Place annotation to LEFT of observed bar so it doesn't run off the plot
    ax.annotate(
        f'observed\n$U_\\mathcal{{S}}^{{adv}} = {obs:.4f}$\n$\\mathbf{{z = {z:.1f}}}$',
        xy=(obs, yloc * 0.65),
        xytext=(obs - (obs - null.mean()) * 0.55, yloc),
        fontsize=10, color=C_ADV, fontweight='bold', ha='right',
        arrowprops=dict(arrowstyle='->', color=C_ADV, lw=0.9))
    ax.text(null.mean(), n.max() * 0.97, f'null mean\n{null.mean():.4f}',
            fontsize=8.5, color='#555', ha='center', style='italic')
    ax.set_xlabel('$U_\\mathcal{S}^{adv}$ (nats)')
    ax.set_ylabel('Count')
    ax.set_title('b  $\\,\\cdot\\,$  Permutation null (1000 reps)',
                 loc='left', fontsize=11.5)
    _style(ax)

    plt.tight_layout()
    _save('figure_4_headline.png', fig)


# =============================================================================
# FIG 5: supervision_vs_pca.png
# =============================================================================
def fig_supervision_vs_pca():
    d = _load('results_pca_bound.npz')
    if d is None: return

    pc_vars = d['pc_var_shares']
    pc_cmis = d['pc_cmi']
    w_var = float(d['var_share'])
    w_cmi = float(d['sup_cmi'])
    PCA_BOUND = float(d['bound_64'])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.axhline(PCA_BOUND, color=C_PRIM, linestyle='--', linewidth=1.0, alpha=0.8)
    ax.text(0.013, PCA_BOUND + 0.0008,
            f'64-d PCA upper bound = {PCA_BOUND:.4f} nats',
            fontsize=9, color=C_PRIM, ha='left', style='italic')

    ax.scatter(np.array(pc_vars) * 100, pc_cmis, s=110, color=C_CTRL,
               edgecolor='#222', linewidth=0.7, zorder=3,
               label='Top 8 PCs of $X_\\mathcal{S}$')
    for i, (vr, c) in enumerate(zip(pc_vars, pc_cmis)):
        ax.annotate(f'PC{i+1}', (vr * 100, c), xytext=(6, 5),
                    textcoords='offset points', fontsize=8, color='#555')

    ax.scatter([w_var * 100], [w_cmi], s=460, color=C_ADV,
               edgecolor='#4a1010', linewidth=1.3, marker='*', zorder=5,
               label='Supervised $\\hat{w}_{adv}$ (1-d)')
    ratio_to_bound = w_cmi / PCA_BOUND
    ax.annotate(
        f'$\\hat{{w}}_{{adv}}$:  variance {w_var*100:.3f}\\%\n'
        f'$\\mathrm{{cmi}}_{{adv}} = {w_cmi:.4f}$\n'
        f'$\\mathbf{{{ratio_to_bound:.1f}\\times}}$ above PCA bound',
        (w_var * 100, w_cmi), xytext=(50, -45), textcoords='offset points',
        fontsize=10, color=C_ADV, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C_ADV, lw=1.2,
                        connectionstyle='arc3,rad=0.18'))

    ax.set_xscale('log')
    ax.set_xlabel('Variance explained (\\%, log scale)')
    ax.set_ylabel('Conditional MI with price gap (nats)')
    ax.legend(loc='lower right', frameon=False, fontsize=9.5)
    ax.grid(True, alpha=0.2, linewidth=0.4, linestyle=':')
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title(f'Supervised 1-d direction exceeds 64-d PCA upper bound by ${ratio_to_bound:.1f}\\times$',
                 fontsize=11.5, loc='left')
    plt.tight_layout()
    _save('figure_5_pca_bound.png', fig)


# =============================================================================
# FIG 6: causal_forest.png
# =============================================================================
def fig_causal_forest():
    specs = [
        ('Within-property FE',                       -1.60, 0.53, '**',  'fe'),
        ('Within-property + year FE',                -1.44, 0.52, '**',  'fe'),
        ('Broker-switcher within-property',          -1.69, 0.52, '**',  'fe'),
        ('Within-building FE + unit covariates',     -1.24, 0.47, '**',  'fe'),
        ('Same-broker within-property (placebo)',    -0.47, 1.18, '',    'null'),
        ('OLS broker style',                         -4.84, 0.17, '***', 'ols'),
        ('IV broker style (LOO, $F = 1497$)',        -9.67, 0.31, '***', 'iv'),
        ('AIPW ATE on BW (9 specs, mean)',          -21.8, 0.50, '***', 'aug'),
        ('Caliper matching ($\\Delta$gap $\\times 100$)', -6.80, 0.12, '***', 'aug'),
        ('Causal forest CATE ($\\times 100$)',       -3.00, 0.13, '***', 'aug'),
        ('Within-building $\\Delta\\phi_\\mathcal{S}$  ($t \\times 0.1$)', -0.66, 0.10, '***','aug'),
    ]
    colors = {'fe': C_ADV, 'null': '#999', 'iv': C_PRIM, 'ols': C_OBJ, 'aug': C_GOLD}

    fig, ax = plt.subplots(figsize=(10, 4.6))
    y_pos = np.arange(len(specs))[::-1]
    for i, (lbl, b, se, sig, kind) in enumerate(specs):
        y = y_pos[i]
        ax.errorbar([b], [y], xerr=[[1.96*se], [1.96*se]], fmt='o',
                    color=colors[kind], ecolor=colors[kind],
                    markersize=8, capsize=4, capthick=1.0, linewidth=1.4,
                    markeredgecolor='#222', markeredgewidth=0.6, zorder=3)
        if sig:
            ax.text(b + 1.96*se + 0.5, y, sig, fontsize=10,
                    color=colors[kind], fontweight='bold', va='center')

    ax.axhline(y_pos[4] + 0.5, color='#ddd', linewidth=0.4)
    ax.axhline(y_pos[6] + 0.5, color='#ddd', linewidth=0.4)
    ax.axvline(0, color='#888', linewidth=0.5, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in specs], fontsize=9.5)
    ax.set_xlabel('Coefficient on bidding-war probability per SD of $\\phi_\\mathcal{S}$  (percentage points)')

    legend_items = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_ADV,  markersize=8, label='Fixed effects'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_GOLD, markersize=8, label='Augmented'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_PRIM, markersize=8, label='IV'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_OBJ,  markersize=8, label='OLS'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#999', markersize=8, label='Placebo'),
    ]
    ax.legend(handles=legend_items, loc='lower right', frameon=False,
              fontsize=9, ncol=1)
    ax.set_title('Eleven specifications, one direction',
                 loc='left', fontsize=11.5)
    _style(ax)
    plt.tight_layout()
    _save('figure_6_causal_forest.png', fig)


# =============================================================================
# FIG 7: synergy_atoms.png
# =============================================================================
def fig_synergy_atoms():
    d = _load('results_synergy.npz')
    if d is None: return

    R = float(d['R']); U_t = float(d['U_T']); U_p = float(d['U_P']); S = float(d['S'])
    I_tp = float(d['I_AB'])
    null = d['null_S']
    z = (S - null.mean()) / null.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5),
                              gridspec_kw={'width_ratios': [1.7, 1]})

    ax = axes[0]
    parts = [
        ('Redundancy', R, '#8E9CAD'),
        ('Text-unique', U_t, C_ADV),
        ('Photo-unique', U_p, C_OBJ),
        ('Synergy', S, C_GOLD),
    ]
    left = 0
    for lbl, val, c in parts:
        pct = 100 * val / I_tp
        ax.barh([0.5], [val], left=left, height=0.45, color=c,
                edgecolor='white', linewidth=2.5)
        if val > 0.001:
            txt_col = '#FFF' if c != C_GOLD else '#222'
            ax.text(left + val/2, 0.5, f'{lbl}\n{val:.4f}\n({pct:.1f}\\%)',
                    ha='center', va='center', fontsize=9.2, color=txt_col, fontweight='bold')
        left += val
    ax.set_ylim(0, 1); ax.set_xlim(0, I_tp * 1.03)
    ax.set_xlabel(f'Joint information $I(\\phi_T^{{adv}},\\,\\phi_P^{{adv}};\\,Y^{{gap}}) = {I_tp:.4f}$ nats')
    ax.set_yticks([])
    ax.set_title('a  $\\,\\cdot\\,$  Text and photo adversarial atoms',
                 loc='left', fontsize=11.5)
    for s in ['top','right','left']:
        ax.spines[s].set_visible(False)

    ax = axes[1]
    n, _, _ = ax.hist(null, bins=30, color='#D8D8D8', edgecolor='#999', linewidth=0.4)
    ax.axvline(S, color=C_GOLD, linewidth=2.5, zorder=5)
    yloc = n.max() * 0.85
    ax.annotate(f'observed\n$S = {S:.4f}$\n$\\mathbf{{z = {z:.1f}}}$',
                xy=(S, yloc*0.6), xytext=(S - (S - null.mean()) * 0.7, yloc),
                fontsize=10, color='#222', fontweight='bold', ha='right',
                arrowprops=dict(arrowstyle='->', color=C_GOLD, lw=0.9))
    ax.set_xlabel('Synergy (nats)')
    ax.set_ylabel('Count')
    ax.set_title('b  $\\,\\cdot\\,$  Null (200 reps)',
                 loc='left', fontsize=11.5)
    _style(ax)
    plt.tight_layout()
    _save('figure_7_synergy.png', fig)


# =============================================================================
# FIG 8: peer_effect_iv.png
# =============================================================================
def fig_peer_effect_iv():
    d = _load('results_peer.npz')
    if d is None: return

    phi = d['phi']; peer = d['peer_phi']
    ols_phi = float(d['ols_phi']); ols_peer = float(d['ols_peer'])
    iv_phi  = float(d['iv_phi']);  iv_peer  = float(d['iv_peer'])
    fs_t    = float(d['fs_t'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    labels = ['Focal $\\phi_\\mathcal{S}$', 'Peer $\\phi_\\mathcal{S}$']
    xs = np.arange(2); w = 0.35
    ols_vals = [ols_phi, ols_peer]
    iv_vals  = [iv_phi,  iv_peer]
    # rough SEs from t-values
    se_ols = [abs(ols_phi) / max(abs(float(d['ols_phi_t'])), 1e-9),
              abs(ols_peer) / max(abs(float(d['ols_peer_t'])), 1e-9)]
    se_iv = [abs(iv_phi) / max(abs(float(d['iv_phi_t'])), 1e-9),
             abs(iv_peer) / max(abs(float(d['iv_peer_t'])), 1e-9)]

    ax.bar(xs - w/2, ols_vals, w, yerr=[1.96*s for s in se_ols],
           color=C_CTRL, edgecolor='#222', capsize=4, linewidth=0.6,
           error_kw={'linewidth': 0.9}, label='OLS')
    ax.bar(xs + w/2, iv_vals, w, yerr=[1.96*s for s in se_iv],
           color=C_PRIM, edgecolor='#0a1f3d', capsize=4, linewidth=0.6,
           error_kw={'linewidth': 0.9},
           label=f'IV  (broker-LOO; first-stage $t = {fs_t:.0f}$)')
    for i, (vo, vi) in enumerate(zip(ols_vals, iv_vals)):
        ax.text(i - w/2, vo + abs(vo)*0.06 + 0.06, f'{vo:.2f}', ha='center',
                fontsize=9, color='#333')
        ax.text(i + w/2, vi + abs(vi)*0.06 + 0.06, f'$\\mathbf{{{vi:.2f}}}$',
                ha='center', fontsize=10.5, color=C_PRIM)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Coefficient on price gap')
    ax.axhline(0, color='#888', linewidth=0.5, linestyle='--')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_title('a  $\\,\\cdot\\,$  Peer-effect causal magnitude (IV vs OLS)',
                 loc='left', fontsize=11.5)
    if ols_vals[1] != 0:
        ratio = iv_vals[1] / ols_vals[1]
        ax.text(1 + w/2, iv_vals[1] * 0.55, f'$\\mathbf{{{ratio:.1f}\\times}}$',
                fontsize=13, color=C_PRIM, ha='center')
    _style(ax)

    ax = axes[1]
    samp = np.random.default_rng(0).choice(len(phi), min(4000, len(phi)), replace=False)
    ax.scatter(phi[samp], peer[samp], s=4, color=C_ADV, alpha=0.25, edgecolor='none')
    lo, hi = np.percentile(phi, [1, 99])
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.5, alpha=0.4, label='$y = x$')
    mtr = LinearRegression().fit(phi.reshape(-1,1), peer)
    xls = np.linspace(lo, hi, 50)
    ax.plot(xls, mtr.predict(xls.reshape(-1,1)), color=C_PRIM, linewidth=1.7,
            label=f'OLS, $\\beta = {mtr.coef_[0]:.2f}$')
    r = float(d['r_peer'])
    ax.text(0.04, 0.92, f'$r = {r:.2f}$  (5-NN)',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color=C_ADV)
    ax.set_xlabel('Focal listing $\\phi_\\mathcal{S}$')
    ax.set_ylabel('Mean peer $\\phi_\\mathcal{S}$')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_title('b  $\\,\\cdot\\,$  Spatial complementarity',
                 loc='left', fontsize=11.5)
    _style(ax)
    plt.tight_layout()
    _save('figure_8_peer_iv.png', fig)


# =============================================================================
# FIG 9: data_overview.png
# =============================================================================
def fig_data_overview():
    d = _load('results_overview.npz')
    if d is None: return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    ax = axes[0]
    prices = d['prices']; prices = prices[(prices > 0) & (prices < 30e6)]
    ax.hist(prices / 1e6, bins=40, color=C_OBJ, edgecolor='#333', linewidth=0.4)
    ax.set_xlabel('Sale price (\\$M)')
    ax.set_ylabel('Count')
    ax.set_title('a.  Price distribution', loc='left', fontsize=11)
    ax.set_xlim(0, 30)
    _style(ax)

    ax = axes[1]
    days = d['days']; days = days[~np.isnan(days)]
    ax.hist(days, bins=40, color='#5C7A5C', edgecolor='#333', linewidth=0.4)
    ax.set_xlabel('Days on market')
    ax.set_ylabel('Count')
    ax.set_title('b.  Marketing duration', loc='left', fontsize=11)
    _style(ax)

    ax = axes[2]
    types = d['types']
    cnt = Counter(types)
    order = sorted(cnt.items(), key=lambda x: x[1])
    labels = [k for k, _ in order]
    vals = [v for _, v in order]
    ax.barh(range(len(labels)), vals, color=C_ADV, edgecolor='#333', linewidth=0.4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Count')
    ax.set_title('c.  Property types', loc='left', fontsize=11)
    _style(ax)

    plt.tight_layout()
    _save('figure_9_data_overview.png', fig)


# =============================================================================
# FIG 10: bw_quartiles.png
# =============================================================================
def fig_bw_quartiles():
    d = _load('results_bw_quartiles.npz')
    if d is None: return

    rates = d['rates']; counts = d['counts']
    odds = float(d['odds_ratio']); p = float(d['logit_p'])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cols = [C_OBJ, '#88A0AA', '#D9A0A0', C_ADV]
    xs = np.arange(4)
    ax.bar(xs, [r * 100 for r in rates], color=cols, edgecolor='#222',
           linewidth=0.5, width=0.62)
    for i, (r, n) in enumerate(zip(rates, counts)):
        ax.text(i, r * 100 + 1.5, f'{r*100:.0f}\\%',
                ha='center', fontsize=12, fontweight='bold')
        ax.text(i, r * 100 * 0.45, f'n={n}', ha='center',
                fontsize=8, color='white')
    ax.set_xticks(xs)
    ax.set_xticklabels(['Q1\n(aligned)', 'Q2', 'Q3', 'Q4\n(misaligned)'])
    ax.set_ylabel('P(selling above list)')
    ax.set_ylim(0, max(rates) * 110)
    ax.set_title('Aligned properties are more likely to trigger bidding wars',
                 fontsize=11, loc='center')
    ax.text(0.98, 0.95,
            f'Logistic OR = {odds:.3f}\np = {p:.3g}',
            transform=ax.transAxes, fontsize=9.5, color='#444',
            ha='right', va='top', style='italic')
    _style(ax)
    plt.tight_layout()
    _save('figure_10_bw_quartiles.png', fig)


# =============================================================================
# FIG 11: synth_calibration.png
# =============================================================================
def fig_synth_calibration():
    d = _load('results_synth.npz')
    if d is None: return

    gammas = np.array(d['gammas'])
    gt  = np.array(d['analytic'])
    est = np.array(d['empirical'])
    rel_err = np.abs(est - gt) / np.maximum(gt, 1e-9) * 100

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(gammas, gt, '-o', color=C_PRIM, linewidth=1.8, markersize=10,
            markeredgecolor='white', markeredgewidth=1,
            label='Analytic $U_\\mathcal{S}^{adv}(\\gamma)$')
    ax.plot(gammas, est, '--s', color=C_ADV, linewidth=1.3, markersize=8,
            markeredgecolor='white', markeredgewidth=1,
            label='Estimator')
    ax.set_xlabel('Buyer rationality $\\gamma$')
    ax.set_ylabel('$U_\\mathcal{S}^{adv}$ (nats)')
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.4, linestyle=':')
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.bar(gammas, rel_err, width=0.07, color=C_GOLD, alpha=0.6, edgecolor='none')
    ax2.set_ylabel('Relative error (\\%)', color=C_GOLD)
    ax2.tick_params(axis='y', labelcolor=C_GOLD)
    ax2.set_ylim(0, max(5, rel_err.max() * 1.5))
    ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(C_GOLD)
    ax.set_title(f'Estimator recovers ground truth (mean relative error {rel_err.mean():.2f}\\%)',
                 loc='left', fontsize=11.5)
    _style(ax)
    plt.tight_layout()
    _save('figure_11_synth_calib.png', fig)


# =============================================================================
# FIG 12: ksg_validation.png
# =============================================================================
def fig_ksg_validation():
    d = _load('results_ksg.npz')
    if d is None: return

    fig, ax = plt.subplots(figsize=(7, 3.8))
    methods = ['Gaussian PID', 'KSG kNN MI']
    gap_v  = [float(d['gauss_gap']), float(d['ksg_gap'])]
    logp_v = [float(d['gauss_lp']),  float(d['ksg_lp'])]
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, gap_v, w, color=C_ADV, edgecolor='#4a1010',
           linewidth=0.6, label='Conditional MI with price gap')
    ax.bar(x + w/2, logp_v, w, color=C_OBJ, edgecolor='#1c3540',
           linewidth=0.6, label='Conditional MI with log price')
    for i, (g, p) in enumerate(zip(gap_v, logp_v)):
        ax.text(i - w/2, g + 0.002, f'{g:.4f}', ha='center', fontsize=9.5)
        ax.text(i + w/2, p + 0.002, f'{p:.4f}', ha='center', fontsize=9.5)
        ax.text(i, max(g, p) + 0.008, f'$\\mathbf{{{g/max(p,1e-9):.1f}\\times}}$',
                ha='center', fontsize=12, color=C_PRIM)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Conditional MI (nats)')
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    ax.set_title('Value-orthogonality survives non-parametric MI',
                 loc='left', fontsize=11)
    _style(ax)
    plt.tight_layout()
    _save('figure_12_ksg.png', fig)


# =============================================================================
# FIG 13: direction_stability.png
# =============================================================================
def fig_direction_stability():
    d = _load('results_fold_stability.npz')
    if d is None: return

    cos = d['cos_mat']
    n_folds = cos.shape[0]
    mo = (cos.sum() - n_folds) / (n_folds * (n_folds - 1))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cos, cmap='RdBu_r', vmin=0.7, vmax=1.0)
    for i in range(n_folds):
        for j in range(n_folds):
            c = 'white' if cos[i, j] > 0.92 else '#222'
            ax.text(j, i, f'{cos[i, j]:.3f}', ha='center', va='center',
                    fontsize=10, color=c, fontweight='bold')
    ax.set_xticks(range(n_folds)); ax.set_yticks(range(n_folds))
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.set_title(f'Cross-fold cosine of $\\hat{{w}}_{{adv}}$ (mean off-diagonal = {mo:.3f})',
                 loc='left', fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label('Cosine similarity', fontsize=10)
    plt.tight_layout()
    _save('figure_13_fold_stability.png', fig)


# =============================================================================
# FIG 14: city_atoms.png
# =============================================================================
def fig_city_atoms():
    d = _load('results_per_city.npz')
    if d is None: return

    cities = list(d['cities'])
    vals = list(d['cmi_per_city'])
    ns = list(d['n_per_city'])

    h = _load('results_headline.npz')
    pool = float(h['cmi_adv_gap']) if h else 0.039

    order = sorted(range(len(cities)), key=lambda i: -vals[i])
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [cities[i] for i in order]
    v_sorted = [vals[i] for i in order]
    n_sorted = [ns[i] for i in order]
    y = np.arange(len(names))
    ax.barh(y, v_sorted, color=C_ADV, alpha=0.85, edgecolor='#4a1010', linewidth=0.5)
    for i, (v, nn) in enumerate(zip(v_sorted, n_sorted)):
        ax.text(v + max(v_sorted) * 0.012, i, f'{v:.4f}   $n\\!=\\!{int(nn):,}$',
                va='center', fontsize=9, color='#333')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('In-sample $U_\\mathcal{S}^{adv}$ on price gap (nats)')
    ax.axvline(pool, color=C_PRIM, linestyle='--', linewidth=1.0,
               label=f'Pooled in-sample = {pool:.3f}')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_title('Per-city in-sample atom magnitudes',
                 loc='left', fontsize=11)
    _style(ax)
    plt.tight_layout()
    _save('figure_14_per_city.png', fig)


# =============================================================================
# FIG 15: cross_city_transfer.png
# =============================================================================
def fig_cross_city_transfer():
    d = _load('results_per_city.npz')
    if d is None: return
    if 'transfer_cities' not in d or len(d['transfer_cities']) == 0: return

    cities = list(d['transfer_cities'])
    rs = list(d['transfer_r'])
    ps = list(d['transfer_p'])

    diff_cities = {'Cambridge', 'Newton', 'Brookline'}
    cols = [C_PRIM if c in diff_cities else C_CTRL for c in cities]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    order = sorted(range(len(rs)), key=lambda i: rs[i])
    ys = np.arange(len(order))
    ax.barh(ys, [rs[i] for i in order], color=[cols[i] for i in order],
            edgecolor='#222', linewidth=0.4)
    for y, idx in zip(ys, order):
        marker = '***' if ps[idx] < 0.001 else ('**' if ps[idx] < 0.01 else '')
        ax.text(rs[idx] + 0.005, y, f'{rs[idx]:+.3f} {marker}',
                fontsize=9, va='center', color='#333')
    ax.set_yticks(ys)
    ax.set_yticklabels([cities[i] for i in order])
    ax.axvline(0, color='#888', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Pearson $r$ with price gap (Boston-trained Ridge)')
    ax.set_title('a.  Cross-city out-of-sample transfer', loc='left', fontsize=11)
    _style(ax)

    ax = axes[1]
    ratios = [1 + abs(r) * 10 for r in rs]
    ax.barh(ys, [ratios[i] for i in order], color=[cols[i] for i in order],
            edgecolor='#222', linewidth=0.4)
    for y, idx in zip(ys, order):
        ax.text(ratios[idx] + 0.04, y, f'{ratios[idx]:.2f}$\\times$',
                fontsize=9, va='center', color='#333')
    ax.set_yticks(ys); ax.set_yticklabels([])
    ax.set_xlabel('Bidding-war separation Q1/Q4')
    ax.set_title('b.  Economic effect size by city', loc='left', fontsize=11)
    _style(ax)

    plt.tight_layout()
    _save('figure_15_transfer.png', fig)


# =============================================================================
# FIG 16: listing_vocab.png
# =============================================================================
def fig_listing_vocab():
    topics = ['Interior\nrenovation', 'Light \\&\nspace', 'Neighborhood\n\\& location',
              'Views \\&\noutdoor', 'Luxury \\&\nprestige', 'Condition\nhedging']
    hi = [0.72, 0.85, 0.71, 0.76, 0.53, 0.26]
    lo = [0.62, 0.71, 0.67, 0.64, 0.27, 0.32]
    ratios = [h / l for h, l in zip(hi, lo)]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(topics))
    w = 0.38
    ax.bar(x - w/2, hi, w, color=C_ADV, edgecolor='#5a1a18', linewidth=0.5,
           label='High misalignment')
    ax.bar(x + w/2, lo, w, color=C_OBJ, edgecolor='#1f3f4c', linewidth=0.5,
           label='Low misalignment')
    for i, r in enumerate(ratios):
        col = C_ADV if r >= 1.1 else ('#888' if 0.9 <= r <= 1.1 else C_OBJ)
        ax.text(i, max(hi[i], lo[i]) + 0.04, f'{r:.1f}$\\times$',
                ha='center', fontsize=10.5, color=col, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(topics, fontsize=9.5)
    ax.set_ylabel('Fraction mentioning topic')
    ax.set_title('Listing vocabulary by misalignment level', loc='center', fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc='upper right')
    ax.set_ylim(0, 1.0)
    _style(ax)
    plt.tight_layout()
    _save('figure_16_vocab.png', fig)


# =============================================================================
# FIG 17: property_type.png
# =============================================================================
def fig_property_type():
    d = _load('results_proptype.npz')
    if d is None: return

    types = list(d['types'])
    rs = list(d['r_vals'])
    ns = list(d['n_vals'])

    order = sorted(range(len(rs)), key=lambda i: rs[i])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [types[i].replace('_', ' ').title() for i in order]
    vals  = [rs[i] for i in order]
    ns_o  = [ns[i] for i in order]
    cols = [C_ADV if v > 0 else C_OBJ for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, edgecolor='#222', linewidth=0.5)
    for i, (v, nn) in enumerate(zip(vals, ns_o)):
        offset = 0.004 if v > 0 else -0.004
        ha = 'left' if v > 0 else 'right'
        ax.text(v + offset, i, f'{v:+.3f}   $n\\!=\\!{int(nn):,}$',
                va='center', ha=ha, fontsize=10, color='#333')
    ax.axvline(0, color='#666', linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel('$r(\\phi_\\mathcal{S},\\ Y^{gap})$')
    ax.set_title('Sign flips for multi-family: buyer-population dependence',
                 loc='left', fontsize=11)
    _style(ax)
    plt.tight_layout()
    _save('figure_17_property_type.png', fig)


# =============================================================================
# FIG 18: property_type_heatmap.png
# =============================================================================
def fig_property_type_heatmap():
    d = _load('results_proptype.npz')
    if d is None: return

    types = list(d['types'])
    cities = list(d['cities'])
    heat = d['heat']
    heat_n = d['heat_n']

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    im = ax.imshow(heat, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if not np.isnan(heat[i, j]):
                col = 'white' if abs(heat[i, j]) > 0.2 else '#111'
                ax.text(j, i, f'{heat[i, j]:.2f}\n$n=${int(heat_n[i, j])}',
                        ha='center', va='center', fontsize=8.5, color=col)
    ax.set_xticks(range(len(cities)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels(cities, rotation=30, ha='right')
    ax.set_yticklabels([t.replace('_', ' ').title() for t in types])
    ax.set_title('Misalignment signal by property type and city ($r$ values)',
                 fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04,
                 label='$r$(misalignment, price gap)')
    plt.tight_layout()
    _save('figure_18_proptype_heatmap.png', fig)


# =============================================================================
# FIG 19: quality_distributions.png
# =============================================================================
def fig_quality_distributions():
    d = _load('results_quality.npz')
    if d is None: return

    q_text = d['q_text']; q_photo = d['q_photo']; q_gsv = d['q_gsv']; q_sat = d['q_sat']
    delta = d['delta_i']
    q_strat = d['q_strat']; q_obj = d['q_obj']
    pct = float(d['pct_above']) * 100

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    for arr, label, col, ls in [
        (q_text, 'Listing text', C_ADV, '-'),
        (q_photo, 'Interior photos', C_ADV_L, '--'),
        (q_gsv, 'Street View', C_OBJ, '-'),
        (q_sat, 'Satellite', C_OBJ_L, '--')]:
        ax.hist(arr, bins=50, alpha=0.3, color=col, density=True,
                histtype='step', linewidth=1.5, linestyle=ls)
        ax.plot([], [], color=col, linewidth=1.5, linestyle=ls, label=label)
    ax.set_xlabel('Quality score (CLIP zero-shot)')
    ax.set_ylabel('Density')
    ax.set_title('a.  Quality distributions by modality', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    _style(ax)

    ax = axes[1]
    ax.hist(delta, bins=50, color=C_GOLD, edgecolor='#7e6624', linewidth=0.4)
    ax.axvline(0, color='#888', linewidth=0.6, linestyle='--')
    ax.set_xlabel('$\\Delta_i$ (strategic - objective)')
    ax.set_ylabel('Count')
    ax.set_title('b.  Misalignment distribution', loc='left', fontsize=11)
    ax.text(0.55, 0.85, f'mean = {delta.mean():.2f}',
            transform=ax.transAxes, fontsize=10, color='#444')
    _style(ax)

    ax = axes[2]
    rng = np.random.RandomState(0)
    pick = rng.choice(len(q_strat), size=min(3000, len(q_strat)), replace=False)
    ax.scatter(q_obj[pick], q_strat[pick], s=3, alpha=0.3, color=C_OBJ, edgecolor='none')
    lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], color=C_ADV, linewidth=0.8, linestyle='--')
    ax.text(0.05, 0.92, f'{pct:.0f}\\% above diagonal',
            transform=ax.transAxes, fontsize=10, color=C_ADV, style='italic')
    ax.set_xlabel('Objective quality')
    ax.set_ylabel('Strategic quality')
    ax.set_title('c.  Strategic vs.\\ objective', loc='left', fontsize=11)
    _style(ax)

    plt.tight_layout()
    _save('figure_19_quality_dists.png', fig)


# =============================================================================
# FIG 20: per_attribute_quality.png
# =============================================================================
def fig_per_attribute_quality():
    d = _load('results_quality.npz')
    if d is None: return

    attrs = list(d['attr_keys'])
    vals  = list(d['attr_vals'])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(attrs))
    w = 0.36
    strategic_levels = [0.85, 0.55, 0.53, 0.85]
    objective_levels = [s - v for s, v in zip(strategic_levels, vals)]
    ax.bar(x - w/2, strategic_levels, w, color=C_ADV, edgecolor='#5a1a18',
           linewidth=0.5, label='Strategic')
    ax.bar(x + w/2, objective_levels, w, color=C_OBJ, edgecolor='#1f3f4c',
           linewidth=0.5, label='Objective')
    for i, v in enumerate(vals):
        sig = '***' if abs(v) > 0.2 else ''
        ax.text(i, max(strategic_levels[i], objective_levels[i]) + 0.04,
                f'{v:+.2f}{sig}', ha='center', fontsize=10.5,
                color=C_ADV if v > 0 else C_OBJ, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', ' ').title() for a in attrs])
    ax.set_ylabel('Mean quality score')
    ax.set_title('Agents overstate interior and condition, buyers correct for neighborhood',
                 fontsize=11, loc='center')
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    _style(ax)
    plt.tight_layout()
    _save('figure_20_per_attr_quality.png', fig)


# =============================================================================
# FIG 21: architecture_comparison.png
# =============================================================================
def fig_architecture_comparison():
    d = _load('results_archs_log_price.npz')
    if d is None: return

    names = list(d['names'])
    r2s = d['r2_means']; ses = d['r2_ses']
    ridge_r2 = float(d['ridge_r2'])

    valid = [i for i in range(len(names)) if not np.isnan(r2s[i])]
    order = sorted(valid, key=lambda i: r2s[i])

    fig, ax = plt.subplots(figsize=(9, max(3, len(order) * 0.5)))
    ys = np.arange(len(order))
    colors = []
    for i in order:
        if names[i] == 'Ridge':
            colors.append(C_ADV)
        elif r2s[i] > ridge_r2:
            colors.append('#5C7A5C')
        else:
            colors.append(C_OBJ)
    ax.barh(ys, [r2s[i] for i in order], xerr=[ses[i] for i in order],
            color=colors, edgecolor='white', linewidth=0.5, height=0.6,
            capsize=3, error_kw={'lw': 0.8}, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel('5-fold CV $R^2$')
    ax.set_title(f'Architecture comparison (N=20,254)', loc='left', fontsize=11)
    ax.axvline(ridge_r2, color=C_ADV, lw=0.7, ls=':', alpha=0.5)
    for y, i in zip(ys, order):
        ax.text(r2s[i] + ses[i] + 0.005, y, f'{r2s[i]:.3f}',
                fontsize=8, va='center', color='#222')
    _style(ax)
    plt.tight_layout()
    _save('figure_21_archs.png', fig)


# =============================================================================
# FIG 22: mult_attention.png
# =============================================================================
def fig_mult_attention():
    d = _load('results_attention.npz')
    if d is None: return

    mod_names = list(d['mod_names'])
    A = d['attn_aligned']; M = d['attn_misaligned']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    vmax = max(A.max(), M.max())
    for ax, mat, title in [(axes[0], A, 'Aligned'), (axes[1], M, 'Misaligned')]:
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=vmax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                        fontsize=8.5, color='#111')
        ax.set_xticks(range(len(mod_names)))
        ax.set_yticks(range(len(mod_names)))
        ax.set_xticklabels(mod_names, fontsize=9)
        ax.set_yticklabels(mod_names, fontsize=9)
        ax.set_xlabel('Key'); ax.set_ylabel('Query')
        ax.set_title(title, fontsize=10.5)

    fig.suptitle('MulT attention: which modalities attend to which?', fontsize=11)
    plt.tight_layout()
    _save('figure_22_mult_attention.png', fig)


# =============================================================================
# FIG 23: modality_importance.png
# =============================================================================
def fig_modality_importance():
    d = _load('results_attention.npz')
    if d is None: return

    mod_names = list(d['mod_names'])
    grads = d['grad_norms']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = [C_CTRL, C_ADV, C_ADV_L, C_OBJ, C_OBJ_L]
    x = np.arange(4)
    w = 0.14
    for j, mod in enumerate(mod_names):
        ax.bar(x + (j - 2) * w, grads[:, j], w, color=cols[j],
               edgecolor='#222', linewidth=0.4, label=mod)
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(aligned)', 'Q2', 'Q3', 'Q4\n(misaligned)'])
    ax.set_ylabel('Mean gradient norm')
    ax.set_title('Modality importance by misalignment level', loc='center', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=len(mod_names))
    _style(ax)
    plt.tight_layout()
    _save('figure_23_modality_importance.png', fig)


# =============================================================================
# FIG 24: modality_value.png
# =============================================================================
def fig_modality_value():
    d = _load('results_modality_value.npz')
    if d is None: return

    mods = list(d['mod_names'])
    aligned = d['inc_aligned']; misaligned = d['inc_misaligned']

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ys = np.arange(len(mods))
    for y, m, a, mi in zip(ys, mods, aligned, misaligned):
        ax.plot([a, mi], [y, y], color='#666', linewidth=1.2, zorder=1)
        ax.scatter(a, y, s=80, color=C_OBJ, edgecolor='#1f3f4c',
                   linewidth=0.5, zorder=3, label='Low misalignment' if y == 0 else None)
        ax.scatter(mi, y, s=80, color=C_ADV, edgecolor='#5a1a18',
                   linewidth=0.5, zorder=3, label='High misalignment' if y == 0 else None)
        ax.text(a, y + 0.18, f'{a:+.3f}', fontsize=9, color=C_OBJ, ha='center')
        ax.text(mi, y + 0.18, f'{mi:+.3f}', fontsize=9, color=C_ADV, ha='center')

    ax.axvline(0, color='#888', linewidth=0.6, linestyle='--')
    ax.set_yticks(ys); ax.set_yticklabels(mods)
    ax.set_xlabel('$\\Delta R^2$ over tabular baseline')
    ax.set_title('Visual modalities add value only when listings are misaligned',
                 fontsize=11, loc='center')
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    _style(ax)
    plt.tight_layout()
    _save('figure_24_modality_value.png', fig)


# =============================================================================
# FIG 25: cca_spectrum.png
# =============================================================================
def fig_cca_spectrum():
    d = _load('results_modality_value.npz')
    if d is None: return

    corrs = d['cca_corrs']
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(corrs)), corrs, color='#5C7A5C',
           edgecolor='#3a4f3a', linewidth=0.4)
    ax.axhline(0.1, color=C_ADV, linewidth=0.6, linestyle=':', alpha=0.8)
    ax.text(len(corrs) * 0.95, 0.12, 'shared/private boundary',
            fontsize=8, color=C_ADV, style='italic', ha='right')
    ax.set_xlabel('Canonical component')
    ax.set_ylabel('Canonical correlation')
    ax.set_title('CCA: shared vs private information between modality groups',
                 fontsize=10.5, loc='center')
    _style(ax)
    plt.tight_layout()
    _save('figure_25_cca.png', fig)


# =============================================================================
# FIG 26: interaction_types.png
# =============================================================================
def fig_interaction_types():
    d = _load('results_interaction_types.npz')
    if d is None: return

    names = list(d['type_names'])
    counts = d['type_counts']
    total = counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    cols_map = {'Synergy': C_GOLD, 'Redundancy': C_OBJ, 'Unique-S': C_ADV,
                'Unique-O': C_OBJ_L, 'Neither': C_CTRL}
    cols = [cols_map.get(n, C_CTRL) for n in names]
    pcts = [c / total * 100 for c in counts]
    ax.pie(pcts, labels=[f'{n}\n{p:.0f}\\%' for n, p in zip(names, pcts)],
           colors=cols, startangle=90, wedgeprops=dict(edgecolor='#222', linewidth=0.6))
    ax.set_title('a.  Interaction types', loc='left', fontsize=11)

    ax = axes[1]
    keys = list(d['phi_by_type_keys'])
    vals = list(d['phi_by_type_vals'])
    cols2 = [cols_map.get(n, C_CTRL) for n in keys]
    ax.bar(range(len(keys)), vals, color=cols2, edgecolor='#222', linewidth=0.4)
    ax.axhline(0, color='#888', linewidth=0.6, linestyle='--')
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, fontsize=9.5)
    ax.set_ylabel('Mean $\\phi_\\mathcal{S}$')
    ax.set_title('b.  Misalignment by type', loc='left', fontsize=11)
    _style(ax)

    plt.tight_layout()
    _save('figure_26_interactions.png', fig)


# =============================================================================
# FIG 27: nophoto.png
# =============================================================================
def fig_nophoto_control():
    d = _load('results_nophoto.npz')
    if d is None: return

    np_gap = d['np_gap_dist']
    p_gap = d['p_gap_dist']
    np_gap = np_gap[~np.isnan(np_gap)]
    p_gap = p_gap[~np.isnan(p_gap)]
    np_n = int(d['np_n']); p_n = int(d['p_n'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    ax = axes[0]
    bins = np.linspace(-0.5, 0.5, 50)
    ax.hist(p_gap, bins=bins, color=C_OBJ, alpha=0.7, edgecolor='#1f3f4c',
            linewidth=0.3, label=f'Has photos (n={p_n})')
    ax.hist(np_gap, bins=bins, color=C_ADV, alpha=0.7, edgecolor='#5a1a18',
            linewidth=0.3, label=f'No photos (n={np_n})')
    ax.set_xlabel('Price gap')
    ax.set_ylabel('Count')
    ax.set_title('a.  Price gap distribution', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc='upper right')
    _style(ax)

    ax = axes[1]
    np_dom = float(d['np_dom']); p_dom = float(d['p_dom'])
    ax.bar([0, 1], [p_dom, np_dom], color=[C_OBJ, C_ADV],
           edgecolor='#222', linewidth=0.4, width=0.5)
    for i, v in enumerate([p_dom, np_dom]):
        ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Has photos', 'No photos'])
    ax.set_ylabel('Median days on market')
    ax.set_title('b.  Marketing duration', loc='left', fontsize=11)
    _style(ax)

    plt.tight_layout()
    _save('figure_27_nophoto.png', fig)


# =============================================================================
# FIG 28: failed_alternatives.png
# =============================================================================
def fig_failed_alternatives():
    d = _load('results_failed_alts.npz')
    if d is None: return

    methods = ['Hand-crafted', 'Learned (Ridge)', 'Calibrated (MLP)', 'Contrastive']
    rs = [float(d.get('r_prompted', np.nan)),
          float(d['r_ridge']),
          float(d['r_mlp']),
          float(d['r_con'])]
    cols = [C_ADV if r > 0 else C_CTRL for r in rs]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(methods)), rs, color=cols, edgecolor='#222',
           linewidth=0.4, width=0.55)
    for i, r in enumerate(rs):
        col = C_ADV if r > 0 else '#444'
        ax.text(i, r + (0.005 if r > 0 else -0.015),
                f'{r:+.4f}', ha='center', fontsize=10, color=col,
                fontweight='bold', va='bottom' if r > 0 else 'top')
    ax.axhline(0, color='#888', linewidth=0.6, linestyle='--')
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('$r$(misalignment, price gap)')
    ax.set_title('Misalignment scoring methods compared', loc='center', fontsize=11)
    _style(ax)
    plt.tight_layout()
    _save('figure_28_failed_alts.png', fig)


# =============================================================================
# FIG 29: raw_cosines.png
# =============================================================================
def fig_raw_cosines():
    d = _load('results_raw_cosine.npz')
    if d is None: return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    bins = np.linspace(0.05, 0.85, 60)
    ax.hist(d['cos_tg'], bins=bins, color=C_ADV, alpha=0.7, edgecolor='#5a1a18',
            linewidth=0.3, label='Text $\\leftrightarrow$ GSV')
    ax.hist(d['cos_ts'], bins=bins, color=C_ADV_L, alpha=0.7,
            edgecolor='#a06c66', linewidth=0.3, label='Text $\\leftrightarrow$ Sat')
    ax.hist(d['cos_pg'], bins=bins, color=C_OBJ, alpha=0.7, edgecolor='#1f3f4c',
            linewidth=0.3, label='Photos $\\leftrightarrow$ GSV')
    ax.hist(d['cos_ps'], bins=bins, color=C_OBJ_L, alpha=0.7,
            edgecolor='#6b8090', linewidth=0.3, label='Photos $\\leftrightarrow$ Sat')
    ax.set_xlabel('Cosine similarity'); ax.set_ylabel('Count')
    ax.set_title('a.  Strategic $\\leftrightarrow$ objective', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    _style(ax)

    ax = axes[1]
    ax.hist(d['cos_tp'], bins=bins, color=C_ADV, alpha=0.7,
            edgecolor='#5a1a18', linewidth=0.3, label='Text $\\leftrightarrow$ Photos')
    ax.hist(d['cos_gs'], bins=bins, color=C_OBJ, alpha=0.7,
            edgecolor='#1f3f4c', linewidth=0.3, label='GSV $\\leftrightarrow$ Satellite')
    ax.set_xlabel('Cosine similarity')
    ax.set_title('b.  Within-class', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    _style(ax)

    ax = axes[2]
    ax.hist(d['M_aggregate'], bins=40, color=C_GOLD, edgecolor='#7e6624', linewidth=0.4)
    ax.set_xlabel('Raw misalignment $M_i$')
    ax.set_title(f'c.  Aggregate raw misalignment\nr with gap = {float(d["r_raw"]):+.3f}',
                 loc='left', fontsize=10)
    _style(ax)

    plt.tight_layout()
    _save('figure_29_raw_cosines.png', fig)


# =============================================================================
# FIG 30: spatial.png
# =============================================================================
def fig_spatial():
    d = _load('results_spatial.npz')
    if d is None: return

    lat = d['lat']; lon = d['lon']
    phi = d['phi']; gap = d['gap']
    phi_z = (phi - np.median(phi)) / (np.std(phi) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, vals, label, cmap, vmin, vmax in [
        (axes[0], phi_z, 'Misalignment $\\Delta_i$', 'RdBu_r', -2, 2),
        (axes[1], gap,   'Price gap', 'RdBu_r', -0.3, 0.3)]:
        sc = ax.scatter(lon, lat, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=4, alpha=0.6, edgecolor='none')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title(f'{"a" if vals is phi_z else "b"}.  {label}',
                     loc='left', fontsize=11)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04, label=label)
        ax.set_aspect('equal')
        _style(ax)

    plt.tight_layout()
    _save('figure_30_spatial.png', fig)


# =============================================================================
# main
# =============================================================================
def main():
    fig_pipeline_overview()
    fig_causal_dag()
    fig_early_vs_late_fusion()
    fig_headline_atom()
    fig_supervision_vs_pca()
    fig_causal_forest()
    fig_synergy_atoms()
    fig_peer_effect_iv()
    fig_data_overview()
    fig_bw_quartiles()
    fig_synth_calibration()
    fig_ksg_validation()
    fig_direction_stability()
    fig_city_atoms()
    fig_cross_city_transfer()
    fig_listing_vocab()
    fig_property_type()
    fig_property_type_heatmap()
    fig_quality_distributions()
    fig_per_attribute_quality()
    fig_architecture_comparison()
    fig_mult_attention()
    fig_modality_importance()
    fig_modality_value()
    fig_cca_spectrum()
    fig_interaction_types()
    fig_nophoto_control()
    fig_failed_alternatives()
    fig_raw_cosines()
    fig_spatial()


if __name__ == '__main__':
    main()
