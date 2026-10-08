# -*- coding: utf-8 -*-
"""Shared Lancet-style helpers for BilinGuard figures (2026-07-20 revamp)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# muted Lancet palette (matched to legacy figures)
C_ARCH = {'ConvNeXt': '#3b4b63', 'ViT': '#c2694f', 'Swin': '#4e9a8f',
          'EfficientNet': '#c8a165', 'Ensemble': '#1a1a1a'}
LS_ARCH = {'ConvNeXt': '-', 'ViT': '--', 'Swin': '-.', 'EfficientNet': ':', 'Ensemble': '-'}
SCOPE_COL = {'Face': '#3b4b63', 'Eyelid': '#c2694f', 'Fusion': '#4e9a8f'}
TASK_COL = {'Binary Screening': '#3b4b63', 'Ternary Grading': '#8fa876', 'DBIL': '#7570b3',
            'IBIL': '#1b9e77', 'Jaundice Type': '#c2694f', 'Child-Pugh': '#c8a165', 'MELD': '#4e9a8f'}


def set_rc():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333', 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'legend.frameon': False, 'legend.fontsize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def panel_title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', fontsize=8, fontweight='bold', loc='center', pad=6)


def grid(ax):
    ax.grid(True, lw=0.4, alpha=0.3, color='#999999')
    ax.set_axisbelow(True)


def roc_axes(ax):
    ax.plot([0, 1], [0, 1], ':', color='#999999', lw=0.7)
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    grid(ax)


def fig_suptitle(fig, text):
    fig.suptitle(text, fontsize=11, fontweight='bold', x=0.02, ha='left', y=0.995)
