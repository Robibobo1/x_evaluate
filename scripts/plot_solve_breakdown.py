#!/usr/bin/env python3
"""Which share of the whole EKLT-VIO processing time would an FPGA kernel of the EKLT cost function cover?

One horizontal bar per run, 100% = time inside the profiled messages, split into:
  - the cost function evaluations with and without Jacobian (Ceres' Solver::Summary timings), the planned HLS kernel
  - the rest of Optimizer::optimizeParameters: linear solves, Levenberg-Marquardt logic, Ceres setup and bookkeeping
  - the rest of EKLT tracking: routing events to patches, event frames, interpolation
  - everything else: backend EKF updates, IMU and image messages
Same inputs and computation as scripts/analyze_time_breakdown.py, so the run folders need profiling.json and an
optimizations.csv with the solve split columns.

usage: plot_solve_breakdown.py <output.png> <run folder> [<run folder> ...]
"""
import argparse
import os
import re

import matplotlib.pyplot as plt

from analyze_time_breakdown import block_tree, read_solves, ticks_per_ns

SURFACE, INK, INK_MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
PARTS = [  # (legend, fill, label ink)
    ('Cost function with derivatives (HLS kernel)', '#2a78d6', '#ffffff'),
    ('Cost function, values only (HLS kernel)', '#86b6ef', INK),
    ('Rest of the optimizer: linear solve, LM logic, Ceres', '#eb6834', INK),
    ('Rest of EKLT: events to patches, event frames', '#c3c2b7', INK),
    ('Backend EKF, IMU and images', '#8a8984', '#ffffff'),
]


def shares(run):
    top, t, _ = block_tree(os.path.join(run, 'profiling.json'))
    solve_s, _, p = read_solves(run, ticks_per_ns(run, top))
    ev = ('Events Message',)
    total = t[ev] + t[('IMU Message',)] + t[('Image Message',)]
    tracking, optimizer = t[ev + ('EKLT Tracking',)], sum(solve_s)
    jac, res = p['jacobian_evaluation_time'], p['residual_evaluation_time']
    return [100 * s / total for s in (jac, res, optimizer - jac - res, tracking - optimizer, total - tracking)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help="image file to write, e.g. time_shares.png")
    parser.add_argument('runs', nargs='+', help="output folders of one dataset each, e.g. .../001_shapes_6dof")
    args = parser.parse_args()

    names = [re.sub(r'^\d+_', '', os.path.basename(os.path.normpath(r))) for r in args.runs]
    data = [shares(r) for r in args.runs]

    fig, ax = plt.subplots(figsize=(9, 1.6 + 0.55 * len(data)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for row, values in enumerate(data):
        left = 0.0
        for value, (_, fill, ink) in zip(values, PARTS):
            # the surface-coloured edge leaves a thin gap between neighbouring segments
            ax.barh(row, value, left=left, height=0.55, color=fill, edgecolor=SURFACE, linewidth=1.5)
            if value >= 4:
                ax.text(left + value / 2, row, f'{value:.1f}%', ha='center', va='center',
                        fontsize=9 if value >= 8 else 7.5, color=ink)
            left += value
        ax.text(101, row, f'HLS kernel: {values[0] + values[1]:.1f}%', ha='left', va='center', fontsize=9.5,
                color=INK, fontweight='bold')

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color=INK_MUTED)
    ax.tick_params(length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=fill) for _, fill, _ in PARTS]
    ax.legend(handles, [label for label, _, _ in PARTS], loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.5, labelcolor=INK)
    fig.suptitle('Where EKLT-VIO spends its time', x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
    fig.text(0.01, 0.865, 'Share of the processing time, first 20 s of each sequence, one laptop core (i5-1345U)',
             fontsize=9.5, color=INK_MUTED)
    fig.subplots_adjust(left=0.13, right=0.83, top=0.55, bottom=0.12)
    fig.savefig(args.output, dpi=200, facecolor=SURFACE)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
