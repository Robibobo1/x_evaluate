#!/usr/bin/env python3
"""Compare two EKLT-VIO runs of the same sequences, e.g. with the original cost function and with a new version of it
(validation levels 2 and 3 of the FPGA work): optimizer solves, feature tracks, trajectory error, solve timing.

usage: compare_eklt_runs.py <reference run> <new run> [--threshold 0.01]

A run is an output folder of test/evaluate.py: evaluation.pickle and one subfolder per sequence. The sequences found in
both runs are compared. The trajectory error needs the x_evaluate package, so run this script in the Docker container.

Tracks: a feature of the reference is matched with the feature of the new run whose tracking started on the same frame
at the same position (Bootstrap row). Ids are not compared: once one feature is lost at a different time, the ids of
all later features shift. The track points of matched features are compared at equal timestamps (same event batch).

Timing: Ceres' own times of each solve (optimizations.csv). The solves written to an optimizer dump are left out, the
dump makes them slower.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from read_optimizer_dump import load as load_dump

def sequences(run):
    """{sequence name: folder}, e.g. {'shapes_6dof': '<run>/001_shapes_6dof'}"""
    return {re.sub(r'^\d+_', '', d): os.path.join(run, d) for d in sorted(os.listdir(run))
            if os.path.isfile(os.path.join(run, d, 'eklt_tracks.csv'))}


def read_solves(folder):
    """optimizations.csv, one row per solve, and which rows were dumped"""
    df = pd.read_csv(os.path.join(folder, 'optimizations.csv'), sep=';')
    dumped = np.zeros(len(df), dtype=bool)
    dump = os.path.join(folder, 'optimizer_dump.bin')
    if os.path.exists(dump):
        index = np.asarray(load_dump(dump)['solve_index'])
        dumped[index[index < len(df)]] = True
    return df, dumped


def read_tracks(folder):
    """Features by (bootstrap time, x, y) -> id, track of each id as (times, positions), and counts. A feature gets a
    new id when its tracking starts (Bootstrap row: KLT on the frame after its detection), the Update rows use it."""
    df = pd.read_csv(os.path.join(folder, 'eklt_tracks.csv'), sep=';',
                     usecols=['id', 'update_type', 'patch_t_current', 'center_x', 'center_y'])
    boot = df[df.update_type == 'Bootstrap']
    keys = dict(zip(zip(boot.patch_t_current, boot.center_x, boot.center_y), boot.id))
    upd = df[df.update_type == 'Update']
    tracks = {i: (g.patch_t_current.to_numpy(), g[['center_x', 'center_y']].to_numpy())
              for i, g in upd.groupby('id', sort=False)}
    counts = {'detected': int((df.update_type == 'Init').sum()), 'tracked': len(boot),
              'lost': int((df.update_type == 'Lost').sum())}
    return keys, tracks, counts


def compare_tracks(ref_folder, new_folder, threshold):
    """Distance of every reference track point to the new run's point at the same time (inf if there is none), and
    for each matched feature what first made its track differ by more than the threshold"""
    ref_keys, ref_tracks, ref_counts = read_tracks(ref_folder)
    new_keys, new_tracks, new_counts = read_tracks(new_folder)
    empty = (np.empty(0), np.empty((0, 2)))
    dists = []
    first = {'none (whole track within)': 0, 'batch at another time': 0, 'lost at another time': 0,
             'position drifted': 0}
    matched = 0
    for key, ref_id in ref_keys.items():
        tr, pr = ref_tracks.get(ref_id, empty)
        d = np.full(len(tr), np.inf)
        new_id = new_keys.get(key)
        if new_id is not None:
            matched += 1
            tn, pn = new_tracks.get(new_id, empty)
            _, ir, jn = np.intersect1d(tr, tn, assume_unique=True, return_indices=True)
            d[ir] = np.hypot(*(pr[ir] - pn[jn]).T)
            # walk both tracks in order until the first point that differs
            m = min(len(tr), len(tn))
            same_time = tr[:m] == tn[:m]
            close = same_time & (np.hypot(*(pr[:m] - pn[:m]).T) < threshold)
            if close.all():
                first['none (whole track within)' if len(tr) == len(tn) else 'lost at another time'] += 1
            else:
                k = int(np.argmin(close))
                first['position drifted' if same_time[k] else 'batch at another time'] += 1
        dists.append(d)
    return np.concatenate(dists), matched, first, ref_counts, new_counts


def trajectory_errors(run):
    """Mean position error in % of the traveled distance per sequence, as in the trajectory summary of evaluate.py
    but not rounded; None without the x_evaluate package"""
    try:
        from evo.core import metrics
        from x_evaluate.scriptlets import read_evaluation_pickle
        from x_evaluate.utils import name_to_identifier
    except ImportError:
        return None
    key = str(metrics.APE(metrics.PoseRelation.translation_part))
    errors = {}
    for name, d in read_evaluation_pickle(run).data.items():
        if d.trajectory_data is not None:
            errors[name_to_identifier(name)] = \
                100 * np.mean(d.trajectory_data.ate_errors[key]) / d.trajectory_data.traj_gt.distances[-1]
    return errors


def row(label, ref, new, fmt, diff='rel'):
    if diff == 'rel':
        d = f'{100 * (new - ref) / ref:+.2f}%'
    else:
        d = f'{new - ref:+.4f} pp'
    print(f'  {label:<40} {ref:>12{fmt}} {new:>12{fmt}} {d:>12}')


def compare(seq, ref_folder, new_folder, threshold, ref_mpe, new_mpe):
    print(f'\n=== {seq} ===')
    print(f'  {"":<40} {"reference":>12} {"new":>12} {"difference":>12}')

    ref_s, ref_dumped = read_solves(ref_folder)
    new_s, new_dumped = read_solves(new_folder)
    dists, matched, first, ref_counts, new_counts = compare_tracks(ref_folder, new_folder, threshold)

    print('Level 2, tracker')
    row('solves', len(ref_s), len(new_s), ',d')
    row('features detected', ref_counts['detected'], new_counts['detected'], ',d')
    row('features tracked', ref_counts['tracked'], new_counts['tracked'], ',d')
    row('features lost', ref_counts['lost'], new_counts['lost'], ',d')
    row('iterations per solve (mean)', ref_s.num_iterations.mean(), new_s.num_iterations.mean(), '.3f')
    row('final cost (mean)', ref_s.final_cost.mean(), new_s.final_cost.mean(), '.4f')
    n = ref_counts['tracked']
    print(f'  reference features started on the same frame at the same position in the new run: {matched:,d} of {n:,d}')
    print(f'  reference track points ({len(dists):,d}) without a new point at the same time within ' +
          ', '.join(f'{t:g} px: {np.sum(dists >= t):,d}' for t in (1e-6, threshold, 0.1, 1.0)))
    same_time = dists[np.isfinite(dists)]
    print(f'  distance between points at the same time [px]: median {np.median(same_time):.1e}, '
          f'99.9% {np.percentile(same_time, 99.9):.1e}, max {same_time.max():.1e}')
    print(f'  matched features, first difference above {threshold:g} px: ' +
          ', '.join(f'{k} {v:,d}' for k, v in first.items()))

    print('Level 3, trajectory')
    if ref_mpe and new_mpe and seq in ref_mpe and seq in new_mpe:
        row('mean position error [%]', ref_mpe[seq], new_mpe[seq], '.4f', diff='pp')
    else:
        print('  mean position error: needs the x_evaluate package (run in the Docker container)')

    print(f'Solve timing, Ceres (dumped solves left out: {ref_dumped.sum()} reference, {new_dumped.sum()} new)')
    timing = []
    for s, dumped in ((ref_s, ref_dumped), (new_s, new_dumped)):
        s = s[~dumped]
        timing.append((1e6 * s.jacobian_evaluation_time.sum() / s.num_jacobian_evaluations.sum(),
                       1e6 * s.residual_evaluation_time.sum() / s.num_residual_evaluations.sum(),
                       1e6 * s.total_time.mean(),
                       s.total_time.sum()))
    for k, label in enumerate(('cost function call with Jacobian [us]', 'cost function call, values only [us]',
                               'solve (mean) [us]', 'all solves [s]')):
        row(label, timing[0][k], timing[1][k], '.1f')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reference', help='run folder of the reference, e.g. /data/runs/rpg/golden_vectors')
    parser.add_argument('new', help='run folder to compare with it')
    parser.add_argument('--threshold', type=float, default=0.01, help='track points closer than this are the same [px]')
    args = parser.parse_args()

    ref_seqs, new_seqs = sequences(args.reference), sequences(args.new)
    ref_mpe, new_mpe = trajectory_errors(args.reference), trajectory_errors(args.new)
    print(f'reference: {args.reference}\nnew:       {args.new}')
    for seq in ref_seqs:
        if seq in new_seqs:
            compare(seq, ref_seqs[seq], new_seqs[seq], args.threshold, ref_mpe, new_mpe)


if __name__ == '__main__':
    main()
