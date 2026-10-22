#!/usr/bin/env python3
"""How does the computational load of an EKLT-VIO run evolve over the sequence, and does it follow the event rate?

Reads the CSV files written by one run of the C++ `evaluate` binary (see scripts/analyze_time_breakdown.py for what
each file contains) and computes, per window of data time:
  - event rate (rows of events.csv, one per processed event)
  - processing time per second of data, split into
      optimizer   Optimizer::optimizeParameters (optimizations.csv)
      event loop  rest of the per-event loop body (events.csv minus optimizer)
      other       everything else: backend updates, IMU and image messages (realtime.csv minus the above)
  - number of optimizer solves, median iterations, active EKLT patches (eklt_tracks.csv, distinct ids updated)
CPU-tick timestamps are mapped to data time by interpolating between the (ts_real, t_sim) pairs of realtime.csv.

usage: analyze_load_over_time.py <run folder> [--window 1.0] [--plot output.png]
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', help="output folder of one dataset, e.g. .../001_poster_6dof")
    parser.add_argument('--window', type=float, default=1.0, help="window length in seconds of data")
    parser.add_argument('--plot', type=str, default=None, help="save a plot to this file")
    args = parser.parse_args()

    rt = pd.read_csv(os.path.join(args.run, 'realtime.csv'), delimiter=';')
    # t_real is the accumulated processing time, while the ts_real ticks also advance between messages (bag reading);
    # for long messages that gap is negligible, so ticks between consecutive messages / processing time = tick rate
    d_ticks = np.diff(rt['ts_real'].values)
    long = rt['process_time_in_us'].values[1:] > 50000
    ticks_per_s = np.median(d_ticks[long] / (rt['process_time_in_us'].values[1:][long] * 1e-6))
    tick_to_sim = lambda ticks: np.interp(ticks, rt['ts_real'], rt['t_sim'])
    t0 = rt['t_sim'].iloc[0]

    ev = pd.read_csv(os.path.join(args.run, 'events.csv'), delimiter=';', dtype=np.int64)
    opt = pd.read_csv(os.path.join(args.run, 'optimizations.csv'), delimiter=';')
    tracks = pd.read_csv(os.path.join(args.run, 'eklt_tracks.csv'), delimiter=';')
    tracks = tracks[tracks['update_type'] == 'Update']

    edges = np.arange(t0, rt['t_sim'].iloc[-1] + args.window, args.window)
    w = lambda t_sim: np.clip(np.digitize(t_sim, edges) - 1, 0, len(edges) - 2)
    n = len(edges) - 1

    ev_w = w(tick_to_sim(ev['ts_start'].values))
    opt_w = w(tick_to_sim(opt['ts_start'].values))
    rt_w = w(rt['t_sim'].values)
    tracks_w = w(tracks['patch_t_current'].values)

    events = np.bincount(ev_w, minlength=n) / args.window
    loop_s = np.bincount(ev_w, (ev['ts_stop'] - ev['ts_start']).values / ticks_per_s, minlength=n) / args.window
    opt_s = np.bincount(opt_w, (opt['ts_stop'] - opt['ts_start']).values / ticks_per_s, minlength=n) / args.window
    total_s = np.bincount(rt_w, rt['process_time_in_us'].values * 1e-6, minlength=n) / args.window
    solves = np.bincount(opt_w, minlength=n) / args.window
    iters = np.array([np.median(opt['num_iterations'].values[opt_w == i]) if np.any(opt_w == i) else np.nan
                      for i in range(n)])
    # active EKLT patches: distinct features that received a tracking update within the window
    features = np.array([tracks['id'].values[tracks_w == i].size and np.unique(tracks['id'].values[tracks_w == i]).size
                         for i in range(n)])

    df = pd.DataFrame({'t': edges[:-1] - t0, 'events_per_s': events, 'rtf': total_s, 'optimizer': opt_s,
                       'event_loop': loop_s - opt_s, 'other': total_s - loop_s, 'solves_per_s': solves,
                       'median_iterations': iters, 'features': features})

    corr = lambda a, b: np.corrcoef(df[a], df[b])[0, 1]
    print(f"run: {args.run}")
    print(f"windows: {n} x {args.window} s | realtime factor: mean {df['rtf'].mean():.2f}, min {df['rtf'].min():.2f}, "
          f"max {df['rtf'].max():.2f} (max/min {df['rtf'].max() / df['rtf'].min():.1f})")
    print(f"share of processing time: optimizer {df['optimizer'].sum() / df['rtf'].sum():.1%}, rest of event loop "
          f"{df['event_loop'].sum() / df['rtf'].sum():.1%}, other (backend, IMU, images) {df['other'].sum() / df['rtf'].sum():.1%}")
    print(f"event rate: mean {df['events_per_s'].mean() / 1e3:.0f} k/s, min {df['events_per_s'].min() / 1e3:.0f} k/s, "
          f"max {df['events_per_s'].max() / 1e3:.0f} k/s | solves: mean {df['solves_per_s'].mean():.0f}/s, "
          f"max {df['solves_per_s'].max():.0f}/s, "
          f"{df['events_per_s'].sum() / df['solves_per_s'].sum():.1f} events per solve, "
          f"{1e3 * df['optimizer'].sum() / df['solves_per_s'].sum():.2f} ms per solve")
    print(f"correlation with load (realtime factor): event rate {corr('events_per_s', 'rtf'):.2f}, solves "
          f"{corr('solves_per_s', 'rtf'):.2f}, active patches {corr('features', 'rtf'):.2f}")
    print(f"correlation event rate vs solves: {corr('events_per_s', 'solves_per_s'):.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axs[0].plot(df['t'], df['events_per_s'] / 1e3, color='tab:purple')
        axs[0].set_ylabel('events [k/s]')
        axs[1].stackplot(df['t'], df['optimizer'], df['event_loop'], df['other'],
                         labels=['optimizer', 'rest of event loop', 'backend, IMU, images'],
                         colors=['tab:red', 'tab:orange', 'tab:gray'])
        axs[1].axhline(1.0, color='k', linestyle='--', linewidth=0.8, label='real time')
        axs[1].set_ylabel('processing time\nper data second [s]')
        axs[1].legend(loc='upper left', fontsize=8)
        axs[2].plot(df['t'], df['solves_per_s'], label='solves / s', color='tab:red')
        axs[2].set_ylabel('optimizer solves / s')
        ax_f = axs[2].twinx()
        ax_f.plot(df['t'], df['features'], label='active EKLT patches', color='tab:blue')
        ax_f.set_ylabel('active EKLT patches', color='tab:blue')
        axs[2].set_xlabel('time in sequence [s]')
        fig.suptitle(os.path.basename(os.path.normpath(args.run)))
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot: {args.plot}")


if __name__ == '__main__':
    main()
