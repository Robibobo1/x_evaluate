#!/usr/bin/env python3
"""Where does EKLT-VIO spend its time? Hierarchical time breakdown of one `evaluate` run.

All inputs are written by one run of the C++ `evaluate` binary into its output folder:

  profiling.json     easy_profiler blocks (the EASY_BLOCK(...) calls in x_events), in ns. evaluate writes
                     profiling.prof; convert it inside the container with
                       LD_LIBRARY_PATH=$(dirname $(find /catkin_ws -name libeasy_profiler.so | head -1)) \
                         /catkin_ws/devel/bin/profiler_converter profiling.prof profiling.json
  optimizations.csv  one row per Ceres solve, written at the end of Optimizer::optimizeParameters (optimizer.cpp),
                     columns ts_start;ts_stop;num_iterations;final_cost, timestamps in raw CPU ticks
  events.csv         one row per event, written by EventPerfHelper around the body of the per-event loop in
                     AsyncFeatureTracker::processEvents (async_feature_tracker.cpp), raw CPU ticks
  realtime.csv       one row per rosbag message (evaluate.cpp), used here only to convert ticks to ns

The block tree is nested: Events Message > EKLT Tracking > per-event loop > optimizeParameters. Each "rest" row is a
parent minus its measured children, i.e. the code inside the parent that has no timer of its own.

usage: analyze_time_breakdown.py <run folder> [<profiling.json>]   (default: <run folder>/profiling.json)
"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict


def block_tree(profiling_json):
    """Sum stop - start per block path, like scripts/plot_performance_pie_charts.py does per level."""
    with open(profiling_json) as f:
        top = json.load(f)['threads'][0]['children']
    total_s, count = defaultdict(float), defaultdict(int)

    def walk(block, path):
        path = path + (block['name'],)
        total_s[path] += (block['stop'] - block['start']) * 1e-9
        count[path] += 1
        for child in block.get('children', []):
            walk(child, path)

    for block in top:
        walk(block, ())
    return top, total_s, count


def ticks_per_ns(run, top_blocks):
    """realtime.csv (ticks) and profiling.json (ns) both mark the end of each message: ratio of the two spans."""
    with open(os.path.join(run, 'realtime.csv')) as f:
        ticks = [int(r['ts_real']) for r in csv.DictReader(f, delimiter=';')]
    # realtime.csv skips the messages before the filter is initialised, so align both lists on their last entries
    ns = [b['stop'] for b in top_blocks if b['name'] != 'GT Message'][-len(ticks):]
    return (ticks[-1] - ticks[0]) / (ns[-1] - ns[0])


def main():
    run = sys.argv[1]
    profiling_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, 'profiling.json')

    top, t, n = block_tree(profiling_json)
    k = ticks_per_ns(run, top)

    solve_s, iterations = [], []
    with open(os.path.join(run, 'optimizations.csv')) as f:
        for r in csv.DictReader(f, delimiter=';'):
            solve_s.append((int(r['ts_stop']) - int(r['ts_start'])) / k * 1e-9)
            iterations.append(int(r['num_iterations']))

    event_loop_s, n_events = 0.0, 0
    with open(os.path.join(run, 'events.csv')) as f:
        next(f)  # header
        for line in f:
            start, stop = line.split(';')
            event_loop_s += (int(stop) - int(start)) / k * 1e-9
            n_events += 1

    ev, imu, img = ('Events Message',), ('IMU Message',), ('Image Message',)
    trk, vis = ev + ('EKLT Tracking',), ev + ('EKLT EKF Vision Updates',)
    fm, upd = vis + ('Backend feature management',), vis + ('Backend EKF Update',)
    total = t[ev] + t[imu] + t[img]

    rows = [  # (depth, name, seconds, calls)
        (0, 'Events Message', t[ev], n[ev]),
        (1, 'EKLT Tracking (frontend)', t[trk], n[trk]),
        (2, 'per-event loop body (events.csv)', event_loop_s, n_events),
        (3, 'Optimizer::optimizeParameters (optimizations.csv)', sum(solve_s), len(solve_s)),
        (3, 'rest: updatePatch x patches, event frames, interpolation, refills', event_loop_s - sum(solve_s), None),
        (2, 'rest: outside the per-event loop', t[trk] - event_loop_s, None),
        (1, 'EKLT EKF Vision Updates (backend)', t[vis], n[vis]),
        (2, 'Backend feature management', t[fm], n[fm]),
        (2, 'Backend EKF Update', t[upd], n[upd]),
        (2, 'rest: preProcess/manageTracks, closestIdx, repropagation', t[vis] - t[fm] - t[upd], None),
        (1, 'rest: msgToEvents conversion', t[ev] - t[trk] - t[vis], None),
        (0, 'IMU Message', t[imu], n[imu]),
        (1, 'EKF IMU Update', t[imu + ('EKF IMU Update',)], n[imu + ('EKF IMU Update',)]),
        (0, 'Image Message', t[img], n[img]),
    ]

    print(f'run: {run}')
    print(f'clock: {k:.6f} CPU ticks per ns (from realtime.csv vs profiling.json)')
    print(f'total time inside profiled messages: {total:.2f} s\n')
    print(f'{"block":<74s}{"seconds":>10s}{"% total":>9s}{"calls":>10s}{"ms/call":>9s}')
    for depth, name, sec, calls in rows:
        per_call = f'{1e3 * sec / calls:9.3f}' if calls else ''
        calls = f'{calls:10d}' if calls else ''
        print(f'{"  " * depth + name:<74s}{sec:10.2f}{100 * sec / total:8.2f}%{calls:>10s}{per_call}')

    q = statistics.quantiles(solve_s, n=10)
    print(f'\nper solve: median {1e3 * statistics.median(solve_s):.2f} ms, p10 {1e3 * q[0]:.2f} ms, '
          f'p90 {1e3 * q[-1]:.2f} ms, iterations reported by Ceres: min {min(iterations)}, max {max(iterations)}')


if __name__ == '__main__':
    main()
