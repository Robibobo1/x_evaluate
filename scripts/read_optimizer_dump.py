#!/usr/bin/env python3
"""Read the golden vectors of the EKLT optimizer (optimizer_dump.bin, written with EKLT_DUMP_EVERY=N).

The record format is defined in x_events include/x/eklt/optimizer_dump.h. Commands:

  summary <dump.bin>                 what the records hold, consistency checks, gradient window needed, slow vs fast
  show <dump.bin> <index> <out.png>  one solve: events, prediction from the template, residuals
  convergence <out.png> <dump.bin>...  how close each cost function call is to the final cost, over all records

The summary also recomputes the residuals and the cost of every evaluation from the record alone (event frame,
gradient window, patch center, parameters), with the same bicubic interpolation as Ceres. If they match the stored
values of the original code, the record is complete: a new implementation of the cost function can be tested on it.
"""
import argparse

import numpy as np

PATCH = 19
NPIX = PATCH * PATCH
HALF = (PATCH - 1) // 2
TERMINATION = {0: 'converged', 1: 'iteration cap', 2: 'failure'}  # ceres::TerminationType


def record_dtype(window, max_evals):
    """numpy layout of one dump::Record (same field order and sizes as the C++ struct, no padding)."""
    evaluation = np.dtype([('params', '<f8', 4), ('with_jacobian', '<f8'), ('cost', '<f8')])
    return np.dtype([
        ('solve_index', '<i8'), ('patch_id', '<i4'), ('num_evals', '<i4'),
        ('window_row0', '<i4'), ('window_col0', '<i4'), ('image_rows', '<i4'), ('image_cols', '<i4'),
        ('num_iterations', '<i4'), ('termination_type', '<i4'),
        ('num_jacobian_evaluations', '<i4'), ('num_residual_evaluations', '<i4'),
        ('t', '<f8'), ('t_init', '<f8'), ('center', '<f8', 2), ('init_center', '<f8', 2),
        ('params_init', '<f8', 4), ('params_final', '<f8', 4),
        ('initial_cost', '<f8'), ('final_cost', '<f8'), ('event_frame_norm', '<f8'),
        ('event_frame', '<f8', (PATCH, PATCH)),
        ('grad_window', '<f8', (window, window, 2)),
        ('evals', evaluation, max_evals),
        ('residuals_init', '<f8', NPIX), ('jacobian_init', '<f8', (NPIX, 4)),
        ('residuals_final', '<f8', NPIX), ('jacobian_final', '<f8', (NPIX, 4)),
    ])


def load(path):
    """Returns the records of a dump file as a numpy structured array (memory mapped)."""
    header = np.fromfile(path, dtype=np.dtype([('magic', 'S8'), ('version', '<u4'), ('header_size', '<u4'),
                                                ('record_size', '<u4'), ('patch_size', '<u4'),
                                                ('window_size', '<u4'), ('max_evals', '<u4')]), count=1)[0]
    if header['magic'] != b'EKLTDUMP' or header['version'] != 1 or header['patch_size'] != PATCH:
        raise ValueError(f'{path}: not an EKLT optimizer dump of version 1 with {PATCH}x{PATCH} patches')
    dtype = record_dtype(int(header['window_size']), int(header['max_evals']))
    if dtype.itemsize != header['record_size']:
        raise ValueError(f'{path}: record size {header["record_size"]}, reader expects {dtype.itemsize}')
    return np.memmap(path, dtype=dtype, mode='r', offset=int(header['header_size']))


def sample_positions(rec, params):
    """Positions (x, y) in the template frame where the cost function samples the gradient, as in ErrorRotation::warp.
    Returns two 19x19 arrays indexed [y, x]."""
    theta, tx, ty = params[0], params[1], params[2]
    a0, a1 = np.cos(theta), -np.sin(theta)
    a2, a3 = -a1, a0
    cx, cy = rec['center']
    y, x = np.mgrid[0:PATCH, 0:PATCH].astype(float)
    x_p = a0 * x + a1 * y + a0 * (cx - HALF) + a1 * (cy - HALF) + tx
    y_p = a2 * x + a3 * y + a2 * (cx - HALF) + a3 * (cy - HALF) + ty
    return x_p, y_p


def window_needed(rec, params):
    """Smallest square window, centered like the stored one, that holds every 4x4 bicubic neighbourhood."""
    x_p, y_p = sample_positions(rec, params)
    half = rec['grad_window'].shape[0] // 2
    center_col, center_row = rec['window_col0'] + half, rec['window_row0'] + half
    need = 0
    for pos, center in ((x_p, center_col), (y_p, center_row)):
        lo = center - (np.floor(pos).min() - 1)       # cells needed before the center
        hi = (np.floor(pos).max() + 2) - center       # cells needed after the center
        need = max(need, 2 * lo, 2 * (hi + 1))        # window [center - W/2, center + W/2 - 1]
    return int(need)


def shift(rec):
    """How far the solve moved the patch: shift in pixels of the sampled position of the patch center."""
    x0, y0 = sample_positions(rec, rec['params_init'])
    x1, y1 = sample_positions(rec, rec['params_final'])
    return np.hypot(x1[HALF, HALF] - x0[HALF, HALF], y1[HALF, HALF] - y0[HALF, HALF])


def cubic(p0, p1, p2, p3, x):
    """ceres::CubicHermiteSpline, value only."""
    a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)
    b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
    c = 0.5 * (-p0 + p2)
    return p1 + x * (c + x * (b + x * a))


def bicubic(window, row0, col0, r, c):
    """ceres::BiCubicInterpolator::Evaluate(r, c) on the stored window (r = image row, c = image column)."""
    row, col = np.floor(r).astype(int), np.floor(c).astype(int)
    wr, wc = row - row0, col - col0
    rows = [cubic(window[wr + i, wc - 1], window[wr + i, wc], window[wr + i, wc + 1], window[wr + i, wc + 2],
                  (c - col)[..., None]) for i in (-1, 0, 1, 2)]
    return cubic(*rows, (r - row)[..., None])


def prediction(rec, params):
    """The event pattern the template predicts, -a / sqrt(S) in ErrorRotation, as a 19x19 array [y, x]."""
    x_p, y_p = sample_positions(rec, params)
    g = bicubic(rec['grad_window'], rec['window_row0'], rec['window_col0'], y_p, x_p)
    a = g[..., 0] * np.cos(params[3]) + g[..., 1] * np.sin(params[3])
    return -a / np.sqrt(1e-3 + np.sum(a ** 2))


def residuals(rec, params):
    """The 361 residuals of ErrorRotation, recomputed from the record alone: events minus prediction."""
    return (rec['event_frame'] / rec['event_frame_norm'] - prediction(rec, params)).ravel()


def rel_err(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def summary(path):
    recs = load(path)
    n = len(recs)
    window = recs[0]['grad_window'].shape[0]
    max_evals = recs[0]['evals'].shape[0]
    print(f'{path}\n{n} records, solves {recs["solve_index"][0]} to {recs["solve_index"][-1]}, '
          f't = {recs["t"][0] - recs["t"][0]:.1f} to {recs["t"][-1] - recs["t"][0]:.1f} s, window {window}x{window}')

    n_evals = recs['num_evals']
    flags = np.array([r['evals']['with_jacobian'][:min(r['num_evals'], max_evals)].sum() for r in recs])
    print('\nPer solve (mean, min-max)')
    print(f'  iterations                {recs["num_iterations"].mean():5.2f}  {recs["num_iterations"].min()}-{recs["num_iterations"].max()}')
    print(f'  evaluations               {n_evals.mean():5.2f}  {n_evals.min()}-{n_evals.max()}'
          f'  (more than {max_evals}, truncated: {(n_evals > max_evals).sum()})')
    print(f'  with Jacobian             {flags.mean():5.2f}')
    print(f'  values only               {(np.minimum(n_evals, max_evals) - flags).mean():5.2f}')
    types, counts = np.unique(recs['termination_type'], return_counts=True)
    print('  termination               ' + ', '.join(f'{TERMINATION.get(k, k)} {c}' for k, c in zip(types, counts)))

    # consistency of what the C++ side stored
    print('\nChecks against the stored values (largest relative difference over all records)')
    cost_final = 0.5 * np.sum(recs['residuals_final'] ** 2, axis=1)
    cost_init = 0.5 * np.sum(recs['residuals_init'] ** 2, axis=1)
    norm = np.linalg.norm(recs['event_frame'].reshape(n, -1), axis=1)
    first_eval = recs['evals']['cost'][:, 0]
    print(f'  0.5 |r_final|^2 vs final cost       {rel_err(cost_final, recs["final_cost"]):.1e}')
    print(f'  0.5 |r_init|^2 vs initial cost      {rel_err(cost_init, recs["initial_cost"]):.1e}')
    print(f'  first evaluation vs initial cost    {rel_err(first_eval, recs["initial_cost"]):.1e}')
    print(f'  norm of the event frame             {rel_err(norm, recs["event_frame_norm"]):.1e}')
    print(f'  evaluations = Ceres counts          {np.all(n_evals == recs["num_jacobian_evaluations"] + recs["num_residual_evaluations"])}')

    # gradient window: how much of it the sampled positions need, over every evaluation of every solve
    need = np.array([max(window_needed(r, e['params']) for e in r['evals'][:min(r['num_evals'], max_evals)])
                     for r in recs])
    complete = need <= window
    print(f'\nGradient window needed (square, over every evaluation): median {np.median(need):.0f}, '
          f'99% {np.percentile(need, 99):.0f}, 99.9% {np.percentile(need, 99.9):.0f}, max {need.max()}')
    print(f'  records complete with the stored {window}x{window} window: {complete.sum()} of {n}')

    # recompute the cost function from the record alone, on the complete records
    err_r, err_c = 0.0, 0.0
    for r in recs[complete]:
        err_r = max(err_r, rel_err(residuals(r, r['params_init']), r['residuals_init']),
                    rel_err(residuals(r, r['params_final']), r['residuals_final']))
        for e in r['evals'][:min(r['num_evals'], max_evals)]:
            err_c = max(err_c, abs(0.5 * np.sum(residuals(r, e['params']) ** 2) - e['cost']) / e['cost'])
    print('\nRecomputed from the record alone (numpy, same bicubic interpolation as Ceres)')
    print(f'  residuals at start and end, largest relative difference   {err_r:.1e}')
    print(f'  cost of every evaluation, largest relative difference     {err_c:.1e}')

    # slow vs fast: the same quantities by time since the start of the sequence
    t = recs['t'] - recs['t'][0]
    step = np.array([shift(r) for r in recs])
    print('\nBy time (the motion gets faster over the sequence)')
    print('  time [s]   records   window needed (median / max)   shift of the patch per solve [px] (median / max)   final cost (median)')
    for t0 in range(0, int(np.ceil(t.max())), 10):
        sel = (t >= t0) & (t < t0 + 10)
        if sel.any():
            print(f'  {t0:3d}-{t0 + 10:<3d}   {sel.sum():7d}   {np.median(need[sel]):13.0f} / {need[sel].max():<4d}'
                  f'           {np.median(step[sel]):21.2f} / {step[sel].max():<6.2f}'
                  f'                    {np.median(recs["final_cost"][sel]):.3f}')

    print('\nWork of the cost function that changes the result little')
    print('\n'.join(work_after_convergence(recs)))


def show(path, index, out):
    """Figure: what the cost function compares for one solve, at its end: events, prediction, residuals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = load(path)[index]
    events = r['event_frame'] / r['event_frame_norm']
    pred = prediction(r, r['params_final'])
    res = r['residuals_final'].reshape(PATCH, PATCH)
    lim = max(np.abs(events).max(), np.abs(pred).max())  # one color scale for the three panels

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, img, title in ((axes[0], events, 'Events'),
                           (axes[1], pred, 'Prediction from the template'),
                           (axes[2], res, 'Residuals (events - prediction)')):
        im = ax.imshow(img, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, label='normalized value')
    fig.suptitle(f'One solve (19x19 patch): cost {r["initial_cost"]:.3f} before, {r["final_cost"]:.3f} after',
                 fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved {out}')


def evaluation_gaps(recs):
    """Relative gap to the final cost of every stored evaluation, (cost - final cost) / final cost, as a
    (records x max evals) array with NaN where a solve made fewer evaluations; also whether each one had a Jacobian."""
    n, m = len(recs), recs[0]['evals'].shape[0]
    gap = np.full((n, m), np.nan)
    jac = np.zeros((n, m), dtype=bool)
    for i, r in enumerate(recs):
        k = min(r['num_evals'], m)
        gap[i, :k] = np.abs(r['evals']['cost'][:k] - r['final_cost']) / r['final_cost']
        jac[i, :k] = r['evals']['with_jacobian'][:k] > 0
    return gap, jac


def convergence(paths, out):
    """Figure: how much of the work of a solve still changes its result (all records of one or more dump files)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, path in enumerate(paths):
        recs = load(path)
        gap, jac = evaluation_gaps(recs)
        gap[gap < 1e-12] = np.nan  # calls at the final parameters (gap 0 up to rounding), not shown on a log axis
        name = path.split('/')[-2].split('_', 1)[-1]
        k = np.arange(gap.shape[1])
        count = np.sum(~np.isnan(gap), axis=0)
        keep = count >= 0.5 * len(recs)  # evaluation numbers reached by at least half of the solves
        lo, med, hi = (np.nanpercentile(gap[:, keep], q, axis=0) for q in (10, 50, 90))
        color = f'C{c}'
        ax.fill_between(k[keep], lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(k[keep], med, '-', color=color, lw=1.8, label=f'{name} ({len(recs)} solves), median and 10-90%')
        mostly_jac = np.nanmean(np.where(np.isnan(gap[:, keep]), np.nan, jac[:, keep]), axis=0) > 0.5
        ax.plot(k[keep][mostly_jac], med[mostly_jac], 'o', color=color, ms=6)
        ax.plot(k[keep][~mostly_jac], med[~mostly_jac], 'x', color=color, ms=8, mew=2)
    ax.plot([], [], 'o', color='0.3', label='with Jacobian')
    ax.plot([], [], 'x', color='0.3', mew=2, label='values only (trial step)')
    ax.set_yscale('log')
    ax.set_xlabel('call of the cost function within one solve')
    ax.set_ylabel('(cost - final cost) / final cost')
    ax.set_xticks(np.arange(0, 21, 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('How close each call of the cost function is to the final result\n'
                 '(calls at the final parameters not shown)', fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved {out}')


def work_after_convergence(recs, thresholds=(1e-3, 1e-6)):
    """Share of all cost function calls made once the cost is within a relative threshold of its final value,
    and share of the values-only calls whose point is evaluated again with the Jacobian right after."""
    gap, jac = evaluation_gaps(recs)
    calls = np.sum(~np.isnan(gap))
    lines = []
    for th in thresholds:
        # a call is "after convergence" if this call and every later one stay within the threshold
        after = 0
        for g in gap:
            g = g[~np.isnan(g)]
            above = np.flatnonzero(g > th)
            after += len(g) - (above[-1] + 1 if len(above) else 0)
        lines.append(f'  calls made once the cost is within {th:.0e} of its final value: {after / calls:.0%}')
    repeated, trials = 0, 0
    for r in recs:
        ev = r['evals'][:min(r['num_evals'], len(r['evals']))]
        for a, b in zip(ev[:-1], ev[1:]):
            if a['with_jacobian'] == 0:
                trials += 1
                repeated += b['with_jacobian'] > 0 and np.array_equal(a['params'], b['params'])
    lines.append(f'  values-only calls evaluated again with the Jacobian right after: {repeated / max(trials, 1):.0%}'
                 f' ({repeated / calls:.0%} of all calls)')
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('summary')
    p.add_argument('dump')
    p = sub.add_parser('show')
    p.add_argument('dump')
    p.add_argument('index', type=int)
    p.add_argument('out')
    p = sub.add_parser('convergence')
    p.add_argument('out')
    p.add_argument('dumps', nargs='+')
    args = parser.parse_args()
    if args.command == 'summary':
        summary(args.dump)
    elif args.command == 'show':
        show(args.dump, args.index, args.out)
    else:
        convergence(args.dumps, args.out)
