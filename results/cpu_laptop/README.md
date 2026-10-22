# CPU baseline on the laptop

EKLT-VIO on the 10 Event Camera Dataset sequences of the paper's Table II, CPU only, on an i5-1345U laptop. This is the
reference the FPGA version is compared against. Setup: Docker image of this repository, X library `jpl-x/x_events`
`fe2facf` (branch `events`) with the tile clamp patch, `test/evaluate_rpg_table2.yaml`, `params_rpg_davis.yaml`,
optimizer capped at 10 iterations as in the published code.

- `accuracy/<sequence>.xlsx`: `trajectory_tracking_summary.xlsx` of each sequence (mean position, rotation and yaw
  errors), written by `test/evaluate.py`. The 10 runs ran in parallel. Average mean position error 0.54%, as in the
  paper; the table per sequence is in `docs/eklt_vio_reproduction_notes.md`, section 7.
- `timing/`: each sequence run alone, pinned to one fast core (`--cpuset-cpus=2`, up to 4.7 GHz), laptop plugged in.
  The trajectories are byte-identical to the accuracy runs.
  - `<sequence>_breakdown.txt`: share of the time per part of the code, from `scripts/analyze_time_breakdown.py`.
  - `<sequence>_load.txt`: load per second of data and its correlation with the event rate and the number of solves,
    from `scripts/analyze_load_over_time.py` (its `--plot` output is not kept here).
  - `summary.txt`: one line per sequence with the main numbers of both.

The full run outputs (CSV logs, plots, pickles; about 57 GB) stay outside the repository, in `data/runs/rpg/`
(`eklt_reproduction/` and `eklt_timing/`). The paths printed in the text files are those of the older layout
(`/data/rpg_runs/...`).
