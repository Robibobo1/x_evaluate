# EKLT-VIO reproduction notes

Findings from reproducing EKLT-VIO (Mahlknecht et al., "Exploring Event Camera-based Odometry for Planetary Robots",
RA-L 2022) with the published code (`jpl-x/x_evaluate` master `5d95acc`, `jpl-x/x_events` branch `events`
`fe2facf`, 2022-02-21) on the public data. Each point states what was observed and how it was established; open points
are marked as such.

## 1. E2VID reconstructions on the Mars-like sequences, and the real-time constraint

- For the Wells Cave and JPL Mars Yard sequences, the paper does not feed EKLT-VIO the camera frames but intensity
  images reconstructed from events with E2VID, one every 15'000 events (Sec. IV-D). All Wells entries in
  `test/evaluate_wells.yaml` use the topic `/dvs/image_reconstructions` for this reason.
- These reconstructions are not part of the public Wells dataset, which only contains the raw DAVIS346 frames
  (`/dvs/image_raw`). The paper's Wells / Mars Yard results can therefore not be reproduced from public data as is.
- E2VID is a neural network. The paper reports 30 FPS at 640 x 480 on a Quadro RTX 4000 desktop GPU (about 160 W) and
  argues that EKLT-VIO only needs a subset of the images, since frames are used for feature initialization only.
- The real-time analysis of the paper (Table IV, RTF median 17.9, max 43.6 on Poster 6DOF) is done with raw frames, so
  the E2VID cost is not included. On the Mars-like sequences, the pipeline that produced the results is thus neither
  real-time (EKLT-VIO alone already is not) nor executable on embedded hardware without a GPU or neural network
  accelerator.
- Consequence for an FPGA study with dataset replay: E2VID images can be computed offline and replayed like a camera,
  so CPU-only and CPU+FPGA runs still see identical input; image reconstruction is then outside the accelerated system
  and must be stated as such.

## 2. Published code differs from the version used for the paper

- `x_events` stops on 2022-02-21, while the authors kept changing `x_evaluate` until 2022-08 (paper accepted 2022-06).
- `params_wells_gt.yaml` (2022-02-24) uses option values the published library does not accept
  (`eklt_ekf_update_strategy: every-n-msec`, `eklt_ekf_update_timestamp: latest-event`; the published parser knows
  `every-n-msec-with-events` and `latest-event-ts`).
- Commits of 2022-04 add the event-filter speedups of Table IV, which are not in the published library.
- Without modification, the published library crashes (segmentation fault) on the authors' own Wells data: Wells
  Test 5 and Test 7 (at 5.0 s). Another user reports a segmentation fault (cause not diagnosed) on an RPG sequence
  in `jpl-x/x_evaluate` issue #4 (2024-03, unanswered). No public issue reports a successful run.

## 3. Bugs found in the published library

- **Feature tile out of range (crash).** `TiledImage::setTileForFeature` computes an unsigned row index that underflows
  when an EKLT feature lies slightly above the image (y between -1 and 0 px); the index is then used on a per-tile
  array in `TrackManager::manageTracks`. Fixed by clamping the tile to the grid,
  `docker/patches/x_events-clamp-feature-tile.patch`. Verified neutral: the 5113 poses before the original crashes are
  identical with and without the patch.
- **Optimizer iteration limit ignored.** `EKLTVIO` constructs the EKLT tracker with default `Params`, whose
  `eklt_max_num_iterations` is 10; `Optimizer::setParams` later stores the real parameters but never updates the Ceres
  `solver_options`. All configs ask for 25 iterations, the solver stops at 10. On Wells Test 7, 98.6% of the per-event
  optimizations hit this cap. Fixing it (25 iterations) does not change the outcome below: the solver then hits the new
  cap in almost every solve as well (median 26 iterations).

## 4. Wells configuration was still being tuned

- `x_evaluate` commit `8535861` (2022-02-22): "Updates newly tuned corner detector parameters for Wells cave (most
  likely wrong, given the ranges of Harris k)". The current `test/evaluate_wells.yaml` contains values such as
  `eklt_harris_k: -0.5` (Test 7).
- `ff4ba08` (2022-02-18) lowered `eklt_tracking_quality` from 0.6 to 0.3 for Wells. A feature is declared lost when its
  tracking quality (1 - final ECC cost / 2) falls below this value. On Wells Test 7 not a single optimization out of
  158'819 scored below 0.3, so lost-feature detection is effectively disabled with this setting; 60% are below 0.6.
- `286d007`, `3c910c9`: IMU-camera time offset and transformation for Wells were "best guess" values.

## 5. Results on public data with raw frames

| Run | Data | Result |
|---|---|---|
| KLT-VIO (`--frontend XVIO`) | Wells Test 7 (21 s, "circle at exit") | Plausible loop, 4.3 x 7.2 m, 2.6 m start-end distance |
| EKLT-VIO | Wells Test 7 | Diverges: 324 m from start after 21 s (quadratic growth) |
| EKLT-VIO, `eklt_tracking_quality=0.6` | Wells Test 7 | Diverges (289 m) |
| EKLT-VIO, iteration limit fixed | Wells Test 7 | Diverges (499 m) |
| EKLT-VIO, both | Wells Test 7 | Diverges (413 m) |
| EKLT-VIO | Wells Test 5 (30 s) | Diverges (kilometres) |

Same bag, calibration, initial state and filter for KLT-VIO and EKLT-VIO: the data and calibration are usable, the
failure is specific to the EKLT path. On Wells Test 7, EKLT tracks agree with KLT on the frames to 1-3 px (median)
between 4.5 and 12 s, but deviate by 12-50 px in the first 4.5 s at the half-saturated cave exit.

Also observed on MVSEC indoor_flying1 (drone, motion capture ground truth; work on branch `mvsec`): KLT-VIO 0.88 m mean
position error over a 26.7 m flight; EKLT-VIO diverges at liftoff, where its tracks on a fine-grained carpet slip by
25-100 px within half a second while being kept by the filter (`eklt_tracking_quality` 0.3 inherited from Wells).

## 6. Open points

- Code version or input? Only the Event Camera Dataset (RPG, DAVIS 240C, motion capture) has published EKLT-VIO numbers
  per sequence, obtained with raw frames (paper Table II, e.g. boxes_6dof 0.84%, poster_6dof 0.35%, average 0.54%).
  Reproducing them with the published code would show whether the code is sound and the Wells / MVSEC failures come
  from the input (raw frames instead of E2VID) and parameters, or whether the published code itself differs. The RPG
  download server (`download.ifi.uzh.ch`) has failed TLS handshakes since 2026-10-21.
- Not yet tested: the authors' last Wells update settings (12 ms fixed rate, latest-event timestamp) and the Wells
  detector values before the "most likely wrong" commit.

## Reproduction

All runs use the Docker setup (`docker/run.sh`). Wells raw-frame runs use the entry of `test/evaluate_wells.yaml` with
`image_topic: /dvs/image_raw`; parameter tests are passed with `evaluate.py --overrides key=value`. The iteration fix is
the two-line change in `Optimizer::setParams` (`solver_options.max_num_iterations = params_.eklt_max_num_iterations;`).
