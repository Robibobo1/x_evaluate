#!/usr/bin/env bash
# Start a shell (or run a command) in the EKLT-VIO container.
#
#   docker/run.sh                 # interactive shell, builds the image on first use
#   docker/run.sh --build         # force an image rebuild (also resets the catkin build volumes)
#   docker/run.sh <cmd> [args]    # run a single command in the container
#
# Environment:
#   DATA_DIR          host data folder, mounted at /data (default: ../data next to this repo); datasets are read from
#                     its datasets/ subfolder, run outputs go to runs/
#   X_SRC             optional host checkout of the X library (Robibobo1/x_events, branch `accelerator`), mounted over
#                     /catkin_ws/src/x to work on it
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=eklt-vio:noetic
VOLUMES=(eklt_ws_build eklt_ws_devel eklt_ws_logs)
DATA_DIR="$(realpath "${DATA_DIR:-$REPO_DIR/../data}")"

if [[ "${1:-}" == "--build" ]]; then
  shift
  docker volume rm -f "${VOLUMES[@]}" >/dev/null
  REBUILD=1
fi

if [[ -n "${REBUILD:-}" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" \
    --build-arg USER_ID="$(id -u)" --build-arg GROUP_ID="$(id -g)" \
    -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR"
fi

ARGS=(
  --rm -it
  -v "$REPO_DIR:/catkin_ws/src/x_evaluate"
  -v "$DATA_DIR:/data"
  # catkin build/devel/logs persist across container runs (seeded from the image on first use)
  -v eklt_ws_build:/catkin_ws/build
  -v eklt_ws_devel:/catkin_ws/devel
  -v eklt_ws_logs:/catkin_ws/logs
  -e XVIO_DATASET_DIR=/data/datasets
  -e X_EVALUATE_SRC_ROOT=/catkin_ws/src/x_evaluate
  # plots are written to files; without this matplotlib picks the Tk backend when DISPLAY is set (tkinter not installed)
  -e MPLBACKEND="${MPLBACKEND:-Agg}"
)
if [[ -n "${X_SRC:-}" ]]; then
  ARGS+=(-v "$(realpath "$X_SRC"):/catkin_ws/src/x")
fi
if [[ -n "${DISPLAY:-}" ]]; then  # for the EKLT feature display windows; run `xhost +local:` once on the host
  ARGS+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
fi

exec docker run "${ARGS[@]}" "$IMAGE" "${@:-bash}"
