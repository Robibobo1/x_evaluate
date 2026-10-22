import argparse
import os

import cv2
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import yaml
from rosbag import Bag
from tqdm import tqdm

from render_events_video import image_to_bgr
from render_tracks_video import TrackReplay, TRACK_COLORS

# RGB; the estimates of several runs take the next colours in turn
GT_COLOR = (170, 170, 170)
RUN_COLORS = [(255, 140, 0), (0, 170, 255), (220, 60, 220), (60, 200, 60)]


def read_poses(csv_file, prefix):
    """Times and poses (N x 7: position, quaternion x y z w) of gt.csv or pose.csv, sorted by time.

    As in x_evaluate.utils.convert_to_evo_trajectory, rows with t = -1 (update without a valid state) or NaNs are
    dropped. Measurement updates are written after the IMU propagation of the same time, so the last row of each time
    is kept.
    """
    df = pd.read_csv(csv_file, delimiter=";")
    columns = [prefix + c for c in ('p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w')]
    df = df[(df['t'] != -1) & df[columns].notna().all(axis=1)]
    df = df.drop_duplicates('t', keep='last').sort_values('t')
    return df['t'].to_numpy(), df[columns].to_numpy()


def resample(times, poses, rate):
    """Latest pose at or before each tick of a grid at the given rate [Hz]."""
    grid = np.arange(times[0], times[-1], 1.0 / rate)
    return grid, poses[np.searchsorted(times, grid, side='right') - 1]


def camera_model(params):
    """Camera matrix in pixels (x stores fx, fy, cx, cy as fractions of the image size) and OpenCV distortion."""
    assert params['cam1_dist_model'] == 'radial-tangential', "only the radial-tangential model is undistorted"
    w, h = params['cam1_img_width'], params['cam1_img_height']
    camera_matrix = np.array([[params['cam1_fx'] * w, 0, params['cam1_cx'] * w],
                              [0, params['cam1_fy'] * h, params['cam1_cy'] * h],
                              [0, 0, 1]])
    return camera_matrix, np.array(params['cam1_dist_params'], dtype=float), (w, h)


def undistort_points(points, camera_matrix, distortion):
    """Pixel coordinates (centers at integers) to undistorted Rerun image coordinates (pixel (0, 0) spans [0, 1))."""
    if len(points) == 0:
        return np.zeros((0, 2))
    points = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, camera_matrix, distortion, P=camera_matrix).reshape(-1, 2) + 0.5


def log_trajectory(entity, label, times, poses, color, t0, args):
    """Current pose (labelled point and axes) and the line of the trajectory up to now, under entity."""
    t_pose, pose = resample(times, poses, args.pose_rate)
    for t, p in zip(t_pose, pose):
        rr.set_time_seconds("time", t - t0)
        rr.log(F"{entity}/body", rr.Transform3D(translation=p[:3], rotation=rr.Quaternion(xyzw=p[3:]),
                                                 axis_length=0.1))
    rr.set_time_seconds("time", t_pose[0] - t0)
    rr.log(F"{entity}/body/marker", rr.Points3D([[0, 0, 0]], colors=color, labels=label,
                                                 radii=rr.Radius.ui_points(5.0)))

    # the whole line up to now is logged at each update, so that each point in time shows its own line
    t_path, path = resample(times, poses, args.path_rate)
    for t in np.arange(t_path[0], t_path[-1], 1.0 / args.line_update_rate):
        rr.set_time_seconds("time", t - t0)
        rr.log(F"{entity}/path", rr.LineStrips3D([path[:np.searchsorted(t_path, t, side='right'), :3]], colors=color,
                                                  radii=rr.Radius.ui_points(1.5)))


def main():
    parser = argparse.ArgumentParser(description="Writes a Rerun recording (.rrd) of one or more runs: ground truth and "
                                                 "estimated trajectories in 3D, the camera of the first run with the "
                                                 "undistorted frame in its frustum, and the EKLT features of the first "
                                                 "run in the camera view. Open it with the Rerun viewer of version "
                                                 "0.22: rerun <file>.rrd")
    parser.add_argument('--input_folders', nargs='+', required=True,
                        help="run folders written by evaluate.py (pose.csv, gt.csv, params.yaml, eklt_tracks.csv), "
                             "e.g. /data/runs/rpg/eklt_reproduction/shapes_translation/001_shapes_translation")
    parser.add_argument('--names', nargs='+', help="run names (default: the evaluation folder of each run); "
                                                   "letters, digits and underscores only")
    parser.add_argument('--input_bag', type=str, required=True, help="dataset bag the runs were computed on")
    parser.add_argument('--events_topic', type=str, default='/dvs/events', help="only used for the time origin")
    parser.add_argument('--image_topic', type=str, default='/dvs/image_raw')
    parser.add_argument('--output', type=str, required=True, help="output .rrd file")
    parser.add_argument('--image_distance', type=float, default=1.0,
                        help="distance of the frame in front of the camera in the 3D view [m]")
    parser.add_argument('--pose_rate', type=float, default=100.0, help="rate of the current poses [Hz]")
    parser.add_argument('--path_rate', type=float, default=50.0, help="points per second of the trajectory lines")
    parser.add_argument('--line_update_rate', type=float, default=10.0,
                        help="how often per second the trajectory lines grow")
    parser.add_argument('--feature_rate', type=float, default=30.0, help="rate of the EKLT feature updates [Hz]")
    parser.add_argument('--trail', type=float, default=0.1, help="length of the feature track lines [s]")
    parser.add_argument('--arrow_length', type=float, default=8.0, help="flow arrow length in image pixels")

    args = parser.parse_args()
    names = args.names or [os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(f))))
                           for f in args.input_folders]
    assert len(names) == len(args.input_folders), "give one name per input folder"
    camera_entity = F"world/{names[0]}/body/camera/image"

    with open(os.path.join(args.input_folders[0], "params.yaml")) as f:
        params = yaml.safe_load(f)
    camera_matrix, distortion, size = camera_model(params)
    q_ic = params['cam1_q_ic']  # [w, x, y, z]: orientation of the camera in the IMU frame

    with Bag(args.input_bag) as bag:
        t0 = next(bag.read_messages([args.events_topic]))[1].events[0].ts.to_sec()

    rr.init("x_evaluate_trajectory_view")
    rr.save(args.output, default_blueprint=rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="3D"),
        rrb.Spatial2DView(origin=camera_entity, name=F"camera of {names[0]} (undistorted)"),
        column_shares=[2, 1])))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # OptiTrack room frame, z up

    print("Logging trajectories")
    t_gt, gt = read_poses(os.path.join(args.input_folders[0], "gt.csv"), "")
    log_trajectory("world/ground_truth", "ground truth", t_gt, gt, GT_COLOR, t0, args)
    t_start = np.inf
    for k, (folder, name) in enumerate(zip(args.input_folders, names)):
        t_est, est = read_poses(os.path.join(folder, "pose.csv"), "estimated_")
        t_start = min(t_start, t_est[0])
        log_trajectory(F"world/{name}", name, t_est, est, RUN_COLORS[k % len(RUN_COLORS)], t0, args)
        # the state is the IMU pose; the camera sits at p_ic, rotated by q_ic, in the IMU frame
        # (x::State::computeCameraPosition / computeCameraOrientation)
        rr.log(F"world/{name}/body/camera", rr.Transform3D(translation=params['cam1_p_ic'],
                                                            rotation=rr.Quaternion(xyzw=q_ic[1:] + q_ic[:1])),
               static=True)
        rr.log(F"world/{name}/body/camera/image", rr.Pinhole(image_from_camera=camera_matrix, resolution=size,
                                                              image_plane_distance=args.image_distance), static=True)

    print("Logging frames")
    with Bag(args.input_bag) as bag:
        for _, msg, _ in tqdm(bag.read_messages([args.image_topic])):
            stamp = msg.header.stamp.to_sec()
            if stamp < t_start:
                continue  # no camera pose yet
            gray = cv2.cvtColor(image_to_bgr(msg, size), cv2.COLOR_BGR2GRAY)
            rr.set_time_seconds("time", stamp - t0)
            rr.log(camera_entity, rr.Image(cv2.undistort(gray, camera_matrix, distortion)).compress(jpeg_quality=85))

    print("Logging EKLT features")
    replay = TrackReplay(args.input_folders[0], args.trail)
    half_size = params['eklt_patch_size'] / 2
    t_end = t_gt[-1]
    for t in tqdm(np.arange(t_start, t_end, 1.0 / args.feature_rate)):
        replay.advance(t, t)  # detections and lost markers are not shown here
        ids = list(replay.tracks)
        centers = undistort_points([replay.tracks[i][0][-1][1:] for i in ids], camera_matrix, distortion)
        angles = np.array([replay.tracks[i][1][0] for i in ids])
        colors = [TRACK_COLORS[i % len(TRACK_COLORS)][::-1] for i in ids]  # BGR to RGB
        trails = [undistort_points([(x, y) for _, x, y in replay.tracks[i][0]], camera_matrix, distortion)
                  for i in ids]
        rr.set_time_seconds("time", t - t0)
        rr.log(F"{camera_entity}/features", rr.Boxes2D(centers=centers, half_sizes=[[half_size, half_size]] * len(ids),
                                                       colors=colors))
        rr.log(F"{camera_entity}/flow", rr.Arrows2D(origins=centers, colors=colors, vectors=args.arrow_length * np.stack(
            [np.cos(angles), np.sin(angles)], axis=-1).reshape(-1, 2)))
        rr.log(F"{camera_entity}/trails", rr.LineStrips2D(trails, colors=colors))

    print(F"Wrote {args.output}")


if __name__ == '__main__':
    main()
