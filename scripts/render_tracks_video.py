import argparse
import os
from collections import deque

import cv2
import numpy as np
import pandas as pd
import yaml
from rosbag import Bag
from tqdm import tqdm

from render_events_video import image_to_bgr
from x_evaluate.event_io import read_events

# BGR, none of them red or blue so that they stand out from the events
TRACK_COLORS = [(0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 165, 255), (160, 255, 160),
                (255, 255, 255), (130, 200, 255)]
MARKER_COLOR = (255, 255, 255)
SOLVE_RATE_WINDOW = 0.1  # [s]


class TrackReplay:
    """Replays the EKLT track log (eklt_tracks.csv) of one run: which features exist at a given data time.

    Rows are applied in the order of their data time (patch_t_current). A track starts with its Bootstrap row (flow
    direction initialized by KLT on the frame after the detection), moves with every Update row (one optimizer solve)
    and ends with its Lost row. Init rows are the Harris corners detected on a frame; a corner added after the first
    frame gets a new id before its Bootstrap (EkltPatch::reset), so Init rows are shown as detections, not linked to
    tracks.
    """

    def __init__(self, run_folder, trail):
        df = pd.read_csv(os.path.join(run_folder, "eklt_tracks.csv"), delimiter=";")
        df = df.sort_values('patch_t_current', kind='stable')
        self._rows = list(zip(df['patch_t_current'], df['id'], df['update_type'], df['center_x'], df['center_y'],
                              df['flow_angle']))
        self._next = 0
        self._trail = trail
        self.tracks = {}  # id -> (deque of (t, x, y) over the last trail seconds, [flow angle])
        self.detections = deque()  # (t, x, y)
        self.lost = deque()  # (t, x, y)
        self.solve_times = df.loc[df['update_type'] == 'Update', 'patch_t_current'].to_numpy()

    def advance(self, t, frame_time):
        """Applies the rows up to data time t; frame_time is the stamp of the frame shown at t. Detections are only
        kept while their frame is shown (their time is the frame stamp + cam1_time_offset), lost markers for as long as
        a track line."""
        while self._next < len(self._rows) and self._rows[self._next][0] <= t:
            t_row, i, kind, x, y, angle = self._rows[self._next]
            self._next += 1
            if kind == 'Init':
                self.detections.append((t_row, x, y))
            elif kind == 'Lost':
                self.tracks.pop(i, None)
                self.lost.append((t_row, x, y))
            else:  # Bootstrap or Update
                positions, flow_angle = self.tracks.setdefault(i, (deque(), [0.0]))
                positions.append((t_row, x, y))
                flow_angle[0] = angle
        for positions, _ in self.tracks.values():
            while len(positions) > 1 and positions[0][0] < t - self._trail:
                positions.popleft()
        while self.detections and self.detections[0][0] < frame_time:
            self.detections.popleft()
        while self.lost and self.lost[0][0] <= t - self._trail:
            self.lost.popleft()

    def solve_rate(self, t):
        a, b = np.searchsorted(self.solve_times, [t - SOLVE_RATE_WINDOW, t], side='right')
        return (b - a) / SOLVE_RATE_WINDOW


def to_px(x, y, scale):
    """Image coordinates (pixel centers at integers) to integer coordinates of the upscaled frame."""
    return int(round((x + 0.5) * scale)), int(round((y + 0.5) * scale))


def draw_tracks(panel, replay: TrackReplay, scale, patch_size, arrow_length):
    half = scale * patch_size / 2
    for i, (positions, flow_angle) in replay.tracks.items():
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        _, x, y = positions[-1]
        cx, cy = to_px(x, y, scale)
        if len(positions) > 1:
            trail = np.array([to_px(px, py, scale) for _, px, py in positions], np.int32)
            cv2.polylines(panel, [trail], False, color, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (int(cx - half), int(cy - half)), (int(cx + half), int(cy + half)), color, 1, cv2.LINE_AA)
        tip = to_px(x + arrow_length * np.cos(flow_angle[0]), y + arrow_length * np.sin(flow_angle[0]), scale)
        cv2.arrowedLine(panel, (cx, cy), tip, color, 1, cv2.LINE_AA, tipLength=0.3)
    for _, x, y in replay.detections:
        cv2.circle(panel, to_px(x, y, scale), 2 * scale, MARKER_COLOR, 1, cv2.LINE_AA)
    for _, x, y in replay.lost:
        cv2.drawMarker(panel, to_px(x, y, scale), MARKER_COLOR, cv2.MARKER_TILTED_CROSS, 3 * scale, 1, cv2.LINE_AA)


def put_text(img, text, origin):
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Renders a video of the camera frame with the events of the last "
                                                 "--window seconds on top (red: brightness increase, blue: decrease) "
                                                 "and the EKLT feature tracks of one or more runs (one panel each).")
    parser.add_argument('--input_folders', nargs='+', required=True,
                        help="run folders written by evaluate.py (with eklt_tracks.csv), e.g. "
                             "/data/runs/rpg/eklt_reproduction/shapes_translation/001_shapes_translation")
    parser.add_argument('--names', nargs='+', help="panel titles (default: the evaluation folder of each run)")
    parser.add_argument('--input_bag', type=str, required=True, help="dataset bag the runs were computed on")
    parser.add_argument('--events_topic', type=str, default='/dvs/events')
    parser.add_argument('--image_topic', type=str, default='/dvs/image_raw')
    parser.add_argument('--output', type=str, required=True, help="output .mp4 file")
    parser.add_argument('--t_from', type=float, default=0.0, help="start, in s after the first event")
    parser.add_argument('--duration', type=float, default=10.0, help="length of the clip in data time [s]")
    parser.add_argument('--window', type=float, default=0.01, help="events shown per video frame [s]")
    parser.add_argument('--trail', type=float, default=0.1, help="length of the track lines [s]")
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--slowdown', type=float, default=1.0, help="e.g. 4 plays the data 4 times slower")
    parser.add_argument('--scale', type=int, default=4, help="upscaling factor of the output video")
    parser.add_argument('--frame_brightness', type=float, default=0.6,
                        help="dims the camera frame (0-1) so that events and tracks stand out")
    parser.add_argument('--arrow_length', type=float, default=8.0, help="flow arrow length in image pixels")

    args = parser.parse_args()
    names = args.names or [os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(f))))
                           for f in args.input_folders]
    assert len(names) == len(args.input_folders), "give one name per input folder"

    with open(os.path.join(args.input_folders[0], "params.yaml")) as f:
        patch_size = yaml.safe_load(f)['eklt_patch_size']

    with Bag(args.input_bag) as bag:
        first_event = next(bag.read_messages([args.events_topic]))[1].events[0].ts.to_sec()
    t_start = first_event + args.t_from
    t_end = t_start + args.duration

    x, y, t, p, size = read_events(args.input_bag, args.events_topic, t_start - args.window, t_end)

    image_times, images = [], []
    with Bag(args.input_bag) as bag:
        for _, msg, _ in bag.read_messages([args.image_topic]):
            stamp = msg.header.stamp.to_sec()
            if t_start - 1.0 <= stamp <= t_end:
                image_times.append(stamp)
                images.append(msg)
    image_times = np.array(image_times)

    replays = [TrackReplay(f, args.trail) for f in args.input_folders]

    w, h = size
    band = 28
    separator = 4
    panel_w, panel_h = args.scale * w, args.scale * h
    out_size = (len(replays) * panel_w + (len(replays) - 1) * separator, band + panel_h + band)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, out_size)

    frame_times = np.arange(t_start, t_end, args.slowdown / args.fps)
    for t_frame in tqdm(frame_times):
        # background: latest camera frame at or before t_frame, events in (t_frame - window, t_frame] on top
        i = np.searchsorted(image_times, t_frame, side='right') - 1
        background = image_to_bgr(images[i], size) if i >= 0 else np.zeros((h, w, 3), np.uint8)
        background = (args.frame_brightness * background).astype(np.uint8)
        a, b = np.searchsorted(t, [t_frame - args.window, t_frame], side='right')
        pos = p[a:b]
        background[y[a:b][~pos], x[a:b][~pos]] = (255, 80, 0)  # blue (BGR): darker
        background[y[a:b][pos], x[a:b][pos]] = (0, 40, 255)  # red: brighter
        background = cv2.resize(background, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

        frame = np.zeros((out_size[1], out_size[0], 3), np.uint8)
        for k, (replay, name) in enumerate(zip(replays, names)):
            replay.advance(t_frame, image_times[i] if i >= 0 else -np.inf)
            panel = background.copy()
            draw_tracks(panel, replay, args.scale, patch_size, args.arrow_length)
            x0 = k * (panel_w + separator)
            frame[band:band + panel_h, x0:x0 + panel_w] = panel
            title = F"{name}: " if len(replays) > 1 else ""
            put_text(frame, F"{title}t = {t_frame - first_event:6.2f} s   tracks: {len(replay.tracks):3d}   "
                            F"solves: {replay.solve_rate(t_frame) / 1e3:4.1f} k/s   events: last "
                            F"{1e3 * args.window:.0f} ms", (x0 + 8, 19))
            put_text(frame, F"square: patch, arrow: flow direction, line: last {1e3 * args.trail:.0f} ms, "
                            F"o: new corner, x: lost", (x0 + 8, band + panel_h + 19))
        writer.write(frame)

    writer.release()
    print(F"Wrote {len(frame_times)} frames ({len(frame_times) / args.fps:.1f} s of video) to {args.output}")


if __name__ == '__main__':
    main()
