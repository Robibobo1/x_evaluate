import argparse

import cv2
import numpy as np
from rosbag import Bag
from tqdm import tqdm

from x_evaluate.event_io import read_events


def image_to_bgr(msg, size):
    """sensor_msgs/Image (mono8 or bgr8/rgb8) to a BGR array of the sensor size; odd-sized frames are padded."""
    channels = 1 if msg.encoding == 'mono8' else 3
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * channels]
    img = img.reshape(msg.height, msg.width, channels)
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((size[1], size[0], 3), np.uint8)
    h, w = min(size[1], img.shape[0]), min(size[0], img.shape[1])
    canvas[:h, :w] = img[:h, :w]
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Renders a video with the camera frame on the left and the events of "
                                                 "the last --window seconds on the right (red: brightness increase, "
                                                 "blue: decrease).")
    parser.add_argument('--input_bag', type=str, required=True)
    parser.add_argument('--events_topic', type=str, default='/dvs/events')
    parser.add_argument('--image_topic', type=str, default='/dvs/image_raw')
    parser.add_argument('--output', type=str, required=True, help="output .mp4 file")
    parser.add_argument('--t_from', type=float, default=0.0, help="start, in s after the first event")
    parser.add_argument('--duration', type=float, default=10.0, help="length of the clip in data time [s]")
    parser.add_argument('--window', type=float, default=0.02, help="events accumulated per video frame [s]")
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--slowdown', type=float, default=1.0, help="e.g. 4 plays the data 4 times slower")
    parser.add_argument('--scale', type=int, default=2, help="upscaling factor of the output video")

    args = parser.parse_args()

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

    w, h = size
    separator = 4
    out_size = (args.scale * (2 * w + separator), args.scale * h)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, out_size)

    frame_times = np.arange(t_start, t_end, args.slowdown / args.fps)
    for t_frame in tqdm(frame_times):
        # latest camera frame at or before t_frame
        i = np.searchsorted(image_times, t_frame, side='right') - 1
        left = image_to_bgr(images[i], size) if i >= 0 else np.zeros((h, w, 3), np.uint8)

        # events in (t_frame - window, t_frame]
        a, b = np.searchsorted(t, [t_frame - args.window, t_frame], side='right')
        right = np.zeros((h, w, 3), np.uint8)
        pos = p[a:b]
        right[y[a:b][~pos], x[a:b][~pos]] = (255, 80, 0)  # blue (BGR): darker
        right[y[a:b][pos], x[a:b][pos]] = (0, 40, 255)  # red: brighter

        frame = np.hstack([left, np.full((h, separator, 3), 128, np.uint8), right])
        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_NEAREST)
        cv2.putText(frame, F"t = {t_frame - first_event:6.2f} s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, F"events, last {1e3 * args.window:.0f} ms", (args.scale * (w + separator) + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(F"Wrote {len(frame_times)} frames ({len(frame_times) / args.fps:.1f} s of video, {len(t)} events) to "
          F"{args.output}")


if __name__ == '__main__':
    main()
