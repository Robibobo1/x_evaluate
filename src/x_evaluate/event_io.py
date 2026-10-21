import numpy as np
from rosbag import Bag
import rospy

# One dvs_msgs/Event as serialized by ROS: uint16 x, uint16 y, time ts (uint32 secs, uint32 nsecs), bool polarity
EVENT_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('secs', '<u4'), ('nsecs', '<u4'), ('polarity', 'u1')])


def parse_event_array(data):
    """Decodes a serialized dvs_msgs/EventArray without creating one Python object per event (much faster).

    Layout: Header (uint32 seq, uint32 secs, uint32 nsecs, string frame_id), uint32 height, uint32 width, Event[] events
    """
    frame_id_length = np.frombuffer(data, '<u4', count=1, offset=12)[0]
    offset = 16 + frame_id_length
    height, width, n = np.frombuffer(data, '<u4', count=3, offset=offset)
    events = np.frombuffer(data, EVENT_DTYPE, count=n, offset=offset + 12)
    t = events['secs'] + events['nsecs'] * 1e-9
    return events['x'], events['y'], t, events['polarity'].astype(bool), (int(width), int(height))


def read_events(bag_file, topic, t_from=None, t_to=None):
    """Returns all events in [t_from, t_to] (absolute times in s) as numpy arrays x, y, t, polarity and the sensor size.

    Everything is loaded into memory (13 bytes per event), so keep the time span to what is needed.
    """
    start = rospy.Time.from_sec(t_from - 1.0) if t_from is not None else None  # message times lag their events
    end = rospy.Time.from_sec(t_to + 1.0) if t_to is not None else None
    chunks, size = [], None
    with Bag(bag_file) as bag:
        for _, msg, _ in bag.read_messages([topic], start_time=start, end_time=end, raw=True):
            x, y, t, p, size = parse_event_array(msg[1])
            chunks.append((x, y, t, p))
    if not chunks:
        raise LookupError(F"No events on '{topic}' in the requested time span")
    x, y, t, p = (np.concatenate(c) for c in zip(*chunks))
    keep = np.ones(len(t), dtype=bool)
    if t_from is not None:
        keep &= t >= t_from
    if t_to is not None:
        keep &= t <= t_to
    order = np.argsort(t[keep], kind='stable')
    return x[keep][order], y[keep][order], t[keep][order], p[keep][order], size
