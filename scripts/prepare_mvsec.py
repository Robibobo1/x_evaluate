import argparse

import numpy as np
import yaml
from geometry_msgs.msg import PoseStamped
from rosbag import Bag
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Combines an MVSEC *_data.bag and *_gt.bag into the single bag the "
                                                 "'evaluate' executable reads (left DAVIS only). The ground truth "
                                                 "/davis/left/pose is the left camera pose relative to its first pose; "
                                                 "it is rewritten as IMU pose in a z-up (gravity-aligned) world frame, "
                                                 "since the evaluation aligns trajectories in position and yaw only. "
                                                 "Gravity is taken from the accelerometer while the sensor rests at "
                                                 "the start of the sequence.")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--calib', type=str, required=True, help="camchain-imucam-*.yaml from the MVSEC calib zip")
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--t_rest', type=float, default=2.0, help="duration at the start where the sensor is at rest [s]")

    args = parser.parse_args()

    with open(args.calib) as f:
        T_cam_imu = np.array(yaml.safe_load(f)['cam0']['T_cam_imu'])
    R_ci, p_ci = T_cam_imu[:3, :3], T_cam_imu[:3, 3]  # p_ci: IMU origin in camera frame

    with Bag(args.gt) as gt_bag:
        gt = [(t, m) for _, m, t in gt_bag.read_messages(['/davis/left/pose'])]
    t0 = gt[0][1].header.stamp.to_sec()

    with Bag(args.data) as data_bag:
        acc = [[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
               for _, m, _ in data_bag.read_messages(['/davis/left/imu']) if m.header.stamp.to_sec() < t0 + args.t_rest]
    a_rest = np.mean(acc, axis=0)

    # up direction in the mocap frame, from the resting specific force, and the smallest rotation taking it to +z
    o = gt[0][1].pose.orientation
    up = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix() @ R_ci @ a_rest
    up /= np.linalg.norm(up)
    axis = np.cross(up, [0, 0, 1])
    R_zup = R.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(up[2]))

    poses = []
    for t, m in gt:
        p, o = m.pose.position, m.pose.orientation
        R_wc = R_zup * R.from_quat([o.x, o.y, o.z, o.w])
        p_wi = R_zup.apply([p.x, p.y, p.z]) + R_wc.apply(p_ci)
        q_wi = (R_wc * R.from_matrix(R_ci)).as_quat()
        msg = PoseStamped()
        msg.header = m.header
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p_wi
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q_wi
        poses.append((t, msg))

    topics = ['/davis/left/events', '/davis/left/image_raw', '/davis/left/imu']
    with Bag(args.output, 'w') as out, Bag(args.data) as data_bag:
        for topic, m, t in tqdm(data_bag.read_messages(topics), total=data_bag.get_message_count(topics)):
            out.write(topic, m, t)
        for t, m in poses:
            out.write('/davis/left/pose', m, t)

    # initial state at the first ground truth pose; the resting accelerometer reading should be R_iw * [0, 0, 9.81],
    # attribute the difference to the accelerometer bias
    p0 = poses[0][1].pose
    q0 = [p0.orientation.w, p0.orientation.x, p0.orientation.y, p0.orientation.z]
    R_wi0 = R.from_quat([p0.orientation.x, p0.orientation.y, p0.orientation.z, p0.orientation.w])
    b_a = a_rest - R_wi0.inv().apply([0, 0, 9.81])
    print(F"Resting specific force {np.linalg.norm(a_rest):.3f} m/s^2 over the first {args.t_rest} s")
    print(F"      # initial state at {t0} from /davis/left/pose (IMU in z-up frame), sensor at rest:")
    print(F"      from: {t0}")
    print(F"      p: {[p0.position.x, p0.position.y, p0.position.z]}")
    print(F"      v: [0.0, 0.0, 0.0]")
    print(F"      q: {q0}  # [w,x,y,z]")
    print(F"      b_a: {list(b_a)}")


if __name__ == '__main__':
    main()
