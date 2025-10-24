import numpy as np
import os

def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])
    return R

def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx

def read_poses(poses_filepath: str, readings_filepath: str, sensor_code: str, pose_time_tolerance: float = 1.0):
    assert sensor_code in ['png', 'jpg'], f"Unknown sensor code: {sensor_code}"

    with open(poses_filepath, "r") as f:
        lines = f.readlines()

    n = len(lines)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    all_poses = np.zeros((n, 4, 4), dtype=np.float64)

    for ndx, line in enumerate(lines):
        temp = [float(e.strip()) for e in line.split()]
        assert len(temp) == 8, f"Invalid line in spline file: {temp}"
        time, x, y, z, qx, qy, qz, qw = temp

        system_timestamps[ndx] = int(time)

        R = quaternion_to_rotation_matrix([qx, qy, qz, qw])
        T = np.array([x, y, z])
        all_poses[ndx] = np.eye(4)
        all_poses[ndx][:3, :3] = R
        all_poses[ndx][:3, 3] = T

    sorted_ndx = np.argsort(system_timestamps)
    system_timestamps = system_timestamps[sorted_ndx]
    all_poses = all_poses[sorted_ndx]

    ext = '.png' if sensor_code == 'png' else '.jpg'

    all_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(readings_filepath) if
                      os.path.splitext(f)[1] == ext]

    all_timestamps.sort()

    timestamps = []
    poses = []
    count_rejected = 0

    for ndx, ts in enumerate(all_timestamps):
        closest_ts_ndx = find_nearest_ndx(ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - ts)
        if delta > pose_time_tolerance * 1000:
            count_rejected += 1
            continue

        timestamps.append(ts)
        poses.append(all_poses[closest_ts_ndx])

    timestamps = np.array(timestamps, dtype=np.int64)
    poses = np.array(poses, dtype=np.float64) 

    print(f'{len(timestamps)} readings from sensor: {sensor_code} with valid pose, {count_rejected} rejected due to unknown pose')
    return timestamps, poses
