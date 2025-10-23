import itertools
import numpy as np
import os
from typing import List
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset, ConcatDataset
import torchvision

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
    assert sensor_code in ['R', 'L'], f"Unknown sensor code: {sensor_code}"

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

    ext = '.png' if sensor_code == 'R' else '.bin'

    all_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(readings_filepath) if
                      os.path.splitext(f)[1] == ext]

    all_timestamps.sort()

    timestamps = []
    poses = []
    count_rejected = 0

    for ndx, ts in enumerate(all_timestamps):
        closest_ts_ndx = find_nearest_ndx(ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - ts)
        # Timestamp is in nanoseconds = 1e-9 second
        if delta > pose_time_tolerance * 1000:
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        timestamps.append(ts)
        poses.append(all_poses[closest_ts_ndx])

    timestamps = np.array(timestamps, dtype=np.int64)
    poses = np.array(poses, dtype=np.float64)     # (northing, easting) position

    print(f'{len(timestamps)} readings from sensor: {sensor_code} with valid pose, {count_rejected} rejected due to unknown pose')
    return timestamps, poses

class MulranPointCloudLoader:
    def set_properties(self):
        self.ground_plane_level = -0.9

    def read_pc(self, file_pathname: str) -> np.ndarray:
        pc = np.fromfile(file_pathname, dtype=np.float32)
        pc = np.reshape(pc, (-1, 4))[:, :3]
        return pc

class MulranSequence(Dataset):
    def __init__(self, dataset_root: str, sequence_name: str, split: str, min_displacement: float = 0.2,
                 use_radar: bool = True, use_lidar: bool = False, radar_type: str = 'polar_continental'):
        assert use_lidar or use_radar
        assert radar_type in ['polar_continental', 'nav_384'], "Invalid radar type."
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_name = sequence_name
        sequence_path = os.path.join(self.dataset_root, self.sequence_name)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'
        self.split = split
        self.min_displacement = min_displacement
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.radar_type = radar_type
        self.pose_time_tolerance = 1.0

        if radar_type == 'polar_continental':
            self.pose_file = os.path.join(sequence_path, 'local_continental_spline.txt')
            self.polar_folder = 'polar_continental'
        elif radar_type == 'nav_384':
            self.pose_file = os.path.join(sequence_path, 'local_navtech_spline.txt')
            self.polar_folder = 'nav_384_rcs'
        else:
            raise ValueError("Unsupported radar type")

        assert os.path.exists(self.pose_file), f'Cannot access pose file: {self.pose_file}'

        timestamps_l = []
        poses_l = []
        rel_filepath_l = []
        sensor_code_l = []

        if self.use_radar:
            self.rel_radar_path = os.path.join(self.sequence_name, self.polar_folder)
            radar_path = os.path.join(self.dataset_root, self.rel_radar_path)
            assert os.path.exists(radar_path), f'Cannot access radar scans: {radar_path}'

            timestamps, poses = read_poses(self.pose_file, radar_path, 'R', self.pose_time_tolerance)
            timestamps, poses = self.filter(timestamps, poses)
            rel_filepath = [os.path.join(self.rel_radar_path, f"{e}.png") for e in timestamps]
            sensor_code = ['R'] * len(timestamps)

            timestamps_l.append(timestamps)
            poses_l.append(poses)
            rel_filepath_l.append(rel_filepath)
            sensor_code_l.append(sensor_code)

        self.timestamps = np.concatenate(timestamps_l, axis=0)
        self.poses = np.concatenate(poses_l, axis=0)
        self.rel_reading_filepath = list(itertools.chain.from_iterable(rel_filepath_l))
        self.sensor_code = list(itertools.chain.from_iterable(sensor_code_l))

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_reading_filepath)
        assert len(self.timestamps) == len(self.sensor_code)

    def __len__(self):
        return len(self.rel_reading_filepath)

    def __getitem__(self, ndx):
        reading_filepath = os.path.join(self.dataset_root, self.rel_reading_filepath[ndx])
        sensor_code = self.sensor_code[ndx]
        reading = torchvision.io.read_image(reading_filepath)
        return {'reading': reading, 'pose': self.poses[ndx], 'ts': self.timestamps[ndx], 'sensor_code': sensor_code}

    def filter(self, ts: np.ndarray, poses: np.ndarray):
        positions = poses[:, :2, 3]
        displacement = np.linalg.norm(positions[:-1] - positions[1:], axis=1)
        mask = displacement > self.min_displacement
        mask = np.concatenate([np.array([True]), mask])
        ts = ts[mask]
        poses = poses[mask]
        return ts, poses

class MulranSequences(Dataset):
    def __init__(self, dataset_root: str, sequence_names: List[str], split: str, min_displacement: float = 0.2,
                 radar_type: str = 'polar_continental'):
        assert len(sequence_names) > 0
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_names = sequence_names
        self.split = split
        self.min_displacement = min_displacement
        self.radar_type = radar_type

        sequences = []
        for seq_name in self.sequence_names:
            ds = MulranSequence(self.dataset_root, seq_name, split=split, min_displacement=min_displacement,
                                use_radar=True, radar_type=radar_type)
            sequences.append(ds)

        self.dataset = ConcatDataset(sequences)


        # Concatenate positions from all sequences
        self.poses = np.zeros((len(self.dataset), 4, 4), dtype=np.float64)
        self.timestamps = np.zeros((len(self.dataset),), dtype=np.int64)
        self.rel_reading_filepath = []
        self.sensor_code = []

        for cum_size, ds in zip(self.dataset.cumulative_sizes, self.dataset.datasets):
            # Consolidated lidar positions, timestamps and relative filepaths
            self.poses[cum_size - len(ds): cum_size, :] = ds.poses
            self.timestamps[cum_size - len(ds): cum_size] = ds.timestamps
            self.rel_reading_filepath.extend(ds.rel_reading_filepath)
            self.sensor_code.extend(ds.sensor_code)

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_reading_filepath)
        assert len(self.timestamps) == len(self.sensor_code)

        # Build a kdtree based on X, Y position
        self.kdtree = KDTree(self.get_xy())



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]

    def get_xy(self):
        # Get X, Y position from (4, 4) pose
        return self.poses[:, :2, 3]

    def find_neighbours_ndx(self, position, radius):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        return neighbours.astype(np.int32)


if __name__ == '__main__':
    dataset_root = '/media/sf_Datasets/MulRan'
    sequence_names = ['ParkingLot']

    db = MulranSequences(dataset_root, sequence_names, use_radar=True, use_lidar=True)
    print(f'Number of scans in the sequence: {len(db)}')
    e = db[0]

    res = db.find_neighbours_ndx(e['pose'][:2, 3], radius=5)
    print(len(res))
    print('.')