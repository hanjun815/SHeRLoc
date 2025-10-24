import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from misc.utils import ModelParams
from datasets.raw import read_poses


class HeRCULES(Dataset):
    def __init__(self, model_params: ModelParams, root="/code/SHeRLoc/datasets/HeRCULES", phase="train",
                 min_displacement=0.3, pos_threshold=5.0, neg_threshold=25.0, pose_time_tolerance=1.0, n_views=6):
        super().__init__()
        self.task = model_params.task
        self.root = root
        self.phase = phase
        self.min_displacement = min_displacement
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.pose_time_tolerance = pose_time_tolerance
        self.n_views = n_views

        if self.phase == "train":
            self.sequences = [
                'Mountain/01', 'Mountain/02', 'Mountain/03',
                'Bridge/01', 'Stream/02', 'Parking_Lot/03', 'Parking_Lot/04'
            ]
        elif self.phase == "val":
            self.sequences = ['Parking_Lot/01', 'Parking_Lot/02']
            # self.sequences = ['Parking_Lot/01']

        else:
            raise ValueError(f"Invalid phase: {self.phase}. Must be 'train' or 'val'.")

        self.data = []
    
        if self.task == "Hetero":
            for seq in self.sequences:
                seq_path = os.path.join(self.root, seq)

                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Continental_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Continental"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                pos_timestamps, pos_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Navtech/0"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_images = sorted([os.path.join(seq_path, "Continental", f"{ts}.png") for ts in query_timestamps])
                pos_images = sorted([os.path.join(seq_path, "Navtech/0", f"{ts}.png") for ts in pos_timestamps])
                query_coords = query_poses[:, :2, 3]
                pos_coords = pos_poses[:, :2, 3]
                filtered_indices = [0]
                for i in range(1, len(query_coords)):
                    displacement = np.linalg.norm(query_coords[i] - query_coords[filtered_indices[-1]])
                    if displacement >= self.min_displacement:
                        filtered_indices.append(i)

                query_timestamps = np.array(query_timestamps)[filtered_indices]
                query_images = np.array(query_images)[filtered_indices]
                query_coords = np.array(query_coords)[filtered_indices]

                self.data.append({
                    'query_images': query_images,
                    'pos_images': pos_images,
                    'query_coords': np.array(query_coords),
                    'pos_coords': np.array(pos_coords)
                })

        elif self.task == "4D":
            for seq in self.sequences:
                seq_path = os.path.join(self.root, seq)

                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Continental_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Continental"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                pos_timestamps, pos_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Continental_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Continental"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_images = sorted([os.path.join(seq_path, "Continental", f"{ts}.png") for ts in query_timestamps])
                pos_images = sorted([os.path.join(seq_path, "Continental", f"{ts}.png") for ts in pos_timestamps])
                query_coords = query_poses[:, :2, 3]
                pos_coords = pos_poses[:, :2, 3]
                filtered_indices = [0]
                for i in range(1, len(query_coords)):
                    displacement = np.linalg.norm(query_coords[i] - query_coords[filtered_indices[-1]])
                    if displacement >= self.min_displacement:
                        filtered_indices.append(i)

                query_timestamps = np.array(query_timestamps)[filtered_indices]
                query_images = np.array(query_images)[filtered_indices]
                query_coords = np.array(query_coords)[filtered_indices]

                self.data.append({
                    'query_images': query_images,
                    'pos_images': pos_images,
                    'query_coords': np.array(query_coords),
                    'pos_coords': np.array(pos_coords)
                })

        elif self.task == "Spinning":
            for seq in self.sequences:
                seq_path = os.path.join(self.root, seq)

                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Navtech_576"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                pos_timestamps, pos_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Navtech_576"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_images = sorted([os.path.join(seq_path, "Navtech_576", f"{ts}.png") for ts in query_timestamps])
                pos_images = sorted([os.path.join(seq_path, "Navtech_576", f"{ts}.png") for ts in pos_timestamps])
                query_coords = query_poses[:, :2, 3]
                pos_coords = pos_poses[:, :2, 3]
                filtered_indices = [0]
                for i in range(1, len(query_coords)):
                    displacement = np.linalg.norm(query_coords[i] - query_coords[filtered_indices[-1]])
                    if displacement >= self.min_displacement:
                        filtered_indices.append(i)

                query_timestamps = np.array(query_timestamps)[filtered_indices]
                query_images = np.array(query_images)[filtered_indices]
                query_coords = np.array(query_coords)[filtered_indices]

                self.data.append({
                    'query_images': query_images,
                    'pos_images': pos_images,
                    'query_coords': np.array(query_coords),
                    'pos_coords': np.array(pos_coords)
                })

        elif self.task == "LiDAR":
            for seq in self.sequences:
                seq_path = os.path.join(self.root, seq)

                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Aeva_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Aeva"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                pos_timestamps, pos_poses = read_poses(
                    poses_filepath=os.path.join(seq_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(seq_path, "Navtech/0"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_images = sorted([os.path.join(seq_path, "Aeva", f"{ts}.png") for ts in query_timestamps])
                pos_images = sorted([os.path.join(seq_path, "Navtech/0", f"{ts}.png") for ts in pos_timestamps])
                query_coords = query_poses[:, :2, 3]
                pos_coords = pos_poses[:, :2, 3]
                filtered_indices = [0]
                for i in range(1, len(query_coords)):
                    displacement = np.linalg.norm(query_coords[i] - query_coords[filtered_indices[-1]])
                    if displacement >= self.min_displacement:
                        filtered_indices.append(i)

                query_timestamps = np.array(query_timestamps)[filtered_indices]
                query_images = np.array(query_images)[filtered_indices]
                query_coords = np.array(query_coords)[filtered_indices]

                self.data.append({
                    'query_images': query_images,
                    'pos_images': pos_images,
                    'query_coords': np.array(query_coords),
                    'pos_coords': np.array(pos_coords)
                })
        else:
            raise ValueError(f"Invalid task: {self.task}.")


    def __len__(self):
        return sum(len(seq_data['query_images']) for seq_data in self.data)

    def compute_similarity(self, img1, img2):
        """Compute similarity between two images using Fourier domain convolution"""
        if not torch.is_tensor(img1):
            img1 = torch.tensor(img1).float()
        if not torch.is_tensor(img2):
            img2 = torch.tensor(img2).float()
        if len(img1.shape) == 2:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) == 2:
            img2 = img2.unsqueeze(0)
        fft1 = torch.fft.fft2(img1)
        fft2 = torch.fft.fft2(img2)
        conv_result = fft1 * torch.conj(fft2)
        correlation = torch.fft.ifft2(conv_result).real
        similarity = torch.max(correlation) / (torch.norm(img1) * torch.norm(img2))
        return similarity

    def __getitem__(self, index):
        for seq_data in self.data:
            if index < len(seq_data['query_images']):
                break
            index -= len(seq_data['query_images'])

        query_image_path = seq_data['query_images'][index]
        query_coords = seq_data['query_coords']
        pos_coords = seq_data['pos_coords']
        query = np.array(Image.open(query_image_path)).astype(np.float32)
        query = torch.tensor(query).unsqueeze(0)  # Add channel dimension
        query_timestamp = int(os.path.splitext(os.path.basename(query_image_path))[0])
        pos_timestamps = [int(os.path.splitext(os.path.basename(p))[0]) for p in seq_data['pos_images']]
        pos_distances = np.linalg.norm(query_coords[index] - pos_coords, axis=1)
        closest_pos_idx = np.argmin(np.abs(np.array(pos_timestamps) - query_timestamp))
        pos_indices = [closest_pos_idx]
        neg_indices = np.where(pos_distances > self.neg_threshold)[0]
        if len(neg_indices) == 0:
            neg_indices = [np.argmax(pos_distances)]
        selected_pos_indices = np.random.choice(pos_indices, size=1, replace=True)
        positives = []
        sim_p = []
        for pos_idx in selected_pos_indices:
            pos_image_path = seq_data['pos_images'][pos_idx]
            pos = np.array(Image.open(pos_image_path)).astype(np.float32)
            pos_tensor = torch.tensor(pos).unsqueeze(0)
            positives.append(pos_tensor)
            sim_p.append(self.compute_similarity(query, pos_tensor))
        positives = torch.stack(positives)
        sim_p = torch.tensor(sim_p)
        selected_neg_indices = np.random.choice(neg_indices, size=5, replace=True)
        negatives = []
        sim_n = []
        for neg_idx in selected_neg_indices:
            neg_image_path = seq_data['pos_images'][neg_idx]
            neg = np.array(Image.open(neg_image_path)).astype(np.float32)
            neg_tensor = torch.tensor(neg).unsqueeze(0)
            negatives.append(neg_tensor)
            sim_n.append(self.compute_similarity(query, neg_tensor))
        negatives = torch.stack(negatives)
        sim_n = torch.tensor(sim_n)

        return query, positives, negatives, sim_p, sim_n



class HeRCULES_test(Dataset):
    def __init__(self, model_params: ModelParams, root="/code/SHeRLoc/datasets/HeRCULES", phase="test", pose_time_tolerance=1.0, min_distance=5.0, min_query_distance=0.3):
    # def __init__(self, root="/code/SHeRLoc/src/dataset/HeRCULES", phase="test", pose_time_tolerance=1.0, min_distance=5.0, min_query_distance=0.5):
        super().__init__()
        self.task = model_params.task
        self.root = root
        self.phase = phase
        self.pose_time_tolerance = pose_time_tolerance
        self.min_distance = min_distance
        self.min_query_distance = min_query_distance
        
        self.query_images = []
        self.map_images = []
        self.query_coords = []
        self.map_coords = []
        self.query_timestamps = []
        self.map_timestamps = []

        # sequences = [('Sports_Complex/01', 'Sports_Complex/01')]
        sequences = [('Sports_Complex/02', 'Sports_Complex/02')]
        # sequences = [('Sports_Complex/03', 'Sports_Complex/03')]
        # sequences = [('Library/01', 'Library/01')]
        # sequences = [('Library/02', 'Library/02')]
        # sequences = [('Library/03', 'Library/03')]
        # sequences = [('River_Island/01', 'River_Island/01')]
        # sequences = [('River_Island/02', 'River_Island/02')]
        # sequences = [('River_Island/03', 'River_Island/03')]

        # sequences = [('Sports_Complex/01', 'Sports_Complex/02')]
        # sequences = [('Sports_Complex/01', 'Sports_Complex/03')]
        # sequences = [('Library/01', 'Library/02')]
        # sequences = [('Library/01', 'Library/03')]
        # sequences = [('River_Island/01', 'River_Island/02')]
        # sequences = [('River_Island/01', 'River_Island/03')]


        def filter_queries(query_coords, query_image_paths, min_query_distance):
            filtered_query_coords = []
            filtered_query_images = []
            
            prev_coord = None
            for i, coord in enumerate(query_coords):
                if prev_coord is None or np.linalg.norm(coord - prev_coord) > min_query_distance:
                    filtered_query_coords.append(coord)
                    filtered_query_images.append(query_image_paths[i])
                    prev_coord = coord
            
            return np.array(filtered_query_coords), filtered_query_images

        if self.task == "Hetero":
            for map_seq, query_seq in sequences:
                map_path = os.path.join(self.root, map_seq)
                query_path = os.path.join(self.root, query_seq) 


                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(query_path, "Continental_gt.txt"),
                    readings_filepath=os.path.join(query_path, "Continental"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                
                map_timestamps, map_poses = read_poses(
                    poses_filepath=os.path.join(map_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(map_path, "Navtech/0"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_image_paths = [os.path.join(query_path, "Continental", f"{ts}.png") for ts in query_timestamps]
                num_views = 36 
                all_view_map_paths = []
                for view_idx in range(num_views):
                    view_folder = os.path.join(map_path, "Navtech", str(view_idx))
                    if os.path.exists(view_folder):
                        view_images = [os.path.join(view_folder, f"{ts}.png") for ts in map_timestamps]
                        all_view_map_paths.append(view_images)
                    else:
                        all_view_map_paths.append([])  
                query_coords = query_poses[:, :2, 3]
                map_coords_base = map_poses[:, :2, 3]  
                query_coords, query_image_paths = filter_queries(query_coords, query_image_paths, self.min_query_distance)


                for i, (query_coord, query_path) in enumerate(zip(query_coords, query_image_paths)):
                    distances = np.linalg.norm(map_coords_base - query_coord, axis=1)
                    if (distances <= self.min_distance).any():
                        distances = np.linalg.norm(map_coords_base - query_coord, axis=1)
                        nearest_map_idx = np.argmin(distances)
                        selected_map_images = []
                        selected_map_coords = []
                        for view_idx in range(num_views):
                            if all_view_map_paths[view_idx]:  
                                selected_map_images.append(all_view_map_paths[view_idx][nearest_map_idx])
                                selected_map_coords.append(map_coords_base[nearest_map_idx])
                            else:
                                selected_map_images.append(None)  
                                selected_map_coords.append(None)
                        self.query_images.append(query_path)
                        self.query_coords.append(query_coord)
                        self.map_images.append(selected_map_images)  
                        self.map_coords.append(selected_map_coords) 

            self.query_coords = np.array(self.query_coords)
            self.map_coords = np.array(self.map_coords)


        elif self.task == "4D":
            for map_seq, query_seq in sequences:
                map_path = os.path.join(self.root, map_seq)
                query_path = os.path.join(self.root, query_seq) 

                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(query_path, "Continental_gt.txt"),
                    readings_filepath=os.path.join(query_path, "Continental"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                map_timestamps, map_poses = read_poses(
                    poses_filepath=os.path.join(map_path, "Continental_gt.txt"),
                    readings_filepath=os.path.join(map_path, "Continental"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_image_paths = [os.path.join(query_path, "Continental", f"{ts}.png") for ts in query_timestamps]
                map_image_paths = [os.path.join(map_path, "Continental", f"{ts}.png") for ts in map_timestamps]
                query_coords = query_poses[:, :2, 3]
                map_coords = map_poses[:, :2, 3]
                for i, query_coord in enumerate(query_coords):
                    distances = np.linalg.norm(map_coords - query_coord, axis=1)
                    if (distances <= self.min_distance).any():
                        self.query_images.append(query_image_paths[i])
                        self.query_coords.append(query_coord)
                        self.map_images = map_image_paths  
                        self.map_coords = map_coords

            self.query_coords = np.array(self.query_coords)
            self.map_coords = np.array(self.map_coords)



        elif self.task == "Spinning":
            for map_seq, query_seq in sequences:
                map_path = os.path.join(self.root, map_seq)
                query_path = os.path.join(self.root, query_seq) 

                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(query_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(query_path, "Navtech_576"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                map_timestamps, map_poses = read_poses(
                    poses_filepath=os.path.join(map_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(map_path, "Navtech_576"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_image_paths = [os.path.join(query_path, "Navtech_576", f"{ts}.png") for ts in query_timestamps]
                map_image_paths = [os.path.join(map_path, "Navtech_576", f"{ts}.png") for ts in map_timestamps]
                query_coords = query_poses[:, :2, 3]
                map_coords = map_poses[:, :2, 3]
                for i, query_coord in enumerate(query_coords):
                    distances = np.linalg.norm(map_coords - query_coord, axis=1)
                    if (distances <= self.min_distance).any():
                        self.query_images.append(query_image_paths[i])
                        self.query_coords.append(query_coord)
                        self.map_images = map_image_paths  
                        self.map_coords = map_coords

            self.query_coords = np.array(self.query_coords)
            self.map_coords = np.array(self.map_coords)



        elif self.task == "LiDAR":
            for map_seq, query_seq in sequences:
                map_path = os.path.join(self.root, map_seq)
                query_path = os.path.join(self.root, query_seq) 

                query_timestamps, query_poses = read_poses(
                    poses_filepath=os.path.join(query_path, "Aeva_gt.txt"),
                    readings_filepath=os.path.join(query_path, "Aeva"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )
                
                map_timestamps, map_poses = read_poses(
                    poses_filepath=os.path.join(map_path, "Navtech_gt.txt"),
                    readings_filepath=os.path.join(map_path, "Navtech/0"),
                    sensor_code="R",
                    pose_time_tolerance=self.pose_time_tolerance
                )

                query_image_paths = [os.path.join(query_path, "Aeva", f"{ts}.png") for ts in query_timestamps]
                num_views = 36 
                all_view_map_paths = []
                for view_idx in range(num_views):
                    view_folder = os.path.join(map_path, "Navtech", str(view_idx))
                    if os.path.exists(view_folder):
                        view_images = [os.path.join(view_folder, f"{ts}.png") for ts in map_timestamps]
                        all_view_map_paths.append(view_images)
                    else:
                        all_view_map_paths.append([])  
                query_coords = query_poses[:, :2, 3]
                map_coords_base = map_poses[:, :2, 3]  
                query_coords, query_image_paths = filter_queries(query_coords, query_image_paths, self.min_query_distance)


                for i, (query_coord, query_path) in enumerate(zip(query_coords, query_image_paths)):
                    distances = np.linalg.norm(map_coords_base - query_coord, axis=1)
                    if (distances <= self.min_distance).any():
                        query_timestamp = int(os.path.splitext(os.path.basename(query_path))[0])
                        map_timestamps_array = np.array(map_timestamps)
                        # timestamp_differences = np.abs(map_timestamps_array - query_timestamp)
                        distances = np.linalg.norm(map_coords_base - query_coord, axis=1)
                        # nearest_map_idx = np.argmin(timestamp_differences)
                        nearest_map_idx = np.argmin(distances)
                        selected_map_images = []
                        selected_map_coords = []
                        for view_idx in range(num_views):
                            if all_view_map_paths[view_idx]:  
                                selected_map_images.append(all_view_map_paths[view_idx][nearest_map_idx])
                                selected_map_coords.append(map_coords_base[nearest_map_idx])
                            else:
                                selected_map_images.append(None)  
                                selected_map_coords.append(None)
                        self.query_images.append(query_path)
                        self.query_coords.append(query_coord)
                        self.map_images.append(selected_map_images)  
                        self.map_coords.append(selected_map_coords) 

            self.query_coords = np.array(self.query_coords)
            self.map_coords = np.array(self.map_coords)


    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index):
        if self.task in ("Hetero", "LiDAR"):
            query_image_path = self.query_images[index]
            query_pos = self.query_coords[index]
            query = np.array(Image.open(query_image_path)).astype(np.float32)
            query = torch.tensor(query).unsqueeze(0)  
            map_image_paths = self.map_images[index]
            map_coords = self.map_coords[index]
            map_data_list = []
            valid_map_coords = []
            for path, coord in zip(map_image_paths, map_coords):
                if path is not None and coord is not None:
                    map_data = np.array(Image.open(path)).astype(np.float32)
                    map_data = torch.tensor(map_data).unsqueeze(0)
                    map_data_list.append(map_data)
                    valid_map_coords.append(coord)
            return query, map_data_list, query_pos, valid_map_coords

        elif self.task in ("4D", "Spinning"):
            query_image_path = self.query_images[index]
            query_pos = self.query_coords[index]
            query = np.array(Image.open(query_image_path)).astype(np.float32)
            query = torch.tensor(query).unsqueeze(0)  
            distances = np.linalg.norm(self.map_coords - query_pos, axis=1)
            nearest_map_idx = np.argmin(distances)
            map_image_path = self.map_images[nearest_map_idx]
            map_pos = self.map_coords[nearest_map_idx]
            map_data = np.array(Image.open(map_image_path)).astype(np.float32)
            map_data = torch.tensor(map_data).unsqueeze(0)
            return query, map_data, query_pos, map_pos
