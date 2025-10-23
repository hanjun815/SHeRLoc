import numpy as np
from pathlib import Path
from PIL import Image
import random
import argparse

class LidarEgoVelocityEstimatorConfig:
    def __init__(self):
        self.min_dist = 0.0  
        self.max_dist = 150.0 
        self.min_reflectivity = 0.1  
        self.filter_min_z = -5.0  
        self.filter_max_z = 10.0  
        self.thresh_zero_velocity = 0.1  
        self.allowed_outlier_percentage = 0.1  
        self.use_ransac = True  
        self.N_ransac_points = 3 
        self.ransac_iter = 10  
        self.inlier_thresh = 0.2  
        self.use_odr = False 
        self.min_speed_odr = 0.5  
        self.sigma_zero_velocity_x = 0.05  
        self.sigma_zero_velocity_y = 0.05  
        self.sigma_zero_velocity_z = 0.05  
        self.sigma_offset_lidar_x = 0.1  
        self.sigma_offset_lidar_y = 0.1 
        self.sigma_offset_lidar_z = 0.1 
        self.max_sigma_x = 1.0
        self.max_sigma_y = 1.0 
        self.max_sigma_z = 1.0 
        self.condition_number_threshold = 1e3 


def load_bin_file_with_structure(file_path, record_size=29):
    with open(file_path, 'rb') as f:
        file_content = f.read()

    num_records = len(file_content) // record_size
    trimmed_content = file_content[:num_records * record_size]

    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('reflectivity', np.float32),
        ('velocity', np.float32),
        ('time_offset_ns', np.int32),
        ('line_index', np.uint8),
        ('intensity', np.float32)
    ])
    data = np.frombuffer(trimmed_content, dtype=dtype)
    return np.vstack([data['x'], data['y'], data['z'], data['velocity'], 
                      np.sqrt(data['x']**2 + data['y']**2 + data['z']**2), 
                      data['reflectivity']]).T


def create_polar_image(points, max_range=150, angle_span=120, threshold=0.7):
    range_bins = 384 
    angle_bins = 192

    x, y, reflectivity = points[:, 0], points[:, 1], points[:, 5]
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))

    mask = (theta >= -angle_span/2) & (theta <= angle_span/2) & (r <= max_range)
    r, theta, reflectivity = r[mask], theta[mask], reflectivity[mask]

    r_pixel = (r / max_range * range_bins).astype(int)
    theta_pixel = angle_bins - ((theta + angle_span/2) / angle_span * angle_bins).astype(int) - 1

    r_pixel = np.clip(r_pixel, 0, range_bins - 1)
    theta_pixel = np.clip(theta_pixel, 0, angle_bins - 1)

    polar_image = np.zeros((range_bins, angle_bins), dtype=np.uint8)
    for rp, tp, ref_val in zip(r_pixel, theta_pixel, reflectivity):
        ref_val = 120 + ref_val
        pixel_value = ref_val
        polar_image[rp, tp] = max(polar_image[rp, tp], pixel_value)

    polar_image[polar_image <= 125] = 0

    return polar_image


def combine_scans_to_image(scans):
    combined_image = None
    for scan in scans:
        if len(scan) == 0:
            continue
        colored_image = create_polar_image(scan)
        if combined_image is None:
            combined_image = colored_image
        else:
            combined_image = np.maximum(combined_image, colored_image)
    return combined_image if combined_image is not None else np.zeros((512, 256), dtype=np.uint8)


def solve_3d_lsq(lidar_data, config, estimate_sigma=True):
    H = lidar_data[:, :3]  
    y = lidar_data[:, 3]  
    HTH = H.T @ H

    U, s, Vt = np.linalg.svd(HTH)
    cond = s[0] / s[-1] if s[-1] != 0 else float('inf')
    if cond > config.condition_number_threshold:
        return False, None, None, None

    v_r = np.linalg.lstsq(H, y, rcond=None)[0]

    if estimate_sigma:
        e = H @ v_r - y
        P_v_r = (e.T @ e) * np.linalg.inv(HTH) / (H.shape[0] - 3)
        sigma_v_r = np.sqrt(np.diag(P_v_r))
        return True, v_r, P_v_r, sigma_v_r
    return True, v_r, None, None


def solve_3d_lsq_ransac(lidar_data, config):
    H_all = lidar_data[:, :3]
    y_all = lidar_data[:, 3]
    n_points = lidar_data.shape[0]

    if n_points < config.N_ransac_points:
        return False, None, None, []

    inlier_idx_best = []
    v_r_best = None
    P_v_r_best = None

    for _ in range(config.ransac_iter):
        idx = random.sample(range(n_points), config.N_ransac_points)
        lidar_data_iter = lidar_data[idx]
        success, v_r, _, _ = solve_3d_lsq(lidar_data_iter, config, estimate_sigma=False)

        if not success:
            continue

        err = np.abs(y_all - H_all @ v_r)
        inlier_idx = [i for i, e in enumerate(err) if e < config.inlier_thresh]
        if len(inlier_idx) > len(inlier_idx_best):
            inlier_idx_best = inlier_idx
            v_r_best = v_r

    if not inlier_idx_best:
        return False, None, None, []

    lidar_data_inlier = lidar_data[inlier_idx_best]
    success, v_r, P_v_r, sigma_v_r = solve_3d_lsq(lidar_data_inlier, config, estimate_sigma=True)
    if not success:
        return False, None, None, []

    return True, v_r, P_v_r, inlier_idx_best


def apply_ransac_and_filter(data, config):
    x, y, z, velocity, r, reflectivity = data.T

    valid_mask = (
        (r > config.min_dist) & (r < config.max_dist) &
        (reflectivity > config.min_reflectivity) &
        (z > config.filter_min_z) & (z < config.filter_max_z)
    )
    valid_data = data[valid_mask]

    if len(valid_data) <= 2:
        print("Not enough valid targets for velocity estimation.")
        return data, 0.0, 0.0

    lidar_data = np.zeros((len(valid_data), 4))
    lidar_data[:, 0] = valid_data[:, 0] / valid_data[:, 4]
    lidar_data[:, 1] = valid_data[:, 1] / valid_data[:, 4] 
    lidar_data[:, 2] = valid_data[:, 2] / valid_data[:, 4]  
    lidar_data[:, 3] = -valid_data[:, 3]  

    velocities = np.abs(lidar_data[:, 3])
    n = int(len(velocities) * (1.0 - config.allowed_outlier_percentage))
    if n >= len(velocities) or n < 0:
        n = len(velocities) - 1
    median = np.partition(velocities, n)[n] if n > 0 else np.median(velocities)

    if median < config.thresh_zero_velocity:
        print("Zero velocity detected!")
        v_r = np.array([0.0, 0.0, 0.0])
        P_v_r = np.diag([
            config.sigma_zero_velocity_x**2,
            config.sigma_zero_velocity_y**2,
            config.sigma_zero_velocity_z**2
        ])
        success = True
        inlier_idx = [i for i, vd in enumerate(velocities) if vd < config.thresh_zero_velocity]
    else:
        success, v_r, P_v_r, inlier_idx = solve_3d_lsq_ransac(lidar_data, config)
        if not success:
            print("Velocity estimation failed.")
            return data, 0.0, 0.0

    if success:
        sigma_v_r = np.sqrt(np.diag(P_v_r))
        if (sigma_v_r[0] < config.max_sigma_x and
            sigma_v_r[1] < config.max_sigma_y and
            sigma_v_r[2] < config.max_sigma_z):
            H = lidar_data[:, :3]
            predicted_v = H @ v_r
            dynamic_mask = np.abs(lidar_data[:, 3] - predicted_v) > config.inlier_thresh
            static_idx = [i for i, is_dynamic in enumerate(dynamic_mask) if not is_dynamic]
            filtered_data = valid_data[static_idx]
            alpha, beta = v_r[0], v_r[1]  
            return filtered_data, alpha, beta

    print("Velocity estimation failed or sigma exceeded limits.")
    return data, 0.0, 0.0


def save_polar_images_with_combined_scans(input_folder, output_folder, reflectivity_threshold=0.1):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(input_folder.glob("*.bin"))
    config = LidarEgoVelocityEstimatorConfig()

    for i in range(len(bin_files)):
        print(f"Processing {bin_files[i].name} with up to 4 previous scans...")

        scans = []
        data = load_bin_file_with_structure(bin_files[i])
        filtered_data, alpha, beta = apply_ransac_and_filter(data, config)
        scans.append(filtered_data)

        print(f"Ego-velocity: alpha={alpha}, beta={beta}")

        combined_image = combine_scans_to_image(scans)

        polar_image_pil = Image.fromarray(combined_image)
        output_path = output_folder / (bin_files[i].stem + ".png")
        polar_image_pil.save(output_path)

    print(f"Every file has been processed. Images saved to {output_folder}")


if __name__ == "__main__":
    default_pcl = "raw_aeva"
    default_img = "Aeva"

    parser = argparse.ArgumentParser(
        description="LiDAR polar image generation with optional RANSAC-based ego-velocity filtering."
    )
    parser.add_argument("--pcl_folder", "--input", dest="pcl_folder", default=default_pcl)
    parser.add_argument("--img_folder", "--output", dest="img_folder", default=default_img)

    args = parser.parse_args()
    save_polar_images_with_combined_scans(args.pcl_folder, args.img_folder )