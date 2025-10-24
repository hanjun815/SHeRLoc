import os
import numpy as np
from pathlib import Path
from PIL import Image
import random

class RadarEgoVelocityEstimatorConfig:
    def __init__(self):
        self.min_dist = 0.0  
        self.max_dist = 150.0 
        self.min_db = -50 
        self.azimuth_thresh_deg = 60.0 
        self.elevation_thresh_deg = 45.0  
        self.filter_min_z = -5.0  
        self.filter_max_z = 10.0  
        self.thresh_zero_velocity = 0.1 
        self.allowed_outlier_percentage = 0.1  
        self.use_ransac = True  
        self.N_ransac_points = 3  
        self.ransac_iter = 20  
        self.inlier_thresh = 0.2  
        self.use_odr = False  
        self.min_speed_odr = 0.5  
        self.sigma_zero_velocity_x = 0.05  
        self.sigma_zero_velocity_y = 0.05 
        self.sigma_zero_velocity_z = 0.05  
        self.sigma_offset_radar_x = 0.1  
        self.sigma_offset_radar_y = 0.1  
        self.sigma_offset_radar_z = 0.1  
        self.max_sigma_x = 1.0  
        self.max_sigma_y = 1.0 
        self.max_sigma_z = 1.0 
        self.doppler_velocity_correction_factor = 1.0 
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
        ('v', np.float32),
        ('r', np.float32),
        ('RCS', np.int8),
        ('azimuth', np.float32),
        ('elevation', np.float32),
    ])
    data = np.frombuffer(trimmed_content, dtype=dtype)
    return np.vstack([data['x'], data['y'], data['z'], data['v'], data['r'], data['RCS'], data['azimuth'], data['elevation']]).T


def create_polar_image(points, max_range=150, angle_span=120, threshold=123):
    range_bins = 384
    angle_bins = 192

    x, y, rcs = points[:, 0], points[:, 1], points[:, 5]
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))

    mask = (theta >= -angle_span/2) & (theta <= angle_span/2) & (r <= max_range)
    r, theta, rcs = r[mask], theta[mask], rcs[mask]

    r_pixel = (r / max_range * range_bins).astype(int)
    theta_pixel = angle_bins - ((theta + angle_span/2) / angle_span * angle_bins).astype(int) - 1

    r_pixel = np.clip(r_pixel, 0, range_bins - 1)
    theta_pixel = np.clip(theta_pixel, 0, angle_bins - 1)

    polar_image = np.zeros((range_bins, angle_bins), dtype=np.uint8)
    for rp, tp, rcs_val in zip(r_pixel, theta_pixel, rcs):
        pixel_value = np.clip((75 + rcs_val) * 2, 0, 255)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr, cc = rp + dr, tp + dc
                if 0 <= rr < range_bins and 0 <= cc < angle_bins:
                    polar_image[rr, cc] = max(polar_image[rr, cc], pixel_value)

    polar_image[polar_image <= threshold] = 0
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
    return combined_image if combined_image is not None else np.zeros((384, 192, 3), dtype=np.uint8)


def solve_3d_lsq(radar_data, config, estimate_sigma=True):
    H = radar_data[:, :3]  
    y = radar_data[:, 3]  
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


def solve_3d_lsq_ransac(radar_data, config):
    H_all = radar_data[:, :3]
    y_all = radar_data[:, 3]
    n_points = radar_data.shape[0]

    if n_points < config.N_ransac_points:
        return False, None, None, []

    inlier_idx_best = []
    v_r_best = None
    P_v_r_best = None

    for _ in range(config.ransac_iter):
        idx = random.sample(range(n_points), config.N_ransac_points)
        radar_data_iter = radar_data[idx]
        success, v_r, _, _ = solve_3d_lsq(radar_data_iter, config, estimate_sigma=False)

        if not success:
            continue

        err = np.abs(y_all - H_all @ v_r)
        inlier_idx = [i for i, e in enumerate(err) if e < config.inlier_thresh]
        if len(inlier_idx) > len(inlier_idx_best):
            inlier_idx_best = inlier_idx
            v_r_best = v_r

    if not inlier_idx_best:
        return False, None, None, []

    radar_data_inlier = radar_data[inlier_idx_best]
    success, v_r, P_v_r, sigma_v_r = solve_3d_lsq(radar_data_inlier, config, estimate_sigma=True)
    if not success:
        return False, None, None, []

    return True, v_r, P_v_r, inlier_idx_best


def apply_ransac_and_filter(data, config):
    x, y, z, v_doppler, r, rcs, azimuth, elevation = data.T

    valid_mask = (
        (r > config.min_dist) & (r < config.max_dist) &
        (rcs > config.min_db) &
        (np.abs(azimuth) < config.azimuth_thresh_deg) &
        (np.abs(elevation) < config.elevation_thresh_deg) &
        (z > config.filter_min_z) 
    )
    valid_data = data[valid_mask]

    if len(valid_data) <= 2:
        print("Not enough valid targets for velocity estimation.")
        return data, 0.0, 0.0  

    radar_data = np.zeros((len(valid_data), 4))
    radar_data[:, 0] = valid_data[:, 0] / valid_data[:, 4]  
    radar_data[:, 1] = valid_data[:, 1] / valid_data[:, 4]  
    radar_data[:, 2] = valid_data[:, 2] / valid_data[:, 4]  
    radar_data[:, 3] = -valid_data[:, 3] * config.doppler_velocity_correction_factor 

    v_dopplers = np.abs(radar_data[:, 3])
    n = int(len(v_dopplers) * (1.0 - config.allowed_outlier_percentage))
    if n >= len(v_dopplers) or n < 0:
        n = len(v_dopplers) - 1
    median = np.partition(v_dopplers, n)[n] if n > 0 else np.median(v_dopplers)

    if median < config.thresh_zero_velocity:
        print("Zero velocity detected!")
        v_r = np.array([0.0, 0.0, 0.0])
        P_v_r = np.diag([
            config.sigma_zero_velocity_x**2,
            config.sigma_zero_velocity_y**2,
            config.sigma_zero_velocity_z**2
        ])
        success = True
        inlier_idx = [i for i, vd in enumerate(v_dopplers) if vd < config.thresh_zero_velocity]
    else:
        success, v_r, P_v_r, inlier_idx = solve_3d_lsq_ransac(radar_data, config)
        if not success:
            print("Velocity estimation failed.")
            return data, 0.0, 0.0  

    if success:
        sigma_v_r = np.sqrt(np.diag(P_v_r))
        if (sigma_v_r[0] < config.max_sigma_x and
            sigma_v_r[1] < config.max_sigma_y and
            sigma_v_r[2] < config.max_sigma_z):
            H = radar_data[:, :3]
            predicted_v = H @ v_r
            dynamic_mask = np.abs(radar_data[:, 3] - predicted_v) > config.inlier_thresh
            static_idx = [i for i, is_dynamic in enumerate(dynamic_mask) if not is_dynamic]
            filtered_data = valid_data[static_idx]
            alpha, beta = v_r[0], v_r[1] 
            return filtered_data, alpha, beta

    print("Velocity estimation failed or sigma exceeded limits.")
    return data, 0.0, 0.0 


def save_polar_images_with_combined_scans(input_folder, output_folder, rcs_threshold=-50):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(input_folder.glob("*.bin"))
    config = RadarEgoVelocityEstimatorConfig()

    for i in range(len(bin_files)):
        print(f"Processing {bin_files[i].name} with up to 4 previous scans...")

        scans = []
        start_index = max(0, i - 4)
        for j in range(start_index, i + 1):
            data = load_bin_file_with_structure(bin_files[j])
            filtered_data, alpha, beta = apply_ransac_and_filter(data, config)
            scans.append(filtered_data)

        print(f"Ego-velocity: alpha={alpha}, beta={beta}")

        combined_image = combine_scans_to_image(scans)

        polar_image_pil = Image.fromarray(combined_image)
        output_path = output_folder / (bin_files[i].stem + ".png")
        polar_image_pil.save(output_path)

    print(f"All files processed. Images saved to {output_folder}")


if __name__ == "__main__":
    current_dir = os.getcwd()
    raw_dir = os.path.join(current_dir, "HeRCULES/Mountaon/01/raw_continental")
    output_dir = os.path.join(current_dir, "HeRCULES/Mountaon/01/Continental_before_calibration")
    save_polar_images_with_combined_scans(raw_dir, output_dir)