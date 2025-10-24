import numpy as np
import glob
import os
import torch
import torch.optim as optim
from scipy.optimize import minimize_scalar
import cv2

def load_image_from_folder(folder_path, timestamp):
    file_list = glob.glob(os.path.join(folder_path, f"{timestamp}.png"))
    if len(file_list) == 0:
        raise ValueError(f"No {timestamp}.png.")
    image = cv2.imread(file_list[0], cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {file_list[0]}")
    return image.astype(np.float64)

def find_closest_timestamp(polar_timestamps, nav_timestamp):
    return min(polar_timestamps, key=lambda t: abs(t - nav_timestamp))

def objective_mae(K, nav_image, polar_image, mask):
    adjusted_nav = nav_image.copy()
    adjusted_nav[mask] += K  
    
    diff = adjusted_nav[mask] - polar_image[mask]
    if diff.size == 0:
        return float('inf')
    return np.mean(np.abs(diff))

def objective_mse(K, nav_image, polar_image, mask):
    adjusted_nav = nav_image.copy()
    adjusted_nav[mask] += K
    diff = adjusted_nav[mask] - polar_image[mask]
    if diff.size == 0:
        return float('inf')
    return np.mean(diff**2)

def objective_huber(K, nav_image, polar_image, mask, delta=10):
    adjusted_nav = nav_image.copy()
    adjusted_nav[mask] += K
    diff = adjusted_nav[mask] - polar_image[mask]
    if diff.size == 0:
        return float('inf')
    abs_diff = np.abs(diff)
    loss = np.where(abs_diff <= delta, 0.5 * diff**2,
                    delta * (abs_diff - 0.5 * delta))
    return np.mean(loss)

def objective_log_cosh(K, nav_image, polar_image, mask):
    adjusted_nav = nav_image.copy()
    adjusted_nav[mask] += K
    diff = adjusted_nav[mask] - polar_image[mask]
    if diff.size == 0:
        return float('inf')
    return np.mean(np.log(np.cosh(diff)))

def objective_cauchy(K, nav_image, polar_image, mask, c=10):
    adjusted_nav = nav_image.copy()
    adjusted_nav[mask] += K
    diff = adjusted_nav[mask] - polar_image[mask]
    if diff.size == 0:
        return float('inf')
    return 0.5 * (c**2) * np.mean(np.log1p((diff / c)**2))

def process_images_individual_all(nav_folder, polar_folder):
    nav_files = sorted(glob.glob(os.path.join(nav_folder, "*.png")))
    polar_files = sorted(glob.glob(os.path.join(polar_folder, "*.png")))
    nav_timestamps = [int(os.path.basename(file).split('.')[0]) for file in nav_files]
    polar_timestamps = [int(os.path.basename(file).split('.')[0]) for file in polar_files]
    loss_objectives = {
        "mae": objective_mae,
        "mse": objective_mse,
        "huber": objective_huber,
        "log_cosh": objective_log_cosh,
        "cauchy": objective_cauchy
    }
    results = {key: [] for key in loss_objectives}
    for nav_timestamp in nav_timestamps:
        closest_polar_timestamp = find_closest_timestamp(polar_timestamps, nav_timestamp)
        nav_image = load_image_from_folder(nav_folder, nav_timestamp)
        polar_image = load_image_from_folder(polar_folder, closest_polar_timestamp)
        mask = (nav_image != 0) & (polar_image != 0)
        for loss_name, obj_func in loss_objectives.items():
            if loss_name == "huber":
                res = minimize_scalar(
                    lambda K: obj_func(K, nav_image, polar_image, mask, delta=10),
                    method='bounded', bounds=(-50, 50)
                )
            elif loss_name == "cauchy":
                res = minimize_scalar(
                    lambda K: obj_func(K, nav_image, polar_image, mask, c=10),
                    method='bounded', bounds=(-50, 50)
                )
            else:
                res = minimize_scalar(
                    lambda K: obj_func(K, nav_image, polar_image, mask),
                    method='bounded', bounds=(-50, 50)
                )
            results[loss_name].append(res.x)
            print(f"[Individual] Timestamp {nav_timestamp} - {loss_name}: k = {res.x:.4f}")
    overall_mean = {loss: np.mean(results[loss]) for loss in results}
    print("="*50)
    for loss, mean_val in overall_mean.items():
        print(f"[Individual] Overall Average k ({loss}): {mean_val:.4f}")
    print("="*50)
    return overall_mean, results

def process_images_joint_all(nav_folder, polar_folder, lambda_reg=0.1, lr=1.0, max_iter=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Joint Optimization] Using device: {device}")
    nav_files = sorted(glob.glob(os.path.join(nav_folder, "*.png")))
    polar_files = sorted(glob.glob(os.path.join(polar_folder, "*.png")))
    nav_timestamps = [int(os.path.basename(file).split('.')[0]) for file in nav_files]
    polar_timestamps = [int(os.path.basename(file).split('.')[0]) for file in polar_files]
    image_pairs = []
    for nav_timestamp in nav_timestamps:
        closest_polar_timestamp = find_closest_timestamp(polar_timestamps, nav_timestamp)
        nav_image = load_image_from_folder(nav_folder, nav_timestamp)
        polar_image = load_image_from_folder(polar_folder, closest_polar_timestamp)
        nav_t  = torch.from_numpy(nav_image).float().to(device)
        polar_t= torch.from_numpy(polar_image).float().to(device)
        mask_t = ((nav_t != 0) & (polar_t != 0))
        image_pairs.append((nav_t, polar_t, mask_t))
    N = len(image_pairs)

    def torch_mae(diff):
        if diff.numel() == 0:
            return torch.tensor(float('inf'), device=device)
        return diff.abs().mean()

    def torch_mse(diff):
        if diff.numel() == 0:
            return torch.tensor(float('inf'), device=device)
        return (diff**2).mean()

    def torch_huber(diff, delta=10.0):
        if diff.numel() == 0:
            return torch.tensor(float('inf'), device=device)
        abs_diff = diff.abs()
        quadratic = 0.5 * diff.pow(2)
        linear = delta * (abs_diff - 0.5 * delta)
        return torch.where(abs_diff <= delta, quadratic, linear).mean()

    def torch_log_cosh(diff):
        if diff.numel() == 0:
            return torch.tensor(float('inf'), device=device)
        return torch.log(torch.cosh(diff)).mean()

    def torch_cauchy(diff, c=10.0):
        if diff.numel() == 0:
            return torch.tensor(float('inf'), device=device)
        return 0.5 * (c**2) * torch.log1p((diff / c).pow(2)).mean()

    loss_funcs = {
        "mae":      (torch_mae, None),
        "mse":      (torch_mse, None),
        "huber":    (torch_huber, 10.0),
        "log_cosh": (torch_log_cosh, None),
        "cauchy":   (torch_cauchy, 10.0)
    }
    joint_results = {}
    overall_mean_joint = {}
    for loss_name, (loss_func, extra_param) in loss_funcs.items():
        print(f"\n[Joint] Start optimization for loss = {loss_name}")
        k_param = torch.zeros(N, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = optim.LBFGS([k_param], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for i, (nav_t, polar_t, mask_t) in enumerate(image_pairs):
                adj_nav = nav_t.clone()  
                adj_nav[mask_t] += k_param[i]
                diff = adj_nav[mask_t] - polar_t[mask_t]
                if extra_param is not None:
                    L = loss_func(diff, extra_param)
                else:
                    L = loss_func(diff)
                total_loss = total_loss + L
            diff_k = k_param[1:] - k_param[:-1]
            smooth_term = (diff_k**2).sum()
            total_loss = total_loss + lambda_reg * smooth_term
            total_loss.backward()
            return total_loss
        optimizer.step(closure)
        k_vector = k_param.detach().cpu().numpy().copy()
        joint_results[loss_name] = k_vector
        overall_mean_joint[loss_name] = np.mean(k_vector)
        for i, ts in enumerate(nav_timestamps):
            print(f"[Joint] Timestamp {ts} - {loss_name}: k = {k_vector[i]:.4f}")
        print(f"[Joint] Overall Average k ({loss_name}): {overall_mean_joint[loss_name]:.4f}")
        print("-"*40)
    print("="*50)
    for loss, mean_val in overall_mean_joint.items():
        print(f"[Joint] Final Overall Average k ({loss}): {mean_val:.4f}")
    print("="*50)
    return overall_mean_joint, joint_results

if __name__ == "__main__":
    navtech_folder = "HeRCULES/Mountaon/01/Navtech_before_calibration"
    continental_folder = "HeRCULES/Mountaon/01/Continental_before_calibration"
    overall_mean_ind, k_values_ind = process_images_individual_all(navtech_folder, continental_folder)
    overall_mean_joint, k_vector_joint = process_images_joint_all(navtech_folder, continental_folder,lambda_reg=0.1,max_iter=100)
    print("\nFinal Returned Average k:")
    print("Individual:")
    for loss, mean_val in overall_mean_ind.items():
        print(f"  {loss}: {mean_val:.4f}")
    print("Joint:")
    for loss, mean_val in overall_mean_joint.items():
        print(f"  {loss}: {mean_val:.4f}")