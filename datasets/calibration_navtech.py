import os
import cv2
import numpy as np
from tqdm import tqdm

def transform_polar_image(image, output_size, threshold):
    input_height, input_width = image.shape
    distance_scale = 150 / 384
    height = int(150 / distance_scale)  
    assert input_height == 3424, f"Unexpected input height: {input_height}"
    resized_image = cv2.resize(image, (input_width, height), interpolation=cv2.INTER_AREA)
    start_300_deg = int((300 / 360) * input_width)  
    end_360_deg = input_width  
    start_0_deg = 0  
    end_60_deg = int((60 / 360) * input_width)
    front_left = resized_image[:, start_300_deg:end_360_deg] 
    front_right = resized_image[:, start_0_deg:end_60_deg] 
    combined_image = np.hstack((front_left, front_right))
    output_image = cv2.resize(combined_image, output_size, interpolation=cv2.INTER_AREA)
    num_rows = output_image.shape[0]
    ranges = np.linspace(0, 150, num_rows) 
    log_range = np.zeros_like(ranges)
    log_range[ranges >= 1] = 40 * np.log10(ranges[ranges >= 1]) 
    log_range = log_range[:, None] 
    output_image = output_image + log_range
    output_image[output_image <= threshold] = 0
    return output_image

def process_images(polar_dir, cart_dir, output_size=(384, 192), threshold=100):
    if not os.path.exists(cart_dir):
        os.makedirs(cart_dir)
    polar_images = [f for f in os.listdir(polar_dir) if f.endswith('.png')]
    for image_file in tqdm(polar_images, desc="Processing polar images"):
        input_path = os.path.join(polar_dir, image_file)
        output_path = os.path.join(cart_dir, image_file)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read {input_path}")
            continue
        transformed_image = transform_polar_image(image, output_size, threshold)
        cv2.imwrite(output_path, transformed_image)
    print(f"Processed {len(polar_images)} images from {polar_dir} to {cart_dir}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    raw_dir = os.path.join(current_dir, 'HeRCULES/Mountaon/01/raw_navtech')  
    output_dir = os.path.join(current_dir, 'HeRCULES/Mountaon/01/Navtech_before_calibration')  
    process_images(raw_dir, output_dir, output_size=(192, 384), threshold=123)