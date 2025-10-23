import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def transform_polar_image(image, output_size, threshold, input_width):
    output_image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)

    num_rows = output_image.shape[0]
    ranges = np.linspace(0, 150, num_rows) 
    log_range = np.zeros_like(ranges)
    log_range[ranges >= 1] = 40 * np.log10(ranges[ranges >= 1])  
    log_range = log_range[:, None] 
    output_image = output_image + log_range
    output_image[output_image <= threshold] = 0
    
    return output_image

def process_images(polar_dir, cart_dir, output_size=(576, 384), threshold=123):
    if not os.path.exists(cart_dir):
        os.makedirs(cart_dir)

    polar_images = [f for f in os.listdir(polar_dir) if f.endswith('.png')]
    for image_file in tqdm(polar_images, desc="Processing polar images"):
        input_path = os.path.join(polar_dir, image_file)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read {input_path}")
            continue
        transformed_image = transform_polar_image(image, output_size, threshold, image.shape[1])
        output_path = os.path.join(cart_dir, image_file)
        cv2.imwrite(output_path, transformed_image)

    print(f"Processed {len(polar_images)} images from {polar_dir} to {cart_dir}")


if __name__ == "__main__":
    default_raw_img = "raw_navtech"
    default_img = "Navtech_576"

    parser = argparse.ArgumentParser(
        description="Radar polar image generation."
    )
    parser.add_argument("--input", dest="input", default=default_raw_img)
    parser.add_argument("--output", dest="output", default=default_img)

    args = parser.parse_args()
    process_images(args.input, args.output, output_size=(576, 384), threshold=123)