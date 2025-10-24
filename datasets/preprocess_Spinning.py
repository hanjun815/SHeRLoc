import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def transform_polar_image(image, output_size, threshold, start_angle, end_angle, input_width):
    input_height, input_width = image.shape
    distance_scale = 150 / 384
    height = int(150 / distance_scale) 
    assert input_height == 3424, f"Unexpected input height: {input_height}"
    resized_image = cv2.resize(image, (input_width, height), interpolation=cv2.INTER_AREA)

    start_angle = start_angle % 360
    end_angle = end_angle % 360

    start_pixel = int((start_angle / 360) * input_width)
    end_pixel = int((end_angle / 360) * input_width)

    if start_pixel >= end_pixel:
        start_pixel_1 = start_pixel
        end_pixel_1 = input_width
        start_pixel_2 = 0
        end_pixel_2 = end_pixel
        combined_image = np.hstack((
            resized_image[:, start_pixel_1:end_pixel_1],
            resized_image[:, start_pixel_2:end_pixel_2]
        ))
    else:
        combined_image = resized_image[:, start_pixel:end_pixel]

    output_image = cv2.resize(combined_image, output_size, interpolation=cv2.INTER_AREA)

    num_rows = output_image.shape[0]
    ranges = np.linspace(0, 150, num_rows)  

    log_range = np.zeros_like(ranges)
    log_range[ranges >= 1] = 40 * np.log10(ranges[ranges >= 1])

    log_range = log_range[:, None] 
    output_image = output_image + log_range
    output_image[output_image <= threshold] = 0
    C_correct = 2.1025
    output_image[output_image > 0] += C_correct
    
    return output_image


def process_images(polar_dir, cart_dir, output_size=(192, 384), threshold=100, n=6):
    if not os.path.exists(cart_dir):
        os.makedirs(cart_dir)

    polar_images = [f for f in os.listdir(polar_dir) if f.endswith('.png')]
    for image_file in tqdm(polar_images, desc="Processing polar images"):
        input_path = os.path.join(polar_dir, image_file)

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read {input_path}")
            continue

        for i in range(n):
            start_angle = 300 + (i * 60)
            end_angle = start_angle + 120
            view_folder = os.path.join(cart_dir, str(i))
            if not os.path.exists(view_folder):
                os.makedirs(view_folder)

            transformed_image = transform_polar_image(image, output_size, threshold, start_angle, end_angle, image.shape[1])

            output_path = os.path.join(view_folder, image_file)
            cv2.imwrite(output_path, transformed_image)

    print(f"Processed {len(polar_images)} images from {polar_dir} to {cart_dir}")


if __name__ == "__main__":
    default_raw_img = "raw_navtech"
    default_img = "Navtech"

    parser = argparse.ArgumentParser(
        description="Radar Multi-view polar image generation."
    )
    parser.add_argument("--input", dest="input", default=default_raw_img)
    parser.add_argument("--output", dest="output", default=default_img)

    args = parser.parse_args()
    process_images(args.input, args.output, output_size=(192, 384), threshold=123, n=36)
