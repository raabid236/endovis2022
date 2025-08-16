import os
import cv2

def resize_images(input_dir, output_dir, size=(256, 256)):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                input_path = os.path.join(root, file)
                
                # Create output folder preserving structure
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                
                output_path = os.path.join(output_folder, file)
                
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    print(f"Warning: Could not read {input_path}")
                    continue
                
                # Resize using nearest neighbor
                resized_img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
                print(f"Max value in resized_img: {resized_img.max()}")
                
                # Save the resized image
                cv2.imwrite(output_path, resized_img)
                # print(f"Saved resized image to {output_path}")

# TODO: Update these to actual paths as needed
input_directory = "/mnt/c/Users/raabi/Downloads/endovis2022/train/"
output_directory = "/mnt/c/Users/raabi/Downloads/endovis256/train/"
resize_images(input_directory, output_directory)
