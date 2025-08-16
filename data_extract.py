import cv2
import os
import zipfile
import tempfile
import shutil

# TODO: set these to your actual paths
zip_folder = "/mnt/c/Users/raabi/Downloads/train/"  # folder containing all video_xx.zip files
output_root = "/mnt/c/Users/raabi/Downloads/endovis2022/train/"

os.makedirs(output_root, exist_ok=True)

# Process each zip file
for zip_name in os.listdir(zip_folder):
    if not zip_name.endswith(".zip"):
        continue

    case_name = os.path.splitext(zip_name)[0]
    print(f"Processing {case_name}...")

    zip_path = os.path.join(zip_folder, zip_name)

    # Temporary extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        case_folder = os.path.join(temp_dir)
        video_path = os.path.join(case_folder, "video_left.avi")
        seg_folder = os.path.join(case_folder, "segmentation")

        # Output folders
        case_output_folder = os.path.join(output_root, case_name)
        frames_folder = os.path.join(case_output_folder, "frames_original")
        seg_output_folder = os.path.join(case_output_folder, "segmentation")

        os.makedirs(frames_folder, exist_ok=True)
        os.makedirs(seg_output_folder, exist_ok=True)

        # Copy segmentation masks to output
        for f in sorted(os.listdir(seg_folder)):
            src = os.path.join(seg_folder, f)
            dst = os.path.join(seg_output_folder, f)
            shutil.copy2(src, dst)

        # Get frame indices from segmentation filenames
        seg_files = sorted(os.listdir(seg_folder))
        frame_numbers = sorted(int(os.path.splitext(f)[0]) for f in seg_files)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ERROR: Cannot open {video_path}")
            continue

        # Extract relevant frames
        for frame_idx in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"  WARNING: Could not read frame {frame_idx}")
                continue

            filename = f"{frame_idx:09d}.png"
            save_path = os.path.join(frames_folder, filename)
            cv2.imwrite(save_path, frame)

        cap.release()
        print(f"  Saved {len(frame_numbers)} frames + masks for {case_name}")

print("All cases processed.")
