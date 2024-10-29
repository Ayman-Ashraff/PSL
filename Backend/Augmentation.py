import os
import cv2
import math

# Paths
dataset_videos = r'/Users/kayan/Downloads/archive (1)/WLPSL/Videos'  # Original video dataset path with action folders containing videos
images_dataset = r'/Users/kayan/Downloads/NewDataSet/Output'  # Output path for augmented frames


# Autogenerate actions (categories) based on folders in dataset_videos
def get_actions(dataset_videos):
    actions = [folder for folder in os.listdir(dataset_videos) if os.path.isdir(os.path.join(dataset_videos, folder))]
    if not actions:
        print(f"No action folders found in '{dataset_videos}'. Ensure each action has a dedicated folder.")
        return []
    print(f"Actions found: {actions}")
    return actions

# Ensure output directory structure is set up based on actions and individual videos
def setup_directory_structure(actions, images_dataset):
    for action in actions:
        action_path = os.path.join(images_dataset, action)
        os.makedirs(action_path, exist_ok=True)
    print("Directory structure set up successfully.")

# Extract frames from each video in each action folder
def extract_frames(actions, dataset_videos, images_dataset):
    for action in actions:
        action_video_path = os.path.join(dataset_videos, action)
        action_output_path = os.path.join(images_dataset, action)
        videos = [v for v in os.listdir(action_video_path) if os.path.isfile(os.path.join(action_video_path, v))]
        
        for video in videos:
            video_path = os.path.join(action_video_path, video)
            video_name = os.path.splitext(video)[0]
            frame_folder = os.path.join(action_output_path, video_name, "1")
            os.makedirs(frame_folder, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                print(f"Skipping video with no frames: {video}")
                cap.release()
                continue

            toskip = max(1, math.floor(frame_count / 50))
            frame_num, count = 0, 1

            while cap.isOpened() and count <= 50:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                frame_num += toskip
                ret, frame = cap.read()

                if ret:
                    frame_path = os.path.join(frame_folder, f"{count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    print(f"Saved frame {count} of {video} to {frame_path}")
                    count += 1
                else:
                    break
            cap.release()

# Crop frames
def crop_frames(images_dataset):
    for action in os.listdir(images_dataset):
        action_path = os.path.join(images_dataset, action)
        
        # Ignore non-directory entries
        if not os.path.isdir(action_path):
            continue
        
        for video_folder in os.listdir(action_path):
            frames_path = os.path.join(action_path, video_folder, '1')
            cropped_folder = os.path.join(action_path, video_folder, '2')
            
            # Ignore non-directory entries
            if not os.path.isdir(os.path.join(action_path, video_folder)):
                continue
            
            os.makedirs(cropped_folder, exist_ok=True)

            if os.path.exists(frames_path):
                for frame in os.listdir(frames_path):
                    img = cv2.imread(os.path.join(frames_path, frame))
                    if img is not None:
                        cropped_img = img[100:-30, 50:-50]
                        cv2.imwrite(os.path.join(cropped_folder, frame), cropped_img)
                        print(f"Cropped frame {frame} saved in {cropped_folder}")


# Rotate frames
def rotate_frames(angle, images_dataset):
    folder_map = { -5: '3', 5: '4', -10: '5', 10: '6', -15: '7', 15: '8', -20: '9', 20: '10' }
    folder = folder_map.get(angle, '10')

    for action in os.listdir(images_dataset):
        action_path = os.path.join(images_dataset, action)
        for video_folder in os.listdir(action_path):
            frames_path = os.path.join(action_path, video_folder, '1')
            rotated_folder = os.path.join(action_path, video_folder, folder)
            os.makedirs(rotated_folder, exist_ok=True)

            if os.path.exists(frames_path):
                for frame in os.listdir(frames_path):
                    img = cv2.imread(os.path.join(frames_path, frame))
                    if img is not None:
                        (h, w) = img.shape[:2]
                        rot_point = (w // 2, h // 2)
                        rot_mat = cv2.getRotationMatrix2D(rot_point, angle, 1.0)
                        rotated_img = cv2.warpAffine(img, rot_mat, (w, h))
                        cv2.imwrite(os.path.join(rotated_folder, frame), rotated_img)
                        print(f"Rotated frame {frame} saved in {rotated_folder}")

# Resize frames
def resize_frames(scale, images_dataset):
    folder_map = { 50: '11', 150: '12', 200: '13', 250: '14', 300: '15' }
    folder = folder_map.get(scale, '15')

    for action in os.listdir(images_dataset):
        action_path = os.path.join(images_dataset, action)
        for video_folder in os.listdir(action_path):
            frames_path = os.path.join(action_path, video_folder, '1')
            resized_folder = os.path.join(action_path, video_folder, folder)
            os.makedirs(resized_folder, exist_ok=True)

            if os.path.exists(frames_path):
                for frame in os.listdir(frames_path):
                    img = cv2.imread(os.path.join(frames_path, frame))
                    if img is not None:
                        width = int(img.shape[1] * scale / 100)
                        height = int(img.shape[0] * scale / 100)
                        resized_img = cv2.resize(img, (width, height))
                        cv2.imwrite(os.path.join(resized_folder, frame), resized_img)
                        print(f"Resized frame {frame} saved in {resized_folder}")

# Main execution
if __name__ == '__main__':
    actions = get_actions(dataset_videos)
    setup_directory_structure(actions, images_dataset)

    extract_frames(actions, dataset_videos, images_dataset)
    crop_frames(images_dataset)
    rotate_frames(-5, images_dataset)
    resize_frames(50, images_dataset)

    print("All augmentations completed.")
