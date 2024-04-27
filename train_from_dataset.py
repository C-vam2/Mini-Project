import os
import cv2

# Define the directory where your dataset is located
dataset_dir = r"C:\Users\shiva\Downloads\dense-trajectories-action-recognition-20240312T070544Z-001\dense-trajectories-action-recognition\ucf_sports_actions"

# Initialize an empty list to store video file paths and labels
video_paths_and_labels = []

# Walk through the dataset directory and its subdirectories
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".avi"):  # Assuming videos have the ".avi" extension
            # Extract label from the immediate subdirectory name
            label = os.path.basename(os.path.dirname(root))  # Extracting the name of the immediate subdirectory
            # Get the full video path
            video_path = os.path.join(root, file)
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(video_path)
            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Check if the number of frames is at least 15
            if total_frames >= 15:
                # Append the video path and its corresponding label to the list
                video_paths_and_labels.append((video_path, label))
            else:
                print(f"Video '{video_path}' has less than {total_frames} frames. Discarding...")

# Define the path for the text file to save the video paths and labels
text_file_path = r"data\train.txt"

# Write video paths and labels to the text file
with open(text_file_path, "w") as f:
    for video_path, label in video_paths_and_labels:
        f.write(f"{video_path} {label}\n")

print("Video labels saved to train.txt successfully.")
