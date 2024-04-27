import os
import pandas as pd
import random
import cv2

# Define the directory where your dataset is located
dataset_dir = r"C:\Users\shiva\Downloads\dense-trajectories-action-recognition-20240312T070544Z-001\dense-trajectories-action-recognition\ucf_sports_actions"

# Initialize empty lists to store video file paths and labels
video_paths = []
labels = []

# Walk through the dataset directory and its subdirectories
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".avi"):  # Assuming videos have the ".avi" extension
            # Get the full file path
            video_path = os.path.join(root, file)
            
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(video_path)
            
            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Check if the number of frames is at least 15
            if total_frames >= 15:
                # Append the video path to the list of video paths
                video_paths.append(video_path)
                
                # Extract label from the immediate subdirectory name
                label = os.path.basename(os.path.dirname(root))  # Extracting the name of the immediate subdirectory
                labels.append(label)
            else:
                print(f"Discarding video '{video_path}' as it has less than 4 frames.")

# Create a DataFrame to store video paths and labels
data = pd.DataFrame({"Video_Path": video_paths, "Label": labels})

# Group the data by label
grouped_data = data.groupby("Label")

# Initialize an empty DataFrame to store the selected videos
selected_videos = pd.DataFrame(columns=["Video_Path", "Label"])

# Iterate over each group (class) to select proportional number of videos
for _, group in grouped_data:
    # Calculate the number of videos to select for this class
    num_videos_class = int(0.25 * len(group))
    # Randomly select the videos for this class
    selected_videos = pd.concat([selected_videos, group.sample(n=num_videos_class, random_state=42)])

# Shuffle the selected videos
selected_videos = selected_videos.sample(frac=1, random_state=42).reset_index(drop=True)

# Write video paths and labels to the test.txt file
with open(r"data\test.txt", "w") as f:
    for index, row in selected_videos.iterrows():
        f.write("{} {}\n".format(row["Video_Path"], row["Label"]))

print("test.txt created successfully with a proper mix of all classes.")
