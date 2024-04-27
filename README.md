Wang, H., & Schmid, C. (2013). Action Recognition with Improved Trajectories. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) (pp. 3551-3558).

# Project2: Descriptor Extraction System (Improved Dense Trajectories)

## Overview:
Project2 is a computer vision system designed for computing video descriptors using a combination of Histogram of Oriented Gradients (HOG), Histogram of Optical Flow (HOF), Motion Boundary Histograms (MBH), Human Detector, and Camera Motion Estimation techniques. It is implemented in C++ using the OpenCV libraries and built using Visual Studio. The system generates an executable file (.exe) upon successful compilation, which is executed from the command line with two input arguments.

## Execution Requirements:
To execute Project2, follow these steps:
1. Build the project from the provided source code, linking the required OpenCV libraries.
2. Execute the generated .exe file from the command line.
3. Provide two arguments during execution:
    - **Video Path:** The path to the video for which descriptors are to be extracted.
    - **Bounding Box File Path:** The path to the file containing precomputed bounding boxes for each frame of the video.

## Input Parameters:
1. **Video Path:** Path to the video file for which descriptors are to be computed.
2. **Bounding Box File Path:** Path to the file containing precomputed bounding boxes for each frame of the video. (This can be generated using a Human Detector TensorFlow model.)

**Note:** The bounding box file is essential for accurately computing descriptors for human-related activities.

## Usage Example:
Project2.exe <Video_Path> <Bounding_Box_File_Path>


**Example:**
Project2.exe video.mp4 bounding_boxes.txt


## Output:
The system processes the video and computes descriptors based on the provided bound

# Human Detection and Descriptor Extraction Pipeline

## Overview

This pipeline integrates the "Project2" video descriptor extraction system with the TensorFlow Object Detection API for real-time human detection. It allows for the extraction of descriptors from videos using "Project2", followed by training a machine learning model using the extracted descriptors.

## Setup Instructions

1. **Build "Project2" Source Code:**
   - Build the "Project2" source code to generate the executable file (`Project2.exe`).
   - Ensure that the necessary OpenCV libraries are linked correctly.

2. **Install TensorFlow:**
   - Install TensorFlow in Python. You can follow the official installation instructions provided by TensorFlow.

3. **Install TensorFlow Object Detection API:**
   - Follow the instructions in [this Medium article](https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6) to install the TensorFlow Object Detection API.

4. **Update Paths in Code:**
   - Make sure that all paths in the code are correctly set, including:
     - Path to your dataset.
     - Path to your `Project2.exe` file.
     - Path to your extracted descriptors, if applicable.

5. **Execute run.py:**
   - Open `run.py` and set `already_computed_descriptors=False` in the `main` function.
   - Execute `run.py` to start the descriptor extraction process.
   - Wait for the descriptors to be extracted from the videos.

6. **Train ML Model:**
   - Ensure that you have `train.txt` and `test.txt` files containing paths to videos and their corresponding labels for training and testing, respectively.
   - Train your machine learning model using the extracted descriptors.

## Note
- The `train.txt` file should contain paths to training videos along with their corresponding labels for model training.
- The `test.txt` file should contain paths to testing videos along with their corresponding labels for model evaluation.

