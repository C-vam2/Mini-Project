
Project2: Descriptor Extraction System

Overview:
Project2 is a computer vision system designed for computing video descriptors using a combination of Histogram of Oriented Gradients (HOG), Histogram of Optical Flow (HOF), Motion Boundary Histograms (MBH), Human Detector, and Camera Motion Estimation techniques. It is implemented in C++ using the OpenCV libraries and built using Visual Studio. The system generates an executable file (.exe) upon successful compilation, which is executed from the command line with two input arguments.

Execution Requirements:
To execute Project2, follow these steps:

Build the project from the provided source code, linking the required OpenCV libraries.
Execute the generated .exe file from the command line.
Provide two arguments during execution:
Video Path: The path to the video for which descriptors are to be extracted.
Bounding Box File Path: The path to the file containing precomputed bounding boxes for each frame of the video.
Input Parameters:

Video Path: Path to the video file for which descriptors are to be computed.
Bounding Box File Path: Path to the file containing precomputed bounding boxes for each frame of the video. (This can be generated using a Human Detector TensorFlow model.)
Note: The bounding box file is essential for accurately computing descriptors for human-related activities.

Usage Example:
Project2.exe <Video_Path> <Bounding_Box_File_Path>
Example:

Project2.exe video.mp4 bounding_boxes.txt
Output:

The system processes the video and computes descriptors based on the provided bounding boxes. The output may vary depending on the specific implementation and configuration.
