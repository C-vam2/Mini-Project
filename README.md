"Project2" used HOG + HOF + MBH + Humandetector + CamereMotion Estimation for computing the descriptor for a video
It requires to be build from Source code by using openCV liberaries in C++. (I Have used Visual Studio for building form the source code by linking required liberaries form the openCV lib)
After the successful build of "Project2" Project2 is a generates a .exe file that needs to be executed from commandline and it takes two arguments as input during execution:
1. Video Path for which we want to extract the descriptors
2. File path for the corresponding video where we have precomputed the bounding boxes for each frame (Don't worry we will show you how to do that using Human Detector TensorFlow model)

