#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include<iostream>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
//#include <iostream>
//#include <fstream>
//#include <vector>
#include <string>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;//::SURF;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

//CascadeClassifier human_cascade;

//vector<Rect> humanDetection(Mat frame,int frameId) {
//	Mat frame_grey;
//	cvtColor(frame, frame_grey, COLOR_BGR2GRAY);
//	//detect humans
//	vector<Rect> humans;
//	human_cascade.detectMultiScale(frame_grey, humans);
//	/*cout << "Frame - " << frameId << endl;
//	for (size_t i = 0; i < humans.size(); i++) {
//		cout << humans[i].x << " " << humans[i].y << " " << humans[i].width << " " << humans[i].height << endl;
//	}*/
//	return humans;
//}

void writeVectorToFile(const std::vector<std::vector<float>>& vec2D, string  videoName) {
	std::string name(videoName);
	std::string filename = "D:\desc" + name + "-descriptors.txt";
	std::ofstream outFile(filename, std::ios::app);

	if (outFile.is_open()) {
		for (const auto& row : vec2D) {
			for (const auto& element : row) {
				//outFile << std::fixed << std::setprecision(5) << element << " ";
				outFile << element << " ";
			}
			outFile << std::endl;
		}
		outFile.close();
		std::cout << "2D vector content written to file " << filename << " successfully." << std::endl;
	}
	else {
		std::cerr << "Unable to open file: " << filename << std::endl;
	}
}

int main(int argc, char* argv[])
{
	//string dir = argv[1];
	/*string humanDetector = "C:\\Users\\shiva\\Downloads\\haarcascade_fullbody.xml";
	if (!human_cascade.load(humanDetector)) {
		cout << "Could not load Classifier";
		return -1;
	}
	cout << "Classifier loaded" << endl;*/

	VideoCapture capture;
	string  video = argv[1];
	bb_file = argv[2];
	//cout << "bb_file" << endl;
	string video_name;
	for (int i = video.size() - 1; i >= 0; i--) {
		if (video[i] == '\\') {
			video_name = video.substr(i);
			break;
		}
	}

	capture.open(video);

	if (!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}


	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;
	
	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo,video);

	std::vector<Frame> bb_list;
	if (bb_file) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}
	/*for (int i = 0; i < bb_list.size(); i++) {
		cout << bb_list[i].frameID << " = ";
		for (auto &it : bb_list[i].BBs) {
			cout << it.TopLeft << " " << it.BottomRight << " --- ";
		}
		cout << endl;
	}*/
	//cout << "Loaded" << endl;
	// flag 
		seqInfo.length = end_frame - start_frame + 1;

		//fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if (show_track == 1)
		namedWindow("DenseTrackStab", 0);


	Ptr<SURF> detector_surf = SURF::create(200); // 200 is the Hessian threshold
	Ptr<SURF> extractor_surf = SURF::create(true, true); // upright and extended


	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points

	vector<vector<float>>final_desc; // my changes
	while (true) {
		Mat frame;
		int i, j, c;
		// get a new frame
		capture >> frame;
		if (frame.empty())
			break;
		
		if (frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}
		cout << frame_num << endl;
		if (frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);
			
			InitPry(frame, fscales, sizes);
			
			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);
			

			frame.copyTo(image);
			cvtColor(image, prev_grey, COLOR_BGR2GRAY);

			for (int iScale = 0; iScale < scale_num; iScale++) {
				
				if (iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale - 1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);
				

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				
				for (i = 0; i < points.size(); i++) {
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
					
				}

				
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);


			human_mask = Mat::ones(frame.size(), CV_8UC1);
			//cout << human_mask << endl;
			//cout << "____________________________________________________" << endl;
			if (bb_file) {
				//cout << "Start" << endl;
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
				//cout <<"Ending" << endl;
			}
			//cout << "Checking...." << endl;
			//cout << human_mask << endl;
			detector_surf->detect(prev_grey, prev_kpts_surf, human_mask); //it detects keypoints for the human mask and stores them in the  prev_kpts _surf
			extractor_surf->compute(prev_grey, prev_kpts_surf, prev_desc_surf); // it computes the descriptors for the detected keypoints and stores them in prev_desc_surf

			frame_num++;
			//cout << frame_num << " checking...." << endl;
			continue;
		}
		
		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, COLOR_BGR2GRAY);

		// match surf features
		if (bb_file) {
			//cout << "Start" << endl;
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
			//cout << "Ending" << endl;
		}
		detector_surf->detect(grey, kpts_surf, human_mask);
		extractor_surf->compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if (pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if (countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);

		for (int iScale = 0; iScale < scale_num; iScale++) {
			if (iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale - 1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
			/*this line of code resizes the image from the previous scale (iScale - 1) in the Gaussian pyramid to the size of the image 
			at the current scale (iScale) using bilinear interpolation, and stores the result in the current scale of the Gaussian pyramid. */

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			/*this code segment initializes descriptor matrices for HOG, HOF, and MBH computation, and then computes these descriptors for
			the corresponding inputs, storing the results in the respective descriptor matrices.*/
			DescMat* hogMat = InitDescMat(height + 1, width + 1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);
	
	
			DescMat* hofMat = InitDescMat(height + 1, width + 1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height + 1, width + 1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height + 1, width + 1, mbhInfo.nBins);
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			//cout << frame_num << "---------------------------------------------------" << tracks.size() << endl;
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width - 1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height - 1);
				/* this code segment retrieves the previous position of a tracked point, rounds its coordinates to integers, and ensures that the
				coordinates are within the bounds of the image (width and height). */

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2 * x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2 * x + 1];
				/*this code snippet updates the position of the tracked point (point) based on the optical flow information (flow_pyr) at the
				specified scale (iScale). It adds the horizontal and vertical components of the optical flow vector to the previous coordinates 
				of the tracked point to obtain the updated coordinates.*/

				if (point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2 * x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2 * x + 1];

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo); //gets a 32 x 32 rectangle around the point
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);
				
				// draw the trajectories at the first scale
				if (show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image,false);

				// if the trajectory achieves the maximal length
				if (iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length + 1);
					for (int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i] * fscales[iScale];

					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i] * fscales[iScale];

					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if (IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {
						// output the trajectory
						printf("frame_num = %d\tmean_x = %f\tmean_y = %f\tvar_x = %f\tvar_y = %f\tlength = %f\tfscales[iScale] = %f\t \n", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);
						if (show_track == 1 && iScale == 0)
							DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image,true);
						// for spatio-temporal pyramid
						//printf("%f \n", std::min<float>(std::max<float>(mean_x / float(seqInfo.width), 0), 0.999));
						//printf("%f \n", std::min<float>(std::max<float>(mean_y / float(seqInfo.height), 0), 0.999));
						//printf("%f \n", std::min<float>(std::max<float>((frame_num - trackInfo.length / 2.0 - start_frame) / float(seqInfo.length), 0), 0.999));

						std::vector<float>HOG, HOF, MBHX, MBHY,TRAJ;
						vector<float>curr;
						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i) {
							TRAJ.push_back(displacement[i].x);
							TRAJ.push_back(displacement[i].y);
							//printf("%f\t%f\t \n", displacement[i].x, displacement[i].y);
						}
						
						PrintDesc(iTrack->hog, hogInfo, trackInfo,HOG);
						//printf("\nDimension of HOG = %d \n",hogInfo.dim);

						PrintDesc(iTrack->hof, hofInfo, trackInfo,HOF);
						//printf("\nDimension of HOF = %d \n", hofInfo.dim);

						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo,MBHX);
						//printf("\nDimension of MBHX = %d \n", mbhInfo.dim);

						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo,MBHY);
						//printf("\nDimension of MBHY = %d \n", mbhInfo.dim);

						for (auto& it : TRAJ)
							curr.push_back(it);
						for (auto& it : HOG)
							curr.push_back(it);
						for (auto& it : HOF)
							curr.push_back(it);
						for (auto& it : MBHX)
							curr.push_back(it);
						for (auto& it : MBHY)
							curr.push_back(it);
						cout <<"number of cols --" << curr.size() << endl;
						final_desc.push_back(curr);
						int n = final_desc.size();
						//cout << frame_num << "--" << n << "---" << final_desc[n - 1].size() << endl;
					}

					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}
			
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if (init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
			std::vector<Point2f> points(0);
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for (i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for (i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		
		//if (frame_num == 4)break;

		if (show_track == 1) {
			//cout << image.size() << endl;
			
			/*for (int i = 0; i < bb_list[frame_num].BBs.size(); i++) {
				rectangle(image, bb_list[frame_num].BBs[i].TopLeft, bb_list[frame_num].BBs[i].BottomRight, Scalar(0, 255, 0), 2);
			}*/
			imshow("DenseTrackStab", image);
			c = waitKey(3);
			if ((char)c == 27) break;
		}
		frame_num++;
	}

	writeVectorToFile(final_desc, video_name);


	if (show_track == 1)
		destroyWindow("DenseTrackStab");

	return 0;
}
