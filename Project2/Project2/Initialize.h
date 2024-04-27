#ifndef INITIALIZE_H_
#define INITIALIZE_H_

#include "DenseTrackStab.h"
#include<string.h>
//#include <windows.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <libgen.h>.

//#define _GNU_SOURCE:
#define _GNU_SOURCE
//#include <string.h>

using namespace cv;
using namespace std;

void InitTrackInfo(TrackInfo* trackInfo, int track_length, int init_gap)
{
	trackInfo->length = track_length;
	trackInfo->gap = init_gap;
}

DescMat* InitDescMat(int height, int width, int nBins)
{
	DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;

	long size = height * width * nBins;
	//cout << size << " kkkkkkkk" << endl;
	descMat->desc = (float*)malloc(size * sizeof(float));
	memset(descMat->desc, 0, size * sizeof(float));
	return descMat;
}

void ReleDescMat(DescMat* descMat)
{
	free(descMat->desc);
	free(descMat);
}

void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int nxy_cell, int nt_cell)
{
	descInfo->nBins = nBins;
	descInfo->isHof = isHof;
	descInfo->nxCells = nxy_cell;
	descInfo->nyCells = nxy_cell;
	descInfo->ntCells = nt_cell;
	descInfo->dim = nBins * nxy_cell * nxy_cell;
	descInfo->height = size;
	descInfo->width = size;
}

void InitSeqInfo(SeqInfo* seqInfo, string video)
{
	VideoCapture capture;
	capture.open(video);

	if (!capture.isOpened())
		fprintf(stderr, "Could not initialize capturing..\n");

	// get the number of frames in the video
	int frame_num = 0;
	while (true) {
		Mat frame;
		capture >> frame;

		if (frame.empty())
			break;

		if (frame_num == 0) {
			seqInfo->width = frame.cols;
			seqInfo->height = frame.rows;
		}

		frame_num++;
	}
	seqInfo->length = frame_num;
}
void usage()
{
	fprintf(stderr, "Extract improved trajectories from a video\n\n");
	fprintf(stderr, "Usage: DenseTrackStab video_file [options]\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h                        Display this message and exit\n");
	fprintf(stderr, "  -S [start frame]          The start frame to compute feature (default: S=0 frame)\n");
	fprintf(stderr, "  -E [end frame]            The end frame for feature computing (default: E=last frame)\n");
	fprintf(stderr, "  -L [trajectory length]    The length of the trajectory (default: L=15 frames)\n");
	fprintf(stderr, "  -W [sampling stride]      The stride for dense sampling feature points (default: W=5 pixels)\n");
	fprintf(stderr, "  -N [neighborhood size]    The neighborhood size for computing the descriptor (default: N=32 pixels)\n");
	fprintf(stderr, "  -s [spatial cells]        The number of cells in the nxy axis (default: nxy=2 cells)\n");
	fprintf(stderr, "  -t [temporal cells]       The number of cells in the nt axis (default: nt=3 cells)\n");
	fprintf(stderr, "  -A [scale number]         The number of maximal spatial scales (default: 8 scales)\n");
	fprintf(stderr, "  -I [initial gap]          The gap for re-sampling feature points (default: 1 frame)\n");
	fprintf(stderr, "  -H [human bounding box]   The human bounding box file to remove outlier matches (default: None)\n");
}


#endif /*INITIALIZE_H_*/
