/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 *  
 */
#ifndef ESVM_HEADER
#define ESVM_HEADER 
#define ESVM_VERSION "0.1_alpha^2"
//includes

//std libraries
#include <omp.h>
#include <math.h>
#include <limits.h>

//datastructures
#include <list>
#include <queue>

//ispc includes
//#include "conv_ispc.h"
#include "hog_ispc.h"
#include "conv3D_ispc.h"

//common (between ispc and C)
#include "esvm_common_structs.h"

//utility includes
#include "esvm_utils.h"

//compatibility includes
#include "resize.hpp"

//compare
#define ESVM_COMPARE(a,b) ( ( (a) > (b) ) - ( (a) < (b) ) )
#define ESVM_INVCOMPARE(a,b) ( ( (a) < (b) ) - ( (a) > (b) ) )

//some other macros
#define ESVM_DESCEND_SORT true
#define ESVM_ASCEND_SORT false
#define ESVM_PAD true
#define ESVM_NO_PAD false
#define ESVM_THRESHOLD true
#define ESVM_NO_THRESHOLD false
#define ESVM_FLOAT_MIN (-FLT_MAX)

//inline functions

static inline float min(float x, float y) { return (x <= y ? x : y); }
static inline float max(float x, float y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

//hog computation functions
int binHist(float hist[],const float gradient[],const int best_ori[],const int visible[2],const int blocks[2],const int cellWidth,const int userTasks);
void computeNorm(float norm[], float hist[],const int blocks[2]);
esvmHog *computeHogWrapper(const char *filename,const int cellWidth=8,
		const bool enablePadding=true,const int padding=5,const int userTasks=4);
esvmHog *computeHog(int im[],const int numRows,const int numCols,
		const int numChannels,const int cellWidth,const bool enablePadding,
		const int padding,const int userTasks);
esvmHogPyr *computeHogScaleWrapper(const char *filename,
		const int cellWidth=8,const int maxLevels=200,
		const int minDimension=5,const int interval=10,const float minScale=0.1,
		const bool enablePadding=true,
		const int padding=5,const int userTasks=4,const bool useMexResize=false);

esvmHogPyr *computeHogScale(IplImage *img,const int cellWidth,const int maxLevels,
		const int minDimension,const int interval,const float minScale,const bool enablePadding,
		const int padding,const int userTasks,const bool useMexResize=false);
//#define ESVM_INTERP CV_INTER_NN
//#define ESVM_INTERP_S "CV_INTER_NN"
#define ESVM_INTERP CV_INTER_LINEAR
#define ESVM_INTERP_S "CV_INTER_LINEAR"
//#define ESVM_INTERP CV_INTER_CUBIC
//#define ESVM_INTERP_S "CV_INTER_CUBIC"


//convolution functions
esvmArr2_f *convolve2D(float im[], const int irows, const int icols, float kern[], const int krows, const int kcols);
esvmArr2_f *convolve3D(const esvmHog *feat, const esvmHog *whog, const bool enablePadding, const int userTasks);
void computeOffsets(int *offset, const int wrows, const int wcols, const int frows, const int fcols);
esvmArr2_f *ompConvolve3D(const esvmHog *feat, const esvmHog *whog, const bool enablePadding, const int userTasks);
inline void computeOffsets(int *offset, const int wrows, const int wcols, const int frows, const int fcols);

//generic vector manipulations
void subtractScalar(float arr[],const int rows,const int cols,const float scalar);
int *sort2DIndex(float arr[], const int rows,const int cols,const bool descendSort,const bool threshold, const float scoreThreshold, int *nkeep,const int padding);

//sorting
void psort(float* in, int m, int K, float* out0, int* out1);

//nms
void nms(float *boxes, const int numBoxes, const float overlapThreshold,
		int *numPruned, float *pruned);

//pipeline
esvmOutput *esvmSIMEWrapper(esvmParameters *params, const char *imageName, esvmModel *model);
esvmOutput *esvmSIME(esvmParameters *params, IplImage *img, esvmModel *model);
esvmParameters *esvmDefaultParameters();
esvmOutput **esvmMIMEWrapper(esvmParameters *params, const char **imageNames, int numImages, esvmModel *model);

#endif
