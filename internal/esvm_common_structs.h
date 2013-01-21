/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 *  
 */

#ifndef ESVM_COMMON_STRUCTS
#define ESVM_COMMON_STRUCTS
//some of the structs used by most files
struct esvmHog
{
	float *feature;
	int rows;
	int cols;
	int bins;
	int cellWidth;
	
	//if HOG is being used as the "w" vector for an SVM
	//we additionally need the fields.
	int classId;	
};

struct esvmImage
{
	int *im;
	int numRows;
	int numCols;
	int numChannels;
};

struct esvmHogPyr
{
	esvmHog **hogs;
	float *scale;
	int num;
};


struct esvmArr2_f
{
	float *arr;
	int rows;
	int cols;
};

struct esvmArr2_i
{
	int *arr;
	int rows;
	int cols;
};

struct esvmIndexedArr2D
{
	float val;
	int row;
	int col;
};

#endif
