/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 *  
 */
#ifndef MEX_RESIZE
#define MEX_RESIZE
int *mexResize(float *im, const int rows, const int cols, const int channels,
		const int rRows, const int rCols);
int *mexResizeTranspose(float *im, const int rows, const int cols, const int channels,
		const int rRows, const int rCols);

#endif
