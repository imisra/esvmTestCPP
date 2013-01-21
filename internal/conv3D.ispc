/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 *  
 */
#include "esvm_common_structs.h"

unmasked void convolve2D_single(uniform float im[], uniform int irows, uniform int icols, 
		uniform float kern[], uniform int krows, uniform int kcols, uniform int apronRows, 
		uniform int apronCols, uniform int offset[], uniform float output[],uniform int orows, uniform int ocols)
{

	uniform int startX = apronRows;
	uniform int endX = irows-apronRows+(1^(krows&1));
	uniform int ksize = krows*kcols;
	uniform int colEnd = icols - apronCols+(1^(kcols&1));
	uniform int colDiff = icols - colEnd;

	uniform int dim1 = startX*icols;
	uniform float *tim = im+dim1;
	uniform float *tout = output+dim1;

	for(uniform int i=startX;i<endX;i++) {
		tim+=apronCols;
		tout+=apronCols;
		for(uniform int j=apronCols;j<colEnd;j++) {
			uniform int *offs = offset;
			uniform float *krn = kern;

			float sum=0;

			foreach(k=0 ... ksize) {
				sum += tim[*(offs+k)]*(*(krn+k));
			}

			(*tout++)=reduce_add(sum);
			tim++;
		}
		tim+=colDiff;
		tout+=colDiff;
	}

}

unmasked inline void mergeConv_single(uniform float inp[], uniform int rows, uniform int cols, uniform int nums)
{
	uniform int dim1 = rows*cols;
	for(uniform int i=0;i<rows;i++) {
		uniform float *tinp = inp+i*cols;
		for(uniform int j=0;j<cols;j++) {
			float sum=0;


			unmasked {
				foreach(k=0 ... nums) {
					sum+=(*(tinp+k*dim1));
				}
			}
			(*tinp++) = reduce_add(sum);
		}
	}
}

inline void mergeConv_single(uniform float inp[], uniform int rows, uniform int cols)
{
	uniform int dim1 = rows*cols;
	for(uniform int i=0;i<rows;i++) {
		uniform float *tinp = inp+i*cols;
		for(uniform int j=0;j<cols;j++) {
			float sum=0;

			foreach(k=0 ... 31) {
				sum+=(*(tinp+k*dim1));
			}
			(*tinp++) = reduce_add(sum);
		}
	}
}


task void convolve3D(uniform esvmHog feat[], uniform esvmHogPyr whogs[], uniform esvmArr2_f outputs[], uniform esvmArr2_i offsets[], uniform bool enablePadding)
{

	uniform esvmHog *whog = whogs->hogs[taskIndex];
//	uniform int *offset = offsets[taskIndex].arr;
	uniform int wrows = whogs->hogs[taskIndex]->rows;
	uniform int wcols = whogs->hogs[taskIndex]->cols;
	uniform int frows = feat->rows;
	uniform int fcols = feat->cols;
	uniform int fbins = feat->bins;
	uniform int apronRows = (wrows>>1);
	uniform int apronCols = (wcols>>1);
	uniform int dimf = frows*fcols;
	uniform int dimw = wrows*wcols;

	for(uniform int i=0;i<fbins;i++) {
		convolve2D_single(feat->feature+i*dimf, frows, fcols, whogs->hogs[taskIndex]->feature+i*dimw, wrows, wcols, apronRows, apronCols, offsets[taskIndex].arr, outputs[taskIndex].arr+i*dimf, frows, fcols); 
	}

	assert(fbins==31);
	mergeConv_single(outputs[taskIndex].arr, frows, fcols);
	outputs[taskIndex].rows = frows;
	outputs[taskIndex].cols = fcols;

}


//FIXME:: Since conv2D_single isn't exactly suitable for row-based thread splits, I need to write another one that is!
export void convolve2D(uniform float im[], uniform int irows, uniform int icols, uniform float kernel[], uniform int krows, uniform int kcols, 
		uniform esvmArr2_f output[], uniform int offsets[])
{	

	uniform int apronRows = (krows>>1);
	uniform int apronCols = (kcols>>1);
	
	convolve2D_single(im, irows, icols, kernel, 
				krows, kcols, apronRows, apronCols, offsets, output->arr, irows, icols); 
	
	output->rows = irows;
	output->cols = icols;

}


export void convolvePyramids(uniform esvmHogPyr feats[], uniform esvmHogPyr whogs[], uniform esvmArr2_i offsets[], uniform bool enablePadding, uniform esvmArr2_f outputs[], uniform int userTasks)
{
	for(uniform int i=0;i<feats->num;i++) {
		launch[whogs->num]convolve3D(feats->hogs[i],whogs,&(outputs[i*whogs->num]),&(offsets[i*whogs->num]),enablePadding);
	}
	sync;

}
