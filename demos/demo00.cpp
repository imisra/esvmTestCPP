/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 * 
 * demo00.cpp : Using the convolution functions alone.
 */


#include "esvm.h"
#include "esvm_utils.h"

int main(int argc,char *argv[]) 
{
	
	//define kernel
	//3x3 kernel defined in row-major order.
	float kernel[9] = { 0, 0, 0, 0, 1.0, 0, 0, 0, 0 };
	const int krows = 3;
	const int kcols = 3;
	
	//matrix
	float imarr[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9};
	const int irows = 3;
	const int icols = 3;
		
	
	//perform convolution
	esvmArr2_f *output = convolve2D(imarr, irows, icols, kernel, krows, kcols);
	
	//print the result
	for(int i=0;i<output->rows;i++) {
		for(int j=0;j<output->cols;j++) {
			printf("%0.2f ",output->arr[i*output->cols+j]);
		}
		printf("\n");
	}
	
					
	//free memory
	free(output->arr);
	free(output);
	
	
	return 0;
}