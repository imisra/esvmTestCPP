/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP 
 * 
 * demo01.cpp : Computing HOG features for an image.
 */
#include "esvm.h"
#include "esvm_utils.h"

using namespace std;


int main(int argc, char *argv[])
{
	const char *imageName = "../sample-data/apples.jpg";
	const char *outputFile = "demo01-hog.txt";
	printf("Computing HoG ... ");
	esvmHog *hog = computeHogWrapper(imageName);
	printf("Done\n");
	
	printf("HoG dimensions are %d rows, %d cols, %d bins\n",hog->rows,hog->cols,hog->bins);
	
	printf("Writing HoG ... ");
	FILE *fp = fopen(outputFile,"w");
	writeHogTxt(hog,fp);
	fclose(fp);
	printf("Done\n");
	
	
	printf("Freeing memory ... ");
	freeHog(hog);
	printf("Done\n");
	
	
	return 0;
}
