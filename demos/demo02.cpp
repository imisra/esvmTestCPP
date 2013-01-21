/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP 
 * 
 * demo02.cpp : Computing HOG feature pyramid
 */
#include "esvm.h"
#include "esvm_utils.h"

using namespace std;


int main(int argc, char *argv[])
{
	const char *imageName = "../sample-data/apples.jpg";
	const char *outputFile = "demo01-hog.txt";
	printf("Computing HoG Pyramid ... ");
	esvmHogPyr *hogpyr = computeHogScaleWrapper(imageName);
	printf("Done\n");
	
	printf("HoG Pyramid levels are %d\n",hogpyr->num);	
	
	printf("Freeing memory ... ");
	freeHogPyramid(hogpyr);
	printf("Done\n");
	
	
	return 0;
}
