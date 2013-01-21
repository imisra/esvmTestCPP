/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 *  
 */
#ifndef ESVM_UTILS_HEADER
#define ESVM_UTILS_HEADER 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <map>

//opencv includes
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

//common (between ispc and C)
#include "esvm_common_structs.h"

//timing
#include "CycleTimer.h"


//compiler macros
#if __GNUC__ < 3
    #define __builtin_expect(foo,bar) (foo)
    #define expect(foo,bar) (foo)
#else
    #define expect(foo,bar) __builtin_expect((long)(foo),bar)
#endif
   
/**
   From the Linux kernel.
   */
#define __likely(foo) expect((foo),1)
#define __unlikely(foo) expect((foo),0)

//mem alignment
//#define ESVM_CONFIG_ALIGN_MEMORY
#define ESVM_CONFIG_ALIGN_MEMORY_BOUNDARY 16

#define ESVM_MAX_EXEMPLARS 1000
#define ESVM_MAX_FILENAME_CHARS 2048

//to access bounding boxes
#define ESVM_BOX_DIM 8
#define ARR_I(arr,i) (arr+i*ESVM_BOX_DIM)

#define ARR_RMIN_P(arr) (*(arr))
#define ARR_CMIN_P(arr) (*(arr+1))
#define ARR_RMAX_P(arr) (*(arr+2))
#define ARR_CMAX_P(arr) (*(arr+3))
#define ARR_SCORE_P(arr) (*(arr+4))
#define ARR_CLASS_P(arr) (*(arr+5)) 
#define ARR_SCALE_P(arr) (*(arr+6))
#define ARR_EXID_P(arr) (*(arr+7))

#define ARR_RMIN(arr,i) (*(ARR_I(arr,i)))
#define ARR_CMIN(arr,i) (*(ARR_I(arr,i)+1))
#define ARR_RMAX(arr,i) (*(ARR_I(arr,i)+2))
#define ARR_CMAX(arr,i) (*(ARR_I(arr,i)+3))
#define ARR_SCORE(arr,i) (*(ARR_I(arr,i)+4))
#define ARR_CLASS(arr,i) (*(ARR_I(arr,i)+5)) 
#define ARR_SCALE(arr,i) (*(ARR_I(arr,i)+6))
#define ARR_EXID(arr,i) (*(ARR_I(arr,i)+7))

#define BOX_I(box,i) ARR_I(box->arr,i)
#define BOX_RMIN(box,i) (*(BOX_I(box,i)))
#define BOX_CMIN(box,i) (*(BOX_I(box,i)+1))
#define BOX_RMAX(box,i) (*(BOX_I(box,i)+2))
#define BOX_CMAX(box,i) (*(BOX_I(box,i)+3))
#define BOX_SCORE(box,i) (*(BOX_I(box,i)+4))
#define BOX_CLASS(box,i) ((int)(*(BOX_I(box,i)+5))) 
#define BOX_SCALE(box,i) (*(BOX_I(box,i)+6)) 
#define BOX_EXID(box,i) ((int)(*(BOX_I(box,i)+7)))


#define ARR_COPY(srcArr,i,dstArr,j) { \
	ARR_RMIN(dstArr,j) = ARR_RMIN(srcArr,i); \
	ARR_CMIN(dstArr,j) = ARR_CMIN(srcArr,i); \
	ARR_RMAX(dstArr,j) = ARR_RMAX(srcArr,i); \
	ARR_CMAX(dstArr,j) = ARR_CMAX(srcArr,i); \
	ARR_SCORE(dstArr,j) = ARR_SCORE(srcArr,i); \
	ARR_CLASS(dstArr,j) = ARR_CLASS(srcArr,i); \
	ARR_SCALE(dstArr,j) = ARR_SCALE(srcArr,i); \
	ARR_EXID(dstArr,j) = ARR_EXID(srcArr,i); \
}
	
#define ARR_COPY_P(srcArr,dstArr) { \
	ARR_RMIN_P(dstArr) = ARR_RMIN_P(srcArr); \
	ARR_CMIN_P(dstArr) = ARR_CMIN_P(srcArr); \
	ARR_RMAX_P(dstArr) = ARR_RMAX_P(srcArr); \
	ARR_CMAX_P(dstArr) = ARR_CMAX_P(srcArr); \
	ARR_SCORE_P(dstArr) = ARR_SCORE_P(srcArr); \
	ARR_CLASS_P(dstArr) = ARR_CLASS_P(srcArr); \
	ARR_SCALE_P(dstArr) = ARR_SCALE_P(srcArr); \
	ARR_EXID_P(dstArr) = ARR_EXID_P(srcArr); \
}	

//some structs which ISPC complains if it sees
//also added esvmBoxes here since the macros were defined in this file
struct esvmBoxes
{
	float *arr;
	int num;	
};

struct esvmParameters
{

	/*User Set Parameters*/

	//related to computing Hog Pyramid
	int cellWidth;
	int maxHogLevels;
	int minHogDim;
	int levelsPerOctave;
	float minImageScale;
	int hogPadding;
	bool hogEnablePadding;
	bool useMexResize;
	bool flipImage;
	bool saveHogPyr;

	//related to convolution
	bool convEnablePadding;

	//related to bounding box computation
	float detectionThreshold;
	float nmsOverlapThreshold;
	int maxWindowsPerExemplar;
	
	//this is used for pre-allocation of memory
	int maxTotalBoxesPerExemplar;


	//related to performance
	int userTasks;
};

#ifdef ESVM_PERFORMANCE_COUNTERS
struct esvmPerformance
{
	double hogTime;
	double convTime;
	double nmsTime;
};
#endif	

struct esvmOutput
{
	/*Callee Set Parameters*/
	esvmHogPyr *hogpyr;
	esvmBoxes *boxes;
	esvmBoxes *flipBoxes;

	//for performance measurements
#ifdef ESVM_PERFORMANCE_COUNTERS
	esvmPerformance perf;
#endif	
};


//NOTE:: DO NOT use malloc to allocate esvmModel
//malloc will not call the constructor for std::map
//and bad things will happen.
//So use "new"
struct esvmModel
{
	esvmHogPyr *hogpyr;
	float *b;
	int *ids;
	std::map<int,std::string> idMap;
	int num;
};

//memory allocation wrappers
void *esvmMalloc(const size_t size);
void *esvmCalloc(const size_t nmemb, const size_t size);

//class to make reading/writing to IplImage easier
//from http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.html
template<class T> class Image
{
private:
	IplImage* imgp;
public:
	Image(IplImage* img=0) {imgp=img;}
	~Image(){imgp=0;}
	void operator=(IplImage* img) {imgp=img;}
	inline T* operator[](const int rowIndx) {
		return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));}
};

typedef struct{
	  unsigned char b,g,r;
} RgbPixel;

typedef struct{
	  float b,g,r;
} RgbPixelFloat;

typedef Image<RgbPixel>       RgbImage;
typedef Image<RgbPixelFloat>  RgbImageFloat;
typedef Image<unsigned char>  BwImage;
typedef Image<float>          BwImageFloat;



/*HOG io stuff*/
void printHog(const esvmHog *s,const int precision=4);

/*
 * Still not sure if they are uber useful. Wait for some more feedback!
 * 
//newer io functions. easier to use.
esvmHog *readHogTxt(const char *filename);
void writeHogTxt(const esvmHog *s,const char *filename);
esvmModel *readModelTxt(const char *filename);*/

//old io functions
void writeHogTxt(const esvmHog *s,FILE *fp, const int precision=4);
esvmHog *readHogTxt(const char *filename,const int rows,const int cols,const int bins);

/*Exemplar io*/
esvmModel *loadExemplars(const char *descFile, int numExemplars);

/*image stuff*/
esvmImage *readImage(const char *filename);
int * RgbtoIm(IplImage *img,const int numRows,const int numCols,const int numChannels);
float * RgbtoImFl(IplImage *img,const int numRows,const int numCols,	const int numChannels);
float * RgbtoImFlTranspose(IplImage *img,const int numRows,const int numCols,
		const int numChannels);
IplImage *ImtoRgb(int *im,const int numRows, const int numCols,const int numChannels);
IplImage *ImtoRgbFl(float *im,const int numRows, const int numCols, const int numChannels);
IplImage *ImTransposetoRgb(int *im,const int numRows, const int numCols, 
		const int numChannels);

/*memory stuff*/
void freeHogPyramid(esvmHogPyr *hogpyr);
void freeHog(esvmHog *hog);


#endif
