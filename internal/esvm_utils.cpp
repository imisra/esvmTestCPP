/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 *  
 */
#include "esvm_utils.h"



void *esvmMalloc(const size_t size)
{
	assert(size>0);
	void *ptr = NULL;
#ifdef ESVM_CONFIG_ALIGN_MEMORY
	int tmp = posix_memalign((void **)&ptr,ESVM_CONFIG_ALIGN_MEMORY_BOUNDARY, size);
	//just to get rid of the annoying warnings of -Wunused-result.
	//It is much bigger than -Wunused-variable :P
#else
	ptr = (void *)malloc(size);
#endif
	return ptr;
}

void *esvmCalloc(const size_t nmemb, const size_t size)
{
	assert(size>0);
	assert(nmemb>0);
	void *ptr = NULL;
#ifdef ESVM_CONFIG_ALIGN_MEMORY
	int tmp = posix_memalign((void **)&ptr,ESVM_CONFIG_ALIGN_MEMORY_BOUNDARY, nmemb*size);
	//just to get rid of the annoying warnings of -Wunused-result
	//It is much bigger than -Wunused-variable :P
	assert(ptr);
	memset((void *)ptr,0,nmemb*size);
#else
	ptr = (void *)calloc(nmemb,size);
#endif	
	return ptr;

}

esvmImage *readImage(const char *filename)
{


	IplImage *img = cvLoadImage(filename,CV_LOAD_IMAGE_COLOR); //force a color image
	RgbImage imarr(img);
	int numRows = img->height;
	int numCols = img->width;
	int numChannels = 3;

	int *im = (int *)esvmMalloc(sizeof(int)*numRows*numCols*numChannels);
	assert(im!=NULL);

	int dim1 = numCols*numRows;

	for(int i=0;i<numRows;i++) {
		for(int j=0;j<numCols;j++) {

			//So the image is arranged such that im[0][1][2] would access
			//pixel at row1, col2 of channel0
			im[0 + i*numCols+j] = imarr[i][j].r;
			im[dim1 + i*numCols+j] = imarr[i][j].g;
			im[2*dim1 + i*numCols+j] = imarr[i][j].b;
		}
	}

	esvmImage *mimg = (esvmImage *)esvmMalloc(sizeof(esvmImage));
	mimg->im = im;
	mimg->numRows = numRows;
	mimg->numCols = numCols;
	mimg->numChannels = numChannels;

	return mimg;
}


int * RgbtoIm(IplImage *img,const int numRows,const int numCols,
		const int numChannels)
{
	assert(numChannels==3);
	RgbImage imarr(img);
	int *im = (int *)esvmMalloc(sizeof(int)*numRows*numCols*numChannels);	
	assert(im!=NULL);

	int dim1 = numCols*numRows;

	for(int i=0;i<numRows;i++) {
		for(int j=0;j<numCols;j++) {

			//So the image is arranged such that im[0][1][2] would access
			//pixel at row1, col2 of channel0
			im[0 + i*numCols+j] = imarr[i][j].r;
			im[dim1 + i*numCols+j] = imarr[i][j].g;
			im[2*dim1 + i*numCols+j] = imarr[i][j].b;			
		}
	}
	
	return im;

}

float * RgbtoImFl(IplImage *img,const int numRows,const int numCols,
		const int numChannels)
{
	assert(numChannels==3);
	RgbImage imarr(img);
	float *im = (float *)esvmMalloc(sizeof(float)*numRows*numCols*numChannels);	
	assert(im!=NULL);

	int dim1 = numCols*numRows;

	for(int i=0;i<numRows;i++) {
		for(int j=0;j<numCols;j++) {

			//So the image is arranged such that im[0][1][2] would access
			//pixel at row1, col2 of channel0
			im[0 + i*numCols+j] = imarr[i][j].r;
			im[dim1 + i*numCols+j] = imarr[i][j].g;
			im[2*dim1 + i*numCols+j] = imarr[i][j].b;			
		}
	}
	
	return im;

}

float * RgbtoImFlTranspose(IplImage *img,const int numRows,const int numCols,
		const int numChannels)
{
	assert(numChannels==3);
	RgbImage imarr(img);
	float *im = (float *)esvmMalloc(sizeof(float)*numRows*numCols*numChannels);	
	assert(im!=NULL);

	int dim1 = numCols*numRows;

	for(int i=0;i<numRows;i++) {
		for(int j=0;j<numCols;j++) {

			//So the image is arranged such that im[0][1][2] would access
			//pixel at row1, col2 of channel0
			im[0 + j*numRows+i] = imarr[i][j].r;
			im[dim1 + j*numRows+i] = imarr[i][j].g;
			im[2*dim1 + j*numRows+i] = imarr[i][j].b;			
		}
	}
	
	return im;

}

IplImage *ImtoRgb(int *im,const int numRows, const int numCols, 
		const int numChannels)
{
	assert(numChannels==3);
	IplImage *img = cvCreateImage(cvSize(numCols,numRows),8,numChannels);
	RgbImage imarr(img);
	
	int dim1 = numCols*numRows;

	for(int i=0;i<numRows;i++) {
		for(int j=0;j<numCols;j++) {

			//So the image is arranged such that im[0][1][2] would access
			//pixel at row1, col2 of channel0
			imarr[i][j].r = im[0 + i*numCols+j];
			imarr[i][j].g = im[dim1 + i*numCols+j];
			imarr[i][j].b = im[2*dim1 + i*numCols+j];			
		}
	}

	return img;
}

IplImage *ImtoRgbFl(float *im,const int numRows, const int numCols, 
		const int numChannels)
{
	assert(numChannels==3);
	IplImage *img = cvCreateImage(cvSize(numCols,numRows),8,numChannels);
	RgbImage imarr(img);
	
	int dim1 = numCols*numRows;

	for(int i=0;i<numRows;i++) {
		for(int j=0;j<numCols;j++) {

			//So the image is arranged such that im[0][1][2] would access
			//pixel at row1, col2 of channel0
			imarr[i][j].r = im[0 + i*numCols+j];
			imarr[i][j].g = im[dim1 + i*numCols+j];
			imarr[i][j].b = im[2*dim1 + i*numCols+j];			
		}
	}

	return img;
}

IplImage *ImTransposetoRgb(int *im,const int numRows, const int numCols, 
		const int numChannels)
{
	assert(numChannels==3);
	IplImage *img = cvCreateImage(cvSize(numCols,numRows),8,numChannels);
	RgbImage imarr(img);
	
	int dim1 = numCols*numRows;
	printf("T: nr %d; nc %d; dim1 %d\n",numRows,numCols,dim1);

	for(int i=0;i<numRows;i++) {
		for(int j=0;j<numCols;j++) {

			//So the image is arranged such that im[0][1][2] would access
			//pixel at row1, col2 of channel0
			imarr[i][j].r = im[0 + j*numRows+i];
			imarr[i][j].g = im[dim1 + j*numRows+i];
			imarr[i][j].b = im[2*dim1 + j*numRows+i];			
		}
	}

	return img;
}

void printHog(const esvmHog *s,const int precision)
{
	if(s==NULL || s->feature==NULL)
	{
		printf("Error printing esvmHog");
		return;
	}

	for(int i=0;i<s->rows;i++) {
		for(int k=0;k<s->bins;k++){
			for(int j=0;j<s->cols;j++) {
					printf("%0.*f",precision,s->feature[k*s->rows*s->cols+i*s->cols+j]);
				if(j+k<s->cols+s->bins-2) {
					printf(" ");

				}
			}
		}
		printf("\n");
	}

	return;

}

void writeHogTxt(const esvmHog *s,FILE *fp, const int precision)
{
	//	NOTE: This function is marked for change in TODO.	
	if(s==NULL || s->feature==NULL)
		{
			printf("Error printing esvmHog");
			return;
		}

		for(int i=0;i<s->rows;i++) {
			for(int k=0;k<s->bins;k++){
				for(int j=0;j<s->cols;j++) {
						fprintf(fp,"%0.*f",precision,s->feature[k*s->rows*s->cols+i*s->cols+j]);
					if(j+k<s->cols+s->bins-2) {
						fprintf(fp," ");

					}
				}
			}
			fprintf(fp,"\n");
		}

		return;
	
}



esvmHog *readHogTxt(const char *filename,const int rows,const int cols,const int bins)
{

	FILE *f = fopen(filename,"r");

	if(__unlikely(f==NULL)) {
		fprintf(stderr,"readHogFromFile:: %s file doesn't exist\n",filename);
		return NULL;
	}

	esvmHog *hog = (esvmHog *)esvmMalloc(sizeof(esvmHog));
	if(__unlikely(hog==NULL)) {
		fprintf(stderr,"readHogFromFile:: Not enough memory reading %s\n",filename);
		return NULL;
	}
	hog->feature = (float *)esvmMalloc(rows*cols*bins*sizeof(float));
//	posix_memalign((void **)&hog->feature,16, rows*cols*bins*sizeof(float));
	if(__unlikely(hog->feature==NULL)) {
		fprintf(stderr,"readHogFromFile:: Not enough memory for hog. Reading %s\n",filename);
		return NULL;
	}
	hog->rows = rows;
	hog->cols = cols;
	hog->bins = bins;

	for(int i=0;i<hog->rows;i++) {
		for(int k=0;k<hog->bins;k++){
			for(int j=0;j<hog->cols;j++) {
			int ret = fscanf(f,"%f",&hog->feature[k*rows*cols+i*cols+j]);
					if(__unlikely(ret==EOF||ret<=0))
					{
						fclose(f);
						free(hog->feature);
						free(hog);
						fprintf(stderr,"readHogFromFile:: Error reading file %s for hog-size (%d,%d,%d). Did you pass the correct arguments ?\n",filename,rows,cols,bins);
						return NULL;

					}

			}
		}
	}
	fclose(f);
	return hog;

}

esvmModel *loadExemplars(const char *descFile,int numExemplars)
{
	assert(numExemplars > 0);
	FILE *f = fopen(descFile,"r");

	if(__unlikely(f==NULL)) {
		fprintf(stderr,"loadExemplars:: %s file doesn't exist\n",descFile);
		return NULL;
	}

	char wfile[ESVM_MAX_FILENAME_CHARS];
	char wclass[ESVM_MAX_FILENAME_CHARS];
	int rows;
	int cols;
	float bval;
	esvmHog **hogs = (esvmHog **)esvmMalloc(numExemplars*sizeof(esvmHog *));
	float *bs = (float *)esvmMalloc(numExemplars*sizeof(float));
	if(__unlikely(hogs==NULL || bs==NULL)) {
		fprintf(stderr,"loadExemplars:: Not enough memory. Try reducing the number of exemplars to be loaded.\n");
		return NULL;

	}
	int count = 0;
	int id = 0;
	bool finishLoad = false;

	//can't use malloc here, since it will not call the std::map
	//constructor. Finally I did end up using new :(
	esvmModel *model = new esvmModel;
	
	while(fscanf(f,"%s %s",wclass,wfile)!=EOF) {

	//	printf("Processing %s\n",wfile);
		FILE *fexem = fopen(wfile,"r");
		if(__unlikely(fexem==NULL)) {
			fprintf(stderr,"loadExemplars:: %s file doesn't exist[2]\n",wfile);
			fclose(f);
			free(bs);
			free(hogs);
			free(model);
			return NULL;
		}

		model->idMap.insert( std::pair<int,std::string>(id,wclass));


		while(fscanf(fexem,"%s %d %d %f",wfile,&rows,&cols,&bval)!=EOF) {
	
			bs[count] = bval;
			hogs[count] = readHogTxt(wfile,rows,cols,31);
			hogs[count]->classId = id;
			count++;
			if(count ==numExemplars) {
				finishLoad = true;
				break;
			}
		}
		if(finishLoad==true) {
			break;
		}
		id++;
	}
	
	bs = (float *)realloc((void *)bs,count*sizeof(float));
	hogs = (esvmHog **)realloc((void *)hogs,count*sizeof(esvmHog *));

	esvmHogPyr *hogpyr = (esvmHogPyr *)esvmMalloc(sizeof(esvmHogPyr));
	hogpyr->hogs = hogs;
	hogpyr->num = count;

	model->hogpyr = hogpyr;
	model->b = bs;
	model->num = count;
	
	fclose(f);

	return model;

}

void freeHog(esvmHog *hog) 
{
	if(hog==NULL)
		return;
	
	free(hog->feature);
}
void freeHogPyramid(esvmHogPyr *hogpyr)
{
	for(int i=0;i<hogpyr->num;i++) {
		freeHog(hogpyr->hogs[i]);
	}
	free(hogpyr->scale);
}
