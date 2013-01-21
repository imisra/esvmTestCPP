/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 * 
 */

#include "esvm.h"
#include "esvm_utils.h"

#include <getopt.h>
#include <execinfo.h>
#include <signal.h>
#include <errno.h>
using namespace std;


void handler(int sig) {
	void *array[100];
	size_t size;

	size = backtrace(array,100);

	fprintf(stderr,"Signal %d caught\n",sig);
	fprintf(stderr,"Signal Name: %s\n",strsignal(sig));
	fprintf(stderr,"Error Name: %s\n",strerror(errno));
	backtrace_symbols_fd(array,size,2);
	exit(1);
}


int main(int argc, char *argv[])
{
	
	if(argc < 5) {
		fprintf(stderr,"Incorrect usage. Specify image, exemplar file and numExemplars\n");
		return 1;
	}

	int opt;

	static struct option long_options[] = {
		{"image", 1, 0, 'i'},
		{"tasks",1,0,'u'},
		{"draw", 1, 0, 'd'},
		{"levelsperoct", 1, 0, 'l'},
		{"numexemplars",1,0,'n'},
	//	{"exemdesc",1,0,'f'},
	};

	//char *imageName = "peppers.png";
	char imageName[ESVM_MAX_FILENAME_CHARS];
	bool shouldDraw = false;
	IplImage *img = NULL;
	int userTasks = 4;
	bool userTasks_g = false;
	int numExemplars = 20;
	int levelsPerOctave = 10;
	bool levelsPer_g = false;
	const char *descFile = "exemplar-txt-files-list";
	//char *descFile;

	while ((opt = getopt_long(argc, argv, "i:u:l:n:d", long_options, NULL)) != EOF) {

		switch (opt) {
			case 'i':
				{
					//imageName = optarg;
					strcpy(imageName,optarg);
					break;
				}
			case 'u':
				{
					userTasks = atoi(optarg);		
					userTasks_g = true;
					break;
				}
			case 'l':
				{
					levelsPerOctave = atoi(optarg);		
					levelsPer_g = true;
					break;
				}

			case 'n':
				{
					//printf("num %s\n",optarg);
					numExemplars = atoi(optarg);
					break;
				}

			case 'd':
				{
					shouldDraw = true;		
					break;
				}

/*				
			case 'f':
				{
					descFile = optarg;
					break;
				}*/

			default:
				fprintf(stderr,"Incorrect usage\n");
				return 1;
		}
	}
	if(shouldDraw==true) {
		img = cvLoadImage(imageName,CV_LOAD_IMAGE_COLOR);
	}

	//install handler for debugging
	signal(SIGSEGV,handler);
	signal(SIGABRT,handler);

	//load the models
	esvmModel *model = loadExemplars(descFile,numExemplars);
	if(numExemplars != model->hogpyr->num) {
		fprintf(stderr,"could not load %d exemplars. Will work with %d exemplars\n",
				numExemplars,model->hogpyr->num);
		numExemplars = model->hogpyr->num;				
	}
	
	//get default parameters for classification
	esvmParameters *params = esvmDefaultParameters();

	if(levelsPer_g==true) {
		params->levelsPerOctave = levelsPerOctave;
	}
	if(userTasks_g==true) {
		params->userTasks = userTasks;
	}

	double start = CycleTimer::currentSeconds();
	esvmOutput *output = esvmSIMEWrapper(params,imageName,model);
	double end = CycleTimer::currentSeconds();
//
//	int totalBoxes = 0;
//
//	for(int i=0;i<numExemplars;i++) {
//		totalBoxes += output->boxes[i].num;
//	}

#ifdef ESVM_PERFORMANCE_COUNTERS	
	printf("Image %s; Hog levels %d; Weights %d; Tasks %d; Boxes %d;"
			" Levels per octave %d; Total Time %0.4lf; Hog Time %0.4lf; "
			"Conv Time %0.4lf; NMS_etc Time %0.4lf\n",
			imageName,output->hogpyr->num,model->num,
			params->userTasks,output->boxes->num,levelsPerOctave,
			(end-start)*1000,output->perf.hogTime,output->perf.convTime,output->perf.nmsTime);
#else	
	printf("Hog levels %d; Weights %d; Boxes %d; Total Time %0.4lf\n",output->hogpyr->num,model->num,totalBoxes,(end-start)*1000);
#endif
	fprintf(stderr,"success:: Image %s; Hog levels %d; Weights %d; Tasks %d; Boxes %d \n",imageName,output->hogpyr->num,model->num,params->userTasks,output->boxes->num);
//	printf("Hog levels %d; Weights %d; Time %0.4lf\n",33,params->numWeights,(end-start)*1000);
//	printf("Hog levels %d; Weights %d; Time %0.4lf\n",params->hogpyr->num,params->numWeights,(seconds)*1000);
//	printf("Returned!\n");


	printf("Total Boxes %d\n",output->boxes->num);

	for(int j=0;j<output->boxes->num;j++) {
		printf("bbox (%0.2f %0.2f %0.2f %0.2f); score %0.3f; scale %0.3f; exid %d; class %s\n",
				BOX_RMIN((output->boxes),j), BOX_CMIN((output->boxes),j),
				BOX_RMAX((output->boxes),j), BOX_CMAX((output->boxes),j),
				BOX_SCORE((output->boxes),j), BOX_SCALE((output->boxes),j),
				BOX_EXID((output->boxes),j),
				model->idMap[BOX_CLASS((output->boxes),j)].c_str());

		if(shouldDraw==true) {
			cvRectangle(img,
					cvPoint(BOX_CMIN((output->boxes),j),BOX_RMIN((output->boxes),j)),
					cvPoint(BOX_CMAX((output->boxes),j),BOX_RMAX((output->boxes),j)),
					cvScalar(0,0,255),
					3,8,0
			);
		}
	}


	if(shouldDraw==true) {
		cvNamedWindow("hog-pipeline",CV_WINDOW_AUTOSIZE);
		cvShowImage("hog-pipeline",img);
		cvWaitKey(0);
		cvDestroyWindow("hog-pipeline");
		cvSaveImage("apple-results.png",img);
		cvReleaseImage(&img);
	}

	return 0;
}
