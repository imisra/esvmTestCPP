/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 *  
 */
#include "esvm.h"

//convolution functions

inline void computeOffsets(int *offset, const int wrows, const int wcols, const int frows, const int fcols)
{
	for (int ix=0;ix<wrows;ix++) {
		for(int iy=0;iy<wcols;iy++) {
		    offset[ix*wcols+iy] = (ix-wrows/2)*fcols+iy-wcols/2;
		}
	}
}


void conv2DValid(float im[], const int irows, const int icols, float kern[], const int krows, 
		const int kcols, const int apronRows, const int apronCols, int offset[], float output[],
		const int orows, const int ocols, const int taskCount, const int taskIndex)
{
	//perform only valid convolution.
	//so if you do need the "full" convolution, send the padded image.
	const int granularity = (irows-krows+1)/taskCount;
	const int startX = apronRows + granularity*(int)taskIndex;
	const int endX = (taskIndex==taskCount-1) ? irows-apronRows+(1^(krows&1)): startX+granularity;
	const int ksize = krows*kcols;
	const int colEnd = icols - apronCols+(1^(kcols&1));
	const int colDiff = icols - colEnd;

	const int dim1 = startX*icols;
	float *tim = im+dim1;
	float *tout = output+dim1;

	for(int i=startX;i<endX;i++) {
		tim+=apronCols;
		tout+=apronCols;
		for(int j=apronCols;j<colEnd;j++) {

#ifdef ESVM_CONFIG_ALIGN_MEMORY
			int *offs __attribute__((aligned (ESVM_CONFIG_ALIGN_MEMORY_BOUNDARY))) = offset;
			float *krn __attribute__((aligned (ESVM_CONFIG_ALIGN_MEMORY_BOUNDARY))) = kern;
#else

			int *offs = offset;
			float *krn = kern;

#endif			
			float sum=0;
			int k;
			#pragma omp parallel for \
			default(shared) private(k) \
			schedule(static) \
			reduction(+:sum)
			for(k=0;k<ksize;k++) {
				sum += tim[*(offs+k)]*(*(krn+k));
			}

			assert(tout-output>=0);
			(*tout++)=sum;
			tim++;
		}
		tim+=colDiff;
		tout+=colDiff;
	}

}

esvmArr2_f *convolve2D(float im[], const int irows, const int icols, float kern[], const int krows, const int kcols)
{
	esvmArr2_f *output = (esvmArr2_f *)esvmMalloc(sizeof(esvmArr2_f));
	output->arr = (float *)esvmCalloc(irows*icols,sizeof(float));
	
	int *offset = (int *)esvmMalloc(krows*kcols*sizeof(int));

	computeOffsets(offset,krows,kcols,irows,icols);
	
	ispc::convolve2D(im, irows, icols, kern, krows, kcols,(ispc::esvmArr2_f *)output,offset);
	
	return output;	
}

void mergeConv(float inp[], const int rows, const int cols, const int nums)
{
	const int dim1 = rows*cols;
	for(int i=0;i<rows;i++) {
		//FIXME:: This has been changed and not tested.
		//Should most likely not be a bug.
		float *tinp = inp+i*cols;//+j;
		for(int j=0;j<cols;j++) {

			float sum=0;
			int k;
			#pragma omp parallel for \
			default(shared) private(k) \
			schedule(static) \
			reduction(+:sum)
			for(k=0;k< nums;k++) {
				sum+=(*(tinp+k*dim1));
			}
			(*tinp++) = (sum);
		}
	}
}

esvmArr2_f *ompConvolve3D(const esvmHog *feat, const esvmHog *whog, const bool enablePadding, const int userTasks)
{
	if(__unlikely(feat==NULL||whog==NULL)) {
		fprintf(stderr,"convolve3D:: got NULL features or NULL weights\n");
		return NULL;
	}

	const int frows = feat->rows;
	const int fcols = feat->cols;
	const int fbins = feat->bins;

	const int wrows = whog->rows;
	const int wcols = whog->cols;
	const int wbins = whog->bins;

	
	if(__unlikely(wbins!=fbins || frows<wrows || fcols<wcols)) {
		fprintf(stderr,"convolve3D:: dimensions of feature and weights don't match. feat(%d,%d,%d) ; weight (%d,%d,%d)\n",frows,fcols,fbins,wrows,wcols,wbins);
		return NULL;
	}


	const int apronRows = floor(wrows/2);
	const int apronCols = floor(wcols/2);
	float *kern = whog->feature;

	float *output = (float *)esvmCalloc(frows*fcols*fbins,sizeof(float));
	if(__unlikely(output==NULL)) {
		fprintf(stderr,"convolve3D:: Not enough memory for output array. Needed %ld bytes\n",frows*fcols*fbins*sizeof(float));
		return NULL;
	}

	int *offset = (int *)esvmMalloc(wrows*wcols*sizeof(int));
	if(__unlikely(offset==NULL)) {
		fprintf(stderr,"convolve3D:: Not enough memory for offset array. Needed %ld bytes\n",wrows*wcols*sizeof(int));
		return NULL;
	}

	computeOffsets(offset,wrows,wcols,frows,fcols);

	const int dim1 = frows*fcols;

	for(int i=0;i<fbins;i++) {

	conv2DValid(feat->feature+i*dim1, frows, fcols, kern+i*wrows*wcols, wrows, wcols, apronRows, apronCols, offset, output+i*dim1, frows, fcols,1,0); 
	}
	mergeConv(output, frows, fcols, fbins);


	//cleanup
	free(offset);

	if(enablePadding==true) {
		//shrink the output
		output= (float *)realloc((void *)output,frows*fcols*sizeof(float));
		esvmArr2_f *soln = (esvmArr2_f *)esvmMalloc(sizeof(esvmArr2_f));
		soln->arr = output;
		soln->rows = frows;
		soln->cols = fcols;
		return soln;
	}
	else {
		const int rowEnd = frows-apronRows+(1^(wrows&1));
        	const int colEnd = fcols - apronCols+(1^(wcols&1));
		const int urows = rowEnd-apronRows;
		const int ucols = colEnd-apronCols;
		float *uout = (float *)esvmMalloc(urows*ucols*sizeof(float));
		for(int i=0;i<urows;i++) {
			float *tmp = uout+i*ucols;
			float *tmpout = output+(i+apronRows)*fcols+apronCols;
			for(int j=0;j<ucols;j++) {
				(*tmp++) = (*tmpout++);
			}
		}
		free(output);
		esvmArr2_f *soln = (esvmArr2_f *)esvmMalloc(sizeof(esvmArr2_f));
		soln->arr = uout;
		soln->rows = urows;
		soln->cols = ucols;
		return soln;
	}

}

esvmArr2_f *convolvePyramids(const esvmHogPyr *feats, const esvmHogPyr *whogs, const bool enablePadding,const int userTasks)
{
	esvmArr2_i *offsets = (esvmArr2_i *)esvmMalloc(sizeof(esvmArr2_i)*feats->num*whogs->num);
	esvmArr2_f *outputs = (esvmArr2_f *)esvmMalloc(sizeof(esvmArr2_f)*feats->num*whogs->num);

#pragma omp parallel for	
	for(int i=0;i<feats->num;i++) {
		for(int j=0;j<whogs->num;j++) {

			const int frows = feats->hogs[i]->rows;
			const int fcols = feats->hogs[i]->cols;
			const int fbins = feats->hogs[i]->bins;

			const int wrows = whogs->hogs[j]->rows;
			const int wcols = whogs->hogs[j]->cols;


			offsets[i*whogs->num+j].arr = (int *)esvmMalloc(wrows*wcols*sizeof(int));
			offsets[i*whogs->num+j].rows = wrows;
			offsets[i*whogs->num+j].cols = wcols;
			outputs[i*whogs->num+j].arr = (float *)esvmMalloc(frows*fcols*fbins*sizeof(float));
			outputs[i*whogs->num+j].rows = frows;
			outputs[i*whogs->num+j].cols = fcols;

			computeOffsets((offsets[i*whogs->num+j].arr),wrows,wcols,frows,fcols);
		}
	}
	ispc::convolvePyramids((ispc::esvmHogPyr *)feats,(ispc::esvmHogPyr *)whogs,(ispc::esvmArr2_i *)offsets,enablePadding,(ispc::esvmArr2_f *)outputs,userTasks);


	if(__unlikely(enablePadding==true)) {

#pragma omp parallel for	
		for(int i=0;i<feats->num;i++) {
			for(int j=0;j<whogs->num;j++) {

				const int frows = feats->hogs[i]->rows;
				const int fcols = feats->hogs[i]->cols;
				outputs[i*whogs->num+j].arr = (float *)realloc((void *)outputs[i*whogs->num+j].arr,frows*fcols*sizeof(float));
			}
		}
	}
	else {
#pragma omp parallel for	
		for(int i=0;i<feats->num;i++) {
			for(int j=0;j<whogs->num;j++) {

				const int frows = feats->hogs[i]->rows;
				const int fcols = feats->hogs[i]->cols;
				const int wrows = whogs->hogs[j]->rows;
				const int wcols = whogs->hogs[j]->cols;
				const int apronRows = (wrows/2);
				const int apronCols = (wcols/2);
				const int rowEnd = frows-apronRows+(1^(wrows&1));
				const int colEnd = fcols - apronCols+(1^(wcols&1));
				const int urows = rowEnd-apronRows;
				const int ucols = colEnd-apronCols;
				float *uout = (float *)esvmMalloc(urows*ucols*sizeof(float));
				float *output = outputs[i*whogs->num+j].arr;
		
				for(int ii=0;ii<urows;ii++) {
					float *tmp = uout+ii*ucols;
					float *tmpout = output+(ii+apronRows)*fcols+apronCols;
					for(int jj=0;jj<ucols;jj++) {
						(*tmp++) = (*tmpout++);
					}
				}

				free(output);
				outputs[i*whogs->num+j].arr = uout;
				outputs[i*whogs->num+j].rows = urows;
				outputs[i*whogs->num+j].cols = ucols;

			}
		}
	}

	return outputs;

}

//hog functions
esvmHog *computeHog(int im[],const int numRows,const int numCols,
		const int numChannels,const int cellWidth,
		const bool enablePadding,const int padding,const int userTasks)
{

	//Compute the number of cells. Allocate memory for histograms and HoG
	int blocks[2];
	blocks[0] = (int)round((float)numRows/(float)cellWidth);
	blocks[1] = (int)round((float)numCols/(float)cellWidth);

	// memory for HOG features
	int out[3];
	out[0] = max(blocks[0]-2, 0);
	out[1] = max(blocks[1]-2, 0);
	out[2] = 27+4;

	if(__unlikely(!out[0]||!out[1])) {
		fprintf(stderr,"computeHog:: was asked to compute 0-sized hog. Input params numrows %d, numcols %d, cellWidth %d\n",numRows,numCols,cellWidth);
		return NULL;
	}


	float *hist = (float *)esvmCalloc(blocks[0]*blocks[1]*18*userTasks, sizeof(float));
	float *norm = (float *)esvmCalloc(blocks[0]*blocks[1], sizeof(float));
	assert(hist!=NULL);
	assert(norm!=NULL);



	float *hog = (float *)esvmCalloc(out[0]*out[1]*out[2],sizeof(float));
	assert(hog!=NULL);

	
	//compute the range of pixels we will be working on
	int visible[2];
	visible[0] = blocks[0]*cellWidth;
	visible[1] = blocks[1]*cellWidth;

	//memory for gradient and storing the best orientation
	float *gradient = (float *)esvmCalloc(visible[0]*visible[1],sizeof(float));
	int *best_ori = (int *)esvmCalloc(visible[0]*visible[1],sizeof(int));

	assert(gradient!=NULL);
	assert(best_ori!=NULL);

	ispc::getGradient(im, numRows, numCols, numChannels, cellWidth, visible, 
			blocks, hist, gradient, best_ori, userTasks);
	int mTasks = binHist(hist,gradient, best_ori, visible, blocks, cellWidth, userTasks);
	ispc::combineHist(hist,mTasks,blocks);
	computeNorm(norm, hist, blocks);
	ispc::getFeatures(hog, blocks, hist, norm, out, userTasks);

	//let us cleanup
	free(gradient);
	free(best_ori);
	free(hist);
	free(norm);

	//assign the computed features to the struct;
	esvmHog *soln = (esvmHog *)esvmMalloc(sizeof(esvmHog));
	if(__unlikely(enablePadding==false)) {
		soln->rows = out[0];
		soln->cols = out[1];
		soln->feature = hog;
		soln->bins = out[2];
		soln->cellWidth = cellWidth;
	}
	else {
		soln->rows = out[0]+(padding<<1)+2; 
		//additional 2 recovers the lost bin (check esvm_pyramid.m)
		soln->cols = out[1]+(padding<<1)+2; 
		//the padding<<1 corresponds to padding in get_pyramid.m
		soln->bins = out[2];		

		soln->feature = (float *)esvmCalloc(soln->rows*soln->cols*soln->bins,sizeof(float));
		assert(soln->feature);
		const int dim1=soln->rows*soln->cols;
		const int dim2=out[0]*out[1];

		for(int i=0;i<soln->bins;i++) {

			for(int r=padding+1;r<soln->rows-padding-1;r++) {
				for(int c=padding+1;c<soln->cols-padding-1;c++) {
					soln->feature[i*dim1+r*soln->cols+c] = hog[i*dim2+(r-padding-1)*out[1]+(c-padding-1)];	
				}
			}
		}
		free(hog);

   	}
	

	return soln; 

}

void computeNorm(float norm[], float hist[], const int blocks[2])
{
	//Purely Serial function

	for(int o=0;o<9;o++) {
		float *src1 = hist + o*blocks[0]*blocks[1];
		float *src2 = hist + (o+9)*blocks[0]*blocks[1];
		float *dst = norm;
		float *end = norm + blocks[1]*blocks[0];
		while (dst < end) {
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

}

int binHist(float hist[],const float gradient[],const int best_ori[],const int visible[2],const int blocks[2],const int cellWidth,const int userTasks)
{

int taskCount = 0;
if(userTasks < 4) {
	fprintf(stderr,"binHist:: Not handled less than 4 threads in OpenMP\n");
	return -1;
}


#pragma omp parallel for num_threads(userTasks)
	for(int u=0;u<omp_get_num_threads();u++) {
		const int taskIndex = omp_get_thread_num();
		taskCount = omp_get_num_threads();
		const int granularity = (visible[0]-1)/taskCount;
		const int startX = 1 + granularity*(int)taskIndex;
		const int endX = (taskIndex==taskCount-1) ? visible[0]-1: startX+granularity;


		for(int i=startX; i<endX;i++) {
			for(int j=1 ;j<visible[1]-1;j++) {

				float xp = ((float)i+0.5)/(float)cellWidth - 0.5;
				float yp = ((float)j+0.5)/(float)cellWidth - 0.5;
				int ixp = (int)floor(xp);
				int iyp = (int)floor(yp);
				float vx0 = xp-ixp;
				float vy0 = yp-iyp;
				float vx1 = 1.0-vx0;
				float vy1 = 1.0-vy0;
				float v0 = gradient[i*visible[1]+j];
				int best_o = best_ori[i*visible[1]+j];

				v0 = sqrt(v0);

				assert(best_o>=0 &&best_o<18);


				if (ixp >= 0 && iyp >= 0) {

					*(hist + taskIndex*18*blocks[1]*blocks[0] +best_o*blocks[0]*blocks[1] + ixp*blocks[1] + iyp) += 
						vx1*vy1*v0;
				}


				if (ixp+1 < blocks[0] && iyp >= 0) {
					*(hist + taskIndex*18*blocks[1]*blocks[0] +best_o*blocks[0]*blocks[1] + (ixp+1)*blocks[1] + iyp) +=
						vx0*vy1*v0;
				}

				if (ixp >= 0 && iyp+1 < blocks[1]) {
					*(hist + taskIndex*18*blocks[1]*blocks[0] +best_o*blocks[0]*blocks[1] + ixp*blocks[1] + (iyp+1)) += 
						vx1*vy0*v0;
				}

				if (ixp+1 < blocks[0] && iyp+1 < blocks[1]) {
					*(hist + taskIndex*18*blocks[1]*blocks[0] +best_o*blocks[0]*blocks[1] + (ixp+1)*blocks[1] + (iyp+1)) +=
						vx0*vy0*v0;
				}


			}


		}


	}

	return taskCount;

}

esvmHogPyr *computeHogScaleWrapper(const char *filename,const int cellWidth,
		const int maxLevels,const int minDimension,const int interval,
		const float minScale,const bool enablePadding,const int padding,const int userTasks,
		const bool useMexResize)
{
	IplImage *img = cvLoadImage(filename,CV_LOAD_IMAGE_COLOR); //force a color image
	return computeHogScale(img,cellWidth,
			maxLevels,minDimension,interval,
			minScale,enablePadding,padding,userTasks,
			useMexResize);
}

esvmHogPyr *computeHogScale(IplImage *img,const int cellWidth,
		const int maxLevels,const int minDimension,const int interval,
		const float minScale,const bool enablePadding,const int padding,const int userTasks,
		const bool useMexResize)
{
	int numRows = img->height;
	int numCols = img->width;
	int numChannels = 3;
	float sc = pow(2,1.0/interval);
	esvmHog **pyr = (esvmHog **)esvmCalloc(maxLevels,sizeof(esvmHog *));
	float *scaleArr = (float *)esvmCalloc(maxLevels,sizeof(float));

	//do the first level outside the loop. So that you don't have to make a call to resize
	//OpenCV resize does some weird stuff on resize factor 1.0
		int *tmpim = RgbtoIm(img,numRows,numCols,numChannels);
		scaleArr[0]=1.0f;
		pyr[0] = computeHog(tmpim, numRows, numCols, numChannels, cellWidth,
				enablePadding, padding, userTasks);
		free(tmpim);
	
	int counter = 1;
	float *flIm;
	IplImage *dst;
	if(useMexResize==true) {
		flIm = RgbtoImFlTranspose(img,numRows,numCols,numChannels);
	}
	for(int i=2;i<=maxLevels;i++) {		
		float scale = 1.0 / pow(sc,i-1);
		scaleArr[i-1] = scale;
		int nr = round((float)numRows * scale);
		int nc = round((float)numCols * scale);		
		if(scale < minScale) {
						break;
		}
		if(min(nr,nc)<=minDimension) {
			break;
		}

		int *im;
		if(useMexResize==false) {
			dst = cvCreateImage(cvSize((int)nc,(int)nr),img->depth,numChannels);		
			cvResize(img,dst,ESVM_INTERP);

			im = RgbtoIm(dst,(int)nr,(int)nc,numChannels);
		}
		else {			
			im = mexResize(flIm,numRows,numCols,numChannels,numRows*scale,numCols*scale); 
		}
		
		pyr[i-1] = computeHog(im, nr, nc, numChannels, cellWidth,
				enablePadding, padding, userTasks);
		counter++;
		
		if(useMexResize==false)		
			cvReleaseImage(&dst);

		free(im);
		
		if(__likely(enablePadding==true)) {
			if( max(pyr[i-1]->rows-(padding<<1),pyr[i-1]->cols-(padding<<1)) <= minDimension) {

				break;
				
			}
		}
		else {
			if(max(pyr[i-1]->rows,pyr[i-1]->cols) <= minDimension) {

				break;
			}
		}
	}

	scaleArr = (float *)realloc((void *)scaleArr,counter*sizeof(float));
	esvmHogPyr *hogpyr = (esvmHogPyr *)esvmMalloc(sizeof(esvmHogPyr));
	hogpyr->num = counter;
	hogpyr->hogs = pyr;
	hogpyr->scale = scaleArr;


	//FIXME:: We can shrink pyr from maxLevels to counter, using realloc.
	//It is not a serious problem since pyr actually contains maxLevels*sizeof(esvmHog *) bytes

	cvReleaseImage(&img);
	if(useMexResize==true)
		free(flIm);

	return hogpyr;

}

esvmHog *computeHogWrapper(const char *filename,const int cellWidth,const bool enablePadding,const int padding,const int userTasks)
{
	esvmImage *img = readImage(filename);
	int numRows = img->numRows;
	int numCols = img->numCols;
	int numChannels = img->numChannels;
	assert(numChannels == 3);
	int *im = img->im;

	esvmHog *soln;
	soln = computeHog(im,numRows,numCols,numChannels,cellWidth,enablePadding,padding,userTasks);


	free(img->im); //frees int *im as well.
	free(img);

	return soln;

}


//matrix manipulation stuff

void subtractScalar(float arr[],const int rows,const int cols,const float scalar)
{
	//parallel loop
	for(int i=0;i<rows;i++)
	{
		float *tmp = arr+i*cols;
		//serial loop
		for(int j=0;j<cols;j++)
		{
			*(tmp) = *tmp - scalar;
			tmp++;
		}
	}
}

int compareIndexedArr2D(const void *a, const void*b)
{
	esvmIndexedArr2D *A = (esvmIndexedArr2D *)a;
	esvmIndexedArr2D *B = (esvmIndexedArr2D *)b;

	return ESVM_COMPARE(A->val,B->val);
}

int invCompareIndexedArr2D(const void *a, const void*b)
{
	esvmIndexedArr2D *A = (esvmIndexedArr2D *)a;
	esvmIndexedArr2D *B = (esvmIndexedArr2D *)b;
	return ESVM_INVCOMPARE(A->val,B->val);

}

int *sort2DIndex(float arr[], const int rows,const int cols,const bool descendSort,const bool threshold, const float scoreThreshold, int *nkeep,const int padding)
{
	esvmIndexedArr2D *in = (esvmIndexedArr2D *)esvmMalloc(rows*cols*sizeof(esvmIndexedArr2D));
	//parallel loop
	for(int i=0;i<rows;i++) {
		const int rowIndex = i*cols;
		//parallel loop
		for(int j=0;j<cols;j++) {
			in[rowIndex+j].val = arr[rowIndex+j];
			in[rowIndex+j].row = i;
			in[rowIndex+j].col = j;
		}
	}
	if(descendSort==ESVM_DESCEND_SORT) {
		qsort((void *)in,rows*cols,sizeof(esvmIndexedArr2D),invCompareIndexedArr2D);
	}
	else {
		qsort((void *)in,rows*cols,sizeof(esvmIndexedArr2D),compareIndexedArr2D);
	}

	int *index = (int *)esvmMalloc(rows*cols*2*sizeof(int));
	const int dim1 = rows*cols;
	if(threshold==ESVM_THRESHOLD) {
		*nkeep=0;
		bool flag=false;

		//serial loop
		for(int i=0;i<rows;i++) {
			const int rowIndex = i*cols;
			//serial loop
			for(int j=0;j<cols;j++) {
				const int tmp = rowIndex+j;
				if(in[tmp].val >= scoreThreshold) {
					arr[tmp] = in[tmp].val;
					index[tmp] = in[tmp].row - padding;
					index[dim1+tmp] = in[tmp].col - padding;
					(*nkeep)++;
				}
				else {
					break;
					flag=true;
				}
			}
			if(flag==true) {
				break;
			}
		}
		//note::to realloc or not realloc, that is the question!
		//I decide not to. This makes the sort function more generic.
		//Also, if someone wants to use all the unthresholded values,
		//this function would be so damn sneaky to mess up the length!
	}
	else {
		//parallel loop
		for(int i=0;i<rows;i++) {
			const int rowIndex = i*cols;
			//parallel loop
			for(int j=0;j<cols;j++) {
				const int tmp = rowIndex+j;
				arr[tmp] = in[tmp].val;
				index[tmp] = in[tmp].row;
				index[dim1+tmp] = in[tmp].col;
			}
		}

	}

	free(in);
	return index;
}

void nms(float *boxes, const int numBoxes, const float overlapThreshold, int *numPruned,
		float **prunedArr)
{
	//boxes contain bboxes corresponding to scores.
	//boxes are indexed according to descending order of scores

	if(numBoxes == 0) {
		*numPruned = 0;
		*prunedArr = NULL;		
		return;
	}
	
	int *picks = (int *)esvmMalloc(numBoxes*sizeof(int));
	int count = 0;
	float *area = (float *)esvmMalloc(numBoxes*sizeof(float));
	float *overlap = (float *)esvmMalloc(numBoxes*sizeof(float));

	std::list<int> indices;
	
	//parallel loop
	for(int i=numBoxes-1;i>=0;i--) {
		indices.push_front(i);		
		area[i] = (ARR_RMAX(boxes,i) - ARR_RMIN(boxes,i)+1)*
				(ARR_CMAX(boxes,i) - ARR_CMIN(boxes,i)+1);
	}
	
	while(indices.empty()==false) {
		int index = indices.front();
		assert(count<numBoxes);
		picks[count++]=index;
		
		std::list<int>::iterator it=indices.begin();
		it++; //it now points to the second element of indices
		while(it != indices.end()) {

			float x1 = max(ARR_RMIN(boxes,index),ARR_RMIN(boxes,(*it)));
			float x2 = min(ARR_RMAX(boxes,index),ARR_RMAX(boxes,(*it)));
			float y1 = max(ARR_CMIN(boxes,index),ARR_CMIN(boxes,(*it)));
			float y2 = min(ARR_CMAX(boxes,index),ARR_CMAX(boxes,(*it)));

			
			float overlap_val = max(0.0,x2-x1+1)*max(0.0,y2-y1+1)/area[*it];

			if(overlap_val > overlapThreshold)
			{
				it=indices.erase(it);
			}
			else
			{
				it++;
			}
		}
		indices.pop_front();		
	}
	
	float *pruned = (float *)esvmMalloc(ESVM_BOX_DIM*count*sizeof(float));
	assert(pruned!=NULL);
	
	for(int i=0;i<count;i++) {
		ARR_COPY(boxes,picks[i],pruned,i);	
	}
	
	*numPruned = count;
	*prunedArr = pruned;
	

	//cleanup!!
	free(picks);
	free(area);
	free(overlap);
	//since indices is a stack variable, it is freed automatically when we exit.
	
	return;

}


//Adapted from Tomasz Malisiewicz's psort

void psort(float* in, int m, int K, float* out0, int* out1)
{
	std::priority_queue<std::pair<float,int> > q;
	//printf("in psort\n");

	// insert first K elements into priority queue
	for (int i = 0; i < K; ++i)
	{
		q.push(std::pair<float,int>(in[i],i));
	}
	
	float curtop = q.top().first;
	
	for (int i = K; i < m; ++i)
	{
		float& curval = in[i];
	
		if (curval < curtop)
		{   	
			q.pop();	
			q.push(std::pair<float,int>(curval,i));
			curtop = q.top().first;	
		}   
	}

	for (int i = 0; i < K; ++i)
	{
		out0[(K-1)-i] = q.top().first;
		out1[(K-1)-i] = q.top().second;
		q.pop();
	}

	if (q.size() != 0)
		fprintf(stderr,"psort::non-empty queue");

}


esvmParameters *esvmDefaultParameters()
{
	esvmParameters *params = (esvmParameters *)esvmMalloc(sizeof(esvmParameters));
	params->cellWidth = 8;
	params->maxHogLevels = 200;
	params->minHogDim = 5;
	params->levelsPerOctave = 10;
	params->minImageScale = 0.01;
	params->hogPadding = 5; //padder parameter in matlab
	params->hogEnablePadding = ESVM_PAD;
	params->convEnablePadding = ESVM_NO_PAD;
	params->detectionThreshold = -1.0;
	params->nmsOverlapThreshold = 0.5;
	params->maxWindowsPerExemplar = 10;
	params->maxTotalBoxesPerExemplar = 100;
	params->userTasks = 4;
	params->useMexResize = true;
	params->flipImage = false;
	params->saveHogPyr = true;

	return params;

}

esvmOutput *esvmSIMEWrapper(esvmParameters *params, const char *imageName, esvmModel *model) 
{
	IplImage *img = cvLoadImage(imageName,CV_LOAD_IMAGE_COLOR);
	esvmOutput *op = esvmSIME(params,img,model);
 
	if(params->flipImage == true) {
		fprintf(stderr,"esvmSIMEWrapper:: Image flipping not implemented yet\n");
		exit(1);
	}
	return op;
	
}

esvmOutput *esvmSIME(esvmParameters *params, IplImage *img, esvmModel *model)
{

	//userTasks needs to be greater than 4.
	//This is for performance. less than 4 threads doesn't make sense!
	//Also I think (there was an assumption made in some version
	//, I forget if it is still there or not.)
	//otherwise binning histograms (binHists)
	//will not work properly.
	//this can be fixed, but I haven't done it.
	assert(params->userTasks >= 4);

	//computing hog pyramid and reading whogs can be made parallel
#ifdef ESVM_PERFORMANCE_COUNTERS
	double hogTime = CycleTimer::currentSeconds();
#endif
	
	esvmHogPyr *hogpyr = computeHogScale(img,params->cellWidth,params->maxHogLevels,params->minHogDim,
			params->levelsPerOctave,params->minImageScale,params->hogEnablePadding,
			params->hogPadding,params->userTasks,params->useMexResize);
	
	
#ifdef ESVM_PERFORMANCE_COUNTERS
	hogTime -= CycleTimer::currentSeconds();
#endif	

	const esvmHogPyr *whogpyr = model->hogpyr;
	const esvmHog **whogs = (const esvmHog **) whogpyr->hogs;
	const int numWeights = whogpyr->num;
	const float *bWeight = model->b;

#ifdef ESVM_PERFORMANCE_COUNTERS
	double convTime = CycleTimer::currentSeconds();
#endif	
	esvmArr2_f *convResults = convolvePyramids(hogpyr,whogpyr,params->convEnablePadding,
			params->userTasks);

#ifdef ESVM_PERFORMANCE_COUNTERS
	convTime -= CycleTimer::currentSeconds();
#endif
	
	//allocate memory for bounding boxes per exemplar
	esvmBoxes *boxesArr = (esvmBoxes *)esvmCalloc(numWeights*hogpyr->num,sizeof(esvmBoxes));

	assert(params->maxTotalBoxesPerExemplar > params->maxWindowsPerExemplar);

	for(int w=0;w<numWeights;w++) {
		boxesArr[w].arr = (float *)esvmMalloc(params->maxTotalBoxesPerExemplar*ESVM_BOX_DIM*sizeof(float));
		boxesArr[w].num = 0;
		std::fill(boxesArr[w].arr,boxesArr[w].arr+params->maxTotalBoxesPerExemplar*ESVM_BOX_DIM,
				ESVM_FLOAT_MIN);
	}

	float *maxers = (float *)esvmMalloc(numWeights*sizeof(float));
	std::fill(maxers,maxers+numWeights,ESVM_FLOAT_MIN);
	float *negScores = (float *)esvmMalloc(params->maxTotalBoxesPerExemplar*sizeof(float));
	float *topScores = (float *)esvmMalloc(params->maxTotalBoxesPerExemplar*sizeof(float));
	int *topInds = (int *)esvmMalloc(params->maxTotalBoxesPerExemplar*sizeof(int));
	float *topBoxes = (float *)esvmMalloc(params->maxTotalBoxesPerExemplar*ESVM_BOX_DIM*sizeof(float)); 


	//parallel loop.

	int numBoxes = 0;


#ifdef ESVM_PERFORMANCE_COUNTERS
	double nmsTime = CycleTimer::currentSeconds();
#endif	
	

	//serial loop because maxers are maintained from higher levels!
	for(int i=hogpyr->num-1;i>=0;i--) {

		//serial loop	
		for(int w=0;w<numWeights;w++) {
			
			esvmArr2_f *convOut = &(convResults[i*numWeights+w]);
			subtractScalar(convOut->arr,convOut->rows,convOut->cols,bWeight[w]);
			int nkeep;

			//hogPadding is subtracted from the indices.
			float detectionThreshold = max(maxers[w],params->detectionThreshold);
			//float detectionThreshold = params->detectionThreshold;
			int *indices = sort2DIndex(convOut->arr,convOut->rows,convOut->cols,
					ESVM_DESCEND_SORT,ESVM_THRESHOLD,detectionThreshold,&nkeep,
					params->hogPadding);
			
			if(nkeep==0) {
				continue;
			}
		
			//arrays for top-k sorting
			const int topK = min(boxesArr[w].num+nkeep,params->maxWindowsPerExemplar);		

			//concatenate current boxes to the boxes already detected by exemplar
			float *bboxes = &(boxesArr[w].arr[boxesArr[w].num*ESVM_BOX_DIM]);
			int *tmpIndex = indices;
			int dim1 = convOut->rows*convOut->cols;
			float resizing = params->cellWidth/hogpyr->scale[i];
			//get the bounding boxes in the original image
			//need to rescale
			assert(boxesArr[w].num+nkeep <= params->maxTotalBoxesPerExemplar);
			//printf("NKEEP is %d\n",nkeep);
			for(int j=0;j<nkeep;j++) {
				float *bboxL = bboxes+j*ESVM_BOX_DIM;
				ARR_RMIN_P(bboxL) = ((*(tmpIndex+j))*resizing);
				ARR_CMIN_P(bboxL) = ((*(tmpIndex+j+dim1))*resizing);
				ARR_RMAX_P(bboxL) = ((*(tmpIndex+j)+whogs[w]->rows)*resizing)-1;
				ARR_CMAX_P(bboxL) = ((*(tmpIndex+j+dim1)+whogs[w]->cols)*resizing)-1;
				
				//Put negative of score inside.
				//This is useful for finding top-k elements
				negScores[boxesArr[w].num+j] = -convOut->arr[j];
				ARR_SCORE_P(bboxL) = convOut->arr[j];
				ARR_SCALE_P(bboxL) = hogpyr->scale[i];
				ARR_CLASS_P(bboxL) = (int)whogs[w]->classId;
				ARR_EXID_P(bboxL) = w;
				
			}
			
			//find top-k boxes
			psort(negScores, boxesArr[w].num+nkeep, topK, topScores, topInds);

			int *tmpTop = topInds;
			bboxes = boxesArr[w].arr;
			for(int j=0;j<topK;j++) {
				float *bboxL = topBoxes+j*ESVM_BOX_DIM;
				float *bboxR = bboxes+topInds[j]*ESVM_BOX_DIM;			
				ARR_COPY_P(bboxR,bboxL);
				
				negScores[j] = -ARR_SCORE_P(bboxR);
			}

			//now copy back the current boxes into the list of boxes for this exemplar
			memcpy(boxesArr[w].arr,topBoxes,topK*ESVM_BOX_DIM*sizeof(float));
			boxesArr[w].num = topK;

			if(topK >= params->maxWindowsPerExemplar) {
				//update maxers if topK > threshold
				maxers[w] = -topScores[topK-1];
			}

			free(indices);
			free(convOut->arr);

		}
	}
	
	//more cleanup
	free(convResults);
	free(maxers);
	free(negScores);
	free(topScores);
	free(topInds);
	free(topBoxes);
	
	//perform nms on each exemplar's boxes
	esvmBoxes *nmsBoxesArr = (esvmBoxes *)esvmCalloc(numWeights*hogpyr->num,sizeof(esvmBoxes));
	
	int totalBoxes = 0;
	for(int w=0;w<numWeights;w++) {	

		nms(boxesArr[w].arr,boxesArr[w].num, params->nmsOverlapThreshold,
				&(nmsBoxesArr[w].num),&(nmsBoxesArr[w].arr));
		totalBoxes += (nmsBoxesArr[w].num);
		
		free(boxesArr[w].arr);
	}
	free(boxesArr);

#ifdef ESVM_PERFORMANCE_COUNTERS
	nmsTime -= CycleTimer::currentSeconds();
#endif	
	
	//assign output
	esvmOutput *output = (esvmOutput *)esvmMalloc(sizeof(esvmOutput));
	
	//collect all the boxes together
	output->boxes = (esvmBoxes *)esvmCalloc(1,sizeof(esvmBoxes));
	output->boxes->num = totalBoxes;
	if(totalBoxes>0)
		output->boxes->arr = (float *)esvmMalloc(totalBoxes*ESVM_BOX_DIM*sizeof(float));
	
	int count = 0;
	for(int w=0;w<numWeights;w++) {
		for(int j=0;j<nmsBoxesArr[w].num;j++) {
			ARR_COPY(nmsBoxesArr[w].arr,j,output->boxes->arr,count);
			count++; 
		}
		free(nmsBoxesArr[w].arr);
	}
	free(nmsBoxesArr);
	
	if(params->saveHogPyr==false) {
		freeHogPyramid(hogpyr);
		output->hogpyr = (esvmHogPyr *)esvmMalloc(sizeof(esvmHogPyr));
		output->hogpyr->num = 0;
	}
	else {
		output->hogpyr = hogpyr;
	}
	
	
#ifdef ESVM_PERFORMANCE_COUNTERS
	output->perf.hogTime = -hogTime*1000;
	output->perf.convTime = -convTime*1000;
	output->perf.nmsTime = -nmsTime*1000;
#endif	
	
	return output;

}

esvmOutput **esvmMIMEWrapper(esvmParameters *params, const char **imageNames, int numImages, esvmModel *model)
{

	esvmOutput **outputs = (esvmOutput **)esvmMalloc(numImages*sizeof(esvmOutput *));

	//parallel loop
	for(int i=0;i<numImages;i++) {
		outputs[i] = esvmSIMEWrapper(params,imageNames[i],model);
	}

	return outputs;
}
