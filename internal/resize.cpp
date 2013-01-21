
	
#include <math.h>
#include <assert.h>
#include <string.h>
#include "esvm_utils.h"

/*
 * Ported to general C++ from the mex version by Ishan Misra
 */

/*
 * Fast image subsampling.
 * This is used to construct the feature pyramid.
 */


// struct used for caching interpolation values
struct alphainfo {
  int si, di;
  float alpha;
};

// copy src into dst using pre-computed interpolation values
void alphacopy(float *src, float *dst, struct alphainfo *ofs, int n) {
  struct alphainfo *end = ofs + n;
  while (ofs != end) {
    dst[ofs->di] += ofs->alpha * src[ofs->si];
    ofs++;
  }
}

// resize along each column
// result is transposed, so we can apply it twice for a complete resize
void resize1dtran(float *src, int sheight, float *dst, int dheight, 
		  int width, int chan) {
  float scale = (float)dheight/(float)sheight;
  float invscale = (float)sheight/(float)dheight;
  
  // we cache the interpolation values since they can be 
  // shared among different columns
  int len = (int)ceil(dheight*invscale) + 2*dheight;
  alphainfo ofs[len];
  int k = 0;
  float fsy1,fsy2;
  int sy1,sy2,sy;
  for (int dy = 0; dy < dheight; dy++) {
    fsy1 = dy * invscale;
    fsy2 = fsy1 + invscale;
    sy1 = (int)ceil(fsy1);
    sy2 = (int)floor(fsy2);       

    if (sy1 - fsy1 > 1e-3) {
      ofs[k].di = dy*width;
      ofs[k].si = sy1-1;
      ofs[k++].alpha = (sy1 - fsy1) * scale;
    }

    for (sy = sy1; sy < sy2; sy++) {
      ofs[k].di = dy*width;
      ofs[k].si = sy;
      ofs[k++].alpha = scale;
    }

    if (fsy2 - sy2 > 1e-3) {
      ofs[k].di = dy*width;
      ofs[k].si = sy2;
      ofs[k++].alpha = (fsy2 - sy2) * scale;
    }
  }

  // resize each column of each color channel
  //bzero(dst, chan*width*dheight*sizeof(double));
  memset(dst,0,chan*width*dheight*sizeof(float));
//  
//  for (int c = 0; c < chan; c++) {
//	for(int i=0; i<width;i++) {
//		float *s = src + c*width*sheight + i;
//		float *d = dst + c*width*dheight + i;
//		alphacopy(s,d,ofs,k);
//	}
//  }
    
  for (int c = 0; c < chan; c++) {
    for (int x = 0; x < width; x++) {
      float *s = src + c*width*sheight + x*sheight;
      float *d = dst + c*width*dheight + x;
      alphacopy(s, d, ofs, k);
    }
  }
}

//returns a transposed version of resized image
//matlab leftover!
int *mexResizeTranspose(float *im, const int rows, const int cols, const int channels,
		const int rRows, const int rCols) {
	
	const float scaleFactor = (float)rRows/(float)rows;
	//assert((int)cols*scaleFactor == rCols);
	
	if(scaleFactor > 1 || scaleFactor <=0) {
		fprintf(stderr,"mexResize:: invalid scaling factor\n");
		return NULL;
	}
	
//	if(channels != 3) {
//		fprintf(stderr,"mexResize:: image should be 3 channeled\n");
//		return NULL;
//	}
	
	printf("rRows %d; rCols %d\n",rRows,rCols);
	
	float *dst = (float *) esvmMalloc(rRows*rCols*channels*sizeof(float));
	float *tmp = (float *) esvmMalloc(rRows*cols*channels*sizeof(int));
	resize1dtran(im,rows,tmp,rRows,cols,channels);
	resize1dtran(tmp,cols,dst,rCols,rRows,channels);
	
	free(tmp);
	int *dstInt = (int *) esvmMalloc(rRows*rCols*channels*sizeof(int));
	
	for(int i=0;i<rRows*rCols*channels;i++) {
		dstInt[i] = (int)dst[i];
	}
	free(dst);
	return dstInt;
}

//accepts a transpose image. returns a non-transposed, resized image.
int *mexResize(float *im, const int rows, const int cols, const int channels,
		const int rRows, const int rCols) 
{
	
	const float scaleFactor = (float)rRows/(float)rows;
	//assert((int)cols*scaleFactor == rCols);
	
	if(scaleFactor > 1 || scaleFactor <=0) {
		fprintf(stderr,"mexResize:: invalid scaling factor\n");
		return NULL;
	}

	if(channels!=3) {
		fprintf(stderr,"mexResize:: input image is not 3 channeled. This is odd, but I will process it anyway.\n");				
	}	

	float *dst = (float *) esvmMalloc(rRows*rCols*channels*sizeof(float));
	float *tmp = (float *) esvmMalloc(rRows*cols*channels*sizeof(int));
	resize1dtran(im,rows,tmp,rRows,cols,channels);
	resize1dtran(tmp,cols,dst,rCols,rRows,channels);

	free(tmp);
	int *dstInt = (int *) esvmMalloc(rRows*rCols*channels*sizeof(int));

	const int dim1 = rRows*rCols;	

	for(int i=0;i<rRows;i++) {
		for(int j=0;j<rCols;j++) {
			dstInt[0*dim1+i*rCols+j] = (int)dst[0*dim1+j*rRows+i];
			dstInt[1*dim1+i*rCols+j] = (int)dst[1*dim1+j*rRows+i];
			dstInt[2*dim1+i*rCols+j] = (int)dst[2*dim1+j*rRows+i];
		}
	}


	free(dst);
	return dstInt;

}