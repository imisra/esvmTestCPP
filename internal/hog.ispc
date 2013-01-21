/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 * 
 * hog.ispc: Based on Pedro's features.cc 
 */
// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
float uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
float vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};
task void computeGradient(uniform int userTasks, uniform int im[], uniform int rows, uniform int cols, uniform int channels, uniform int cellWidth, uniform int visible[2], uniform int blocks[2], uniform float hist[], uniform float gradient[], uniform int best_ori[]) 
{
	//Sanity check. If user defined totalTasks are different from taskCount,
	//then the user has either allocated more or less memory for the hist array.
	//this arises from the fact that the hist array has memory allocated for task-specific historgrams.

	if(taskCount>userTasks)
	{
		print("computeGradient:error:: user specified tasks are less than ISPC launched tasks.");
		return;
	}

	uniform int dim1 = rows*cols;
	uniform int granularity = (visible[0]-1)/taskCount;
	uniform int startX = 1 + granularity*(int)taskIndex;
	uniform int endX = (taskIndex==taskCount-1) ? visible[0]-1: startX+granularity;

	foreach(i=startX ... endX, j=1 ... visible[1]-1) {

		//FIXME:: Might need to change the image layout in memory.
		//Right now all of the pointer memory access are serialized 
		//It gives the Performance Warning: Gather required to load value.

		//		print("#% %#\n",i,j);

		//R-channel
		int *s = im + min(i,rows-2)*cols + min(j,cols-2);
		float dx0 = *(s+cols) - *(s-cols);
		float dy0 = *(s+1) - *(s-1);
		float v0 = dx0*dx0 + dy0*dy0;


		//G-channel
		s = s+dim1;
		float dx1 = *(s+cols) - *(s-cols);
		float dy1 = *(s+1) - *(s-1);
		float v1 = dx1*dx1 + dy1*dy1;

		//B-channel
		s = s+dim1;
		float dx2 = *(s+cols) - *(s-cols);
		float dy2 = *(s+1) - *(s-1);
		float v2 = dx2*dx2 + dy2*dy2;

		if (v1 > v0) {
			v0 = v1;
			dx0 = dx1;
			dy0 = dy1;
		} 
		if (v2 > v0) {
			v0 = v2;
			dx0 = dx2;
			dy0 = dy2;
		}

	
		// snap to one of 18 orientations
		float best_dot = 0;
		int best_o = 0;
		for (int o = 0; o < 9; o++) {
			float dot = uu[o]*dy0 + vv[o]*dx0;
			if (dot > best_dot) {
				best_dot = dot;
				best_o = o;
			} else if (-dot > best_dot) {
				best_dot = -dot;
				best_o = o+9;
			}
		}
		gradient[i*visible[1]+j]=v0;
		best_ori[i*visible[1]+j]=best_o;


	}

}

task void binHist(uniform float hist[], uniform float gradient[], uniform int best_ori[], uniform int visible[2], uniform int blocks[2], uniform int cellWidth, uniform int userTasks)
{
	uniform int granularity = (visible[0]-1)/taskCount;
	uniform int startX = 1 + granularity*(int)taskIndex;
	uniform int endX = (taskIndex==taskCount-1) ? visible[0]-1: startX+granularity;

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

task void mergeHist(uniform float hist[], uniform int userTasks, uniform int blocks[])
{

	uniform int granularity = (blocks[0])/taskCount;
	uniform int startX = granularity*(int)taskIndex;
	uniform int endX = (taskIndex==taskCount-1) ? blocks[0]: startX+granularity;

	//Note:: The actual parallelism in the problem is each cell-slice of the mxnx18 histogram.
	//so ideally we should be able to do something like let tasks take up their own rows.
	//within the task, split the cell-slices amongst gangs.
	//but each cell slice needs to be done serially.
	uniform int mn = blocks[0]*blocks[1];


	foreach(i=startX ... endX, j=0 ... blocks[1]) {
	
		for(int u = 1;u<userTasks;u++) {
			float *dst = hist+i*blocks[1]+j;
			float *src = dst+u*18*mn;
			for(int o=0;o<18;o++) {
				(*dst)+=(*src);
				src+=mn;
				dst+=mn;
			}
		}

	}
	
}


task void computeFeatures(uniform float hog[], uniform int blocks[2], uniform float hist[], uniform float norm[], uniform int out[3]) 
{
	
	uniform int granularity = (out[0])/taskCount;
	uniform int startX = granularity*(int)taskIndex;
	uniform int endX = (taskIndex==taskCount-1) ? out[0]: startX+granularity;
	
	// compute features
	foreach(x=startX ... endX, y=0 ... out[1])
	{
		
		float *dst = hog + x*out[1] + y;      
		float *src, *p, n1, n2, n3, n4;

		p = norm + (x+1)*blocks[1] + y+1;
		n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
		p = norm + (x+1)*blocks[1] + y;
		n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
		p = norm + x*blocks[1] + y+1;
		n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
		p = norm + x*blocks[1] + y;      
		n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);

		float t1 = 0;
		float t2 = 0;
		float t3 = 0;
		float t4 = 0;

		// contrast-sensitive features
		//FIXME:: Can we use a reduce primitive for these operations ??
		src = hist + (x+1)*blocks[1] + (y+1);

		for (int o = 0; o < 18; o++) {
			float h1 = min(*src * n1, 0.2);
			float h2 = min(*src * n2, 0.2);
			float h3 = min(*src * n3, 0.2);
			float h4 = min(*src * n4, 0.2);
			*dst = 0.5 * (h1 + h2 + h3 + h4);
			t1 += h1;
			t2 += h2;
			t3 += h3;
			t4 += h4;
			dst += out[0]*out[1];
			src += blocks[0]*blocks[1];
		}

		

		// contrast-insensitive features
		src = hist + (x+1)*blocks[1] + (y+1);
		for (int o = 0; o < 9; o++) {
			float sum = *src + *(src + 9*blocks[0]*blocks[1]);
			float h1 = min(sum * n1, 0.2);
			float h2 = min(sum * n2, 0.2);
			float h3 = min(sum * n3, 0.2);
			float h4 = min(sum * n4, 0.2);
			*dst = 0.5 * (h1 + h2 + h3 + h4);
			dst += out[0]*out[1];
			src += blocks[0]*blocks[1];
		}

		

		// texture features
		*dst = 0.2357 * t1;
		dst += out[0]*out[1];
		*dst = 0.2357 * t3;
		dst += out[0]*out[1];
		*dst = 0.2357 * t2;
		dst += out[0]*out[1];
		*dst = 0.2357 * t4;
	}

}


export void getGradient(uniform int im[], uniform int rows, uniform int cols, uniform int channels, uniform int cellWidth, uniform int visible[2], uniform int blocks[2], uniform float hist[], uniform float gradient[], uniform int best_ori[], uniform int userTasks)
{
	launch[userTasks] computeGradient(userTasks, im, rows, cols, channels, cellWidth, visible, blocks, hist, gradient, best_ori); 
	sync;
}


export void combineHist(uniform float hist[], uniform int userTasks, uniform int blocks[2])
{
	launch[userTasks]mergeHist(hist, userTasks, blocks);
	sync;
}
export void getFeatures(uniform float hog[], uniform int blocks[2], uniform float hist[], uniform float norm[], uniform int out[3], uniform int userTasks) 
{
	launch[userTasks] computeFeatures(hog, blocks, hist, norm, out) ;
	sync;
}

