#include <math.h>
#include <stdio.h>

#include <time.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include"structSET.h"

#define BLOCK_SIZE 16 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void updateDistanceOfPixForLabel(double *map, 
											unsigned long *labels, 
											const unsigned long nameLabel, 
											const double angle, 
											const double xmean, 
											const double ymean, 
											const double angleInit, 
											const double xmeanInit, 
											const double ymeanInit, 
											const int * skePx, 
											const int * cPx, 
											const int nbPxS, 
											const int nbPxC, 
											const long * segmentation, 
											const short * ske, 
											const long * contours, 
											const unsigned long height, 
											const unsigned long width)
{

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; //threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y; //threadIdx.y;
	if (x >=height  || y >= width) return;


	float xTransf = ((x-xmean)*cos(angle-angleInit) - (y-ymean)*sin(angle-angleInit)) + xmeanInit;
	float yTransf = ((x-xmean)*sin(angle-angleInit) + (y-ymean)*cos(angle-angleInit)) + ymeanInit;

	int xTransf_px = (int)floor(xTransf);
	int yTransf_px = (int)floor(yTransf);
	if (xTransf_px<0 )
	{
		xTransf_px = 0;
	}
	else if (xTransf_px>=height)
	{
		xTransf_px = height-1;
	}
	if (yTransf_px<0)
	{
		yTransf_px = 0;
	}
	else if (yTransf_px>=width)
	{
		yTransf_px = width -1;
	}
	int xf_1 = xTransf_px +1 ;
	int yf_1 = yTransf_px +1 ;
	if (xf_1<0 )
	{
		xf_1 = 0;
	}
	else if (xf_1>=height)
	{
		xf_1 = height-1;
	}
	if (yf_1<0)
	{
		yf_1 = 0;
	}
	else if (yf_1>=width)
	{
		yf_1 = width -1;
	}
	float wx = xTransf-xTransf_px;
	float wy = yTransf-yTransf_px;
	float w00 = (1-wy)*(1-wx);
	float w10 = wy*(1-wx);
	float w01 = (1-wy)*wx;
	float w11 = wy*wx;
	
	float interpolateValueC = w00*(contours[(yTransf_px*height)+xTransf_px] == nameLabel)
							+w10*(contours[(yf_1*height)+xTransf_px] == nameLabel) 
							+w01*(contours[(yTransf_px*height)+xf_1] == nameLabel)
							+w11*(contours[(yf_1*height)+xf_1] == nameLabel);
	float interpolateValueSeg = w00*(segmentation[(yTransf_px*height)+xTransf_px] == nameLabel)
								+w10*(segmentation[(yf_1*height)+xTransf_px] == nameLabel)
								+w01*(segmentation[(yTransf_px*height)+xf_1] == nameLabel)
								+w11*(segmentation[(yf_1*height)+xf_1] == nameLabel);

	int isPixInLabel = (interpolateValueC >0.5) || (interpolateValueSeg>0.5);
 
	double distC = INFINITY ; 
	double distS = INFINITY ; 
	double tmpDist;


	int i,xi,yi;
	for (i=0; i<nbPxC; i++)
	{
	xi = cPx[i*2];
	yi = cPx[i*2+1];

		tmpDist = sqrtf(powf(xTransf-xi,2) +powf(yTransf-yi,2));
			
		if (tmpDist < distC)
		{
			distC = tmpDist;
		}	
			
	}

	for (i=0; i<nbPxS; i++)
	{
		xi = skePx[i*2];
		yi = skePx[i*2+1];

		tmpDist = sqrtf(powf(xTransf-xi,2) +powf(yTransf-yi,2));
		if (tmpDist < distS)
		{
			distS = tmpDist;
		}	
		
	}

	double out;
	if (isPixInLabel == 1)
	{
		out = distS / (distS+distC);
	}
	else 
	{
		if (distS == distC)
		{
			distS = sqrtf(powf(xTransf-xmeanInit,2) +powf(yTransf-ymeanInit,2));
			out = distS/ (distS-distC);
		}
		else 
		{
			out = distS / (distS-distC);
		}
	}
	//out = 1/(1+exp(-5*out));
	//printf(" out : %f \n",out);
	//printf(" map : %f \n",map[(y*height)+x]);
	if (out < map[(y*height)+x] )
	{
		map[(y*height)+x] = out; //(*(map+(i*width)+j))
		labels[(y*height)+x] = nameLabel ; //sqrtf(powf(xTransf-xmeanInit,2) +powf(yTransf-ymeanInit,2));
	}	


}


extern "C"{

void initSETonGPU(setImage *set, 
					const unsigned long nbLabels, 
					long * segmentationIn, 
					short * skeIn, 
					long * contoursIn, 
					const unsigned long height, 
					const unsigned long width, 
					const int gpu)
{	
	cudaSetDevice(gpu);
	int device;
	cudaGetDevice(&device);
	printf("device : %d \n",device);
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);
	printf("using %d multiprocessors \n",properties.multiProcessorCount);
	printf("max threads per processor: %d \n",properties.maxThreadsPerMultiProcessor);


	(*set).height = height;
	(*set).width = width;
	(*set).nbLabels = nbLabels;

	(*set).nbPxS = (int *) malloc(nbLabels*sizeof(int));
	(*set).nbPxC = (int *) malloc(nbLabels*sizeof(int));

	int **pxCIn = (int **) malloc(nbLabels*sizeof(int*));
	int **pxSIn = (int **) malloc(nbLabels*sizeof(int*));
	for(int l = 0; l<nbLabels; l++)
	{
		(*set).nbPxS[l] = 0;
		(*set).nbPxC[l] = 0;
		pxCIn[l] = (int *) malloc(sizeof(int)*2*width*height);
		pxSIn[l] = (int *) malloc(sizeof(int)*2*width*height);
	}

	// data preprocessing
	int i,j,l;
	for (i=0; i<height; i++)
	{

		for (j=0; j<width; j++)
		{
			if (contoursIn[(j*height)+i] != -1)
			{
			l = contoursIn[(j*height)+i];
			pxCIn[l][((*set).nbPxC[l])*2] = i;
			pxCIn[l][((*set).nbPxC[l])*2+1] = j;
			(*set).nbPxC[l] +=1;
				
			}
			if ((segmentationIn[(j*height)+i] != -1) && (skeIn[(j*height)+i] != 0) )
			{
			l = segmentationIn[(j*height)+i];
			pxSIn[l][((*set).nbPxS[l])*2] = i;
			pxSIn[l][((*set).nbPxS[l])*2+1] = j;
			(*set).nbPxS[l] +=1;
		
			}

		}
	}


	(*set).pxC = (int **) malloc(nbLabels*sizeof(int*));
	(*set).pxS = (int **) malloc(nbLabels*sizeof(int*));

	for(int l = 0; l<nbLabels; l++)
	{
		pxCIn[l] = (int *) realloc(pxCIn[l], (*set).nbPxC[l]*2*sizeof(int));
		pxSIn[l] = (int *) realloc(pxSIn[l], (*set).nbPxS[l]*2*sizeof(int));

		gpuErrchk(cudaMalloc((int**)&((*set).pxS[l]), sizeof(int)*2*(*set).nbPxS[l]));
		gpuErrchk(cudaMalloc((int**)&((*set).pxC[l]), sizeof(int)*2*(*set).nbPxC[l]));

		gpuErrchk(cudaMemcpy((*set).pxS[l], pxSIn[l], sizeof(int)*2*(*set).nbPxS[l], cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy((*set).pxC[l], pxCIn[l], sizeof(int)*2*(*set).nbPxC[l], cudaMemcpyHostToDevice));

		free(pxCIn[l]);
		free(pxSIn[l]);
		if ((*set).nbPxC[l]==0)
		{
			printf("no skeleton for label at index %d \n",l);
		}
	}
	free(pxCIn);
	free(pxSIn);	


	// cuda init
	const int sizeSegC = height * width * sizeof(long);
	const int sizeSke = height * width * sizeof(short);
	const int sizeM = height * width * sizeof(double);
	const int sizeL = height * width * sizeof(unsigned long);



    gpuErrchk(cudaMalloc((long**)&((*set).segmentation), sizeSegC));
    gpuErrchk(cudaMalloc((short**)&((*set).ske), sizeSke));
    gpuErrchk(cudaMalloc((long**)&((*set).contours), sizeSegC));
	gpuErrchk(cudaMalloc((double**)&((*set).map), sizeM));
    gpuErrchk(cudaMalloc((unsigned long**)&((*set).labels), sizeL));


	gpuErrchk(cudaMemcpy((*set).segmentation, segmentationIn, sizeSegC, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((*set).ske, skeIn, sizeSke, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((*set).contours, contoursIn, sizeSegC, cudaMemcpyHostToDevice));




}

void freeMemory(setImage *set)
{
	cudaFree((*set).labels);
	cudaFree((*set).map);

	cudaFree((*set).segmentation);
	cudaFree((*set).ske);
	cudaFree((*set).contours);

	for(int l = 0; l<((*set).nbLabels); l++)
	{
		cudaFree((*set).pxS[l]);
		cudaFree((*set).pxC[l]);
	}

	free((*set).nbPxC);
	free((*set).nbPxS);

	free((*set).pxC);
	free((*set).pxS);
}

void fullTesselation(double * mapOut, 
					unsigned long * labelsOut, 
					const unsigned long *nameLabels, 
					const double *angles, 
					const double *posMeans, 
					const double *angleInits, 
					const double *posMeansInit, 
					setImage *set)
{

	const int sizeM = (*set).height * (*set).width * sizeof(double);
	const int sizeL = (*set).height * (*set).width * sizeof(unsigned long);
	gpuErrchk(cudaMemcpy((*set).map, mapOut, sizeM, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((*set).labels, labelsOut, sizeL, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(ceil((float)(*set).height/ threadsPerBlock.x), ceil((float)(*set).width/threadsPerBlock.y));

	int iL;
	for (iL=0; iL<(*set).nbLabels; iL++) 
	{
		updateDistanceOfPixForLabel<<< numBlocks,threadsPerBlock>>>
				((*set).map, 
				(*set).labels, 
				nameLabels[iL],
				angles[iL],
				posMeans[iL*2+1],
				posMeans[iL*2],
				angleInits[iL],
				posMeansInit[iL*2+1],
				posMeansInit[iL*2],
				(*set).pxS[iL],
				(*set).pxC[iL],
				(*set).nbPxS[iL],
				(*set).nbPxC[iL],
				 (*set).segmentation,
				 (*set).ske,
				 (*set).contours,
				 (*set).height,
				 (*set).width) ; 
		gpuErrchk(cudaGetLastError()); 
		gpuErrchk( cudaDeviceSynchronize()); 
	}
	cudaMemcpy(mapOut, (*set).map, sizeM,  cudaMemcpyDeviceToHost);
	cudaMemcpy(labelsOut, (*set).labels, sizeL,  cudaMemcpyDeviceToHost);


}

}
