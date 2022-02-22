#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include "structSET.h"


double updateDistanceOfPixForLabel(const unsigned int x, 
									unsigned const int y, 
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

	if (nbPxS==0)
	{
		distS =sqrtf(powf(xTransf-xmeanInit,2) +powf(yTransf-ymeanInit,2));
	}
	else 
	{
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

	return out;
}

void updateDistanceOfLabel(double * map, 
							unsigned long * labels, 
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
	double tmp;
	unsigned int i,j;
	//printf("label %ld", nameLabel);
	//fflush(stdout);	
	#pragma omp parallel for private(i,j,tmp) shared(map,labels) 
	for (i=0; i<height; i++)
	{

		for (j=0; j<width; j++)
		{
			tmp =  updateDistanceOfPixForLabel(i, j, nameLabel, angle, xmean, ymean, angleInit, xmeanInit, ymeanInit,skePx,cPx,nbPxS,nbPxC, segmentation, ske, contours, height, width);
			if (tmp < map[(j*height)+i] )
			{
				map[(j*height)+i] = tmp; 
				labels[(j*height)+i] = nameLabel;
			}
			

		}

	}
}

void initSET(setImage *set, 
			const unsigned long nbLabels, 
			const long * segmentationIn, 
			const short * skeIn, 
			const long * contoursIn, 
			const unsigned long height, 
			const unsigned long width)
{
	(*set).height = height;
	(*set).width = width;
	(*set).nbLabels = nbLabels;

	(*set).nbPxS = (int *) malloc(nbLabels*sizeof(int));
	(*set).nbPxC = (int *) malloc(nbLabels*sizeof(int));

	(*set).pxS = (int **) malloc(nbLabels*sizeof(int*));
	(*set).pxC = (int **) malloc(nbLabels*sizeof(int*));

	for(int l = 0; l<nbLabels; l++)
	{
		(*set).nbPxS[l] = 0;
		(*set).nbPxC[l] = 0;
		(*set).pxC[l] = (int *) malloc(sizeof(int)*2*width*height);
		(*set).pxS[l] = (int *) malloc(sizeof(int)*2*width*height);

	}
	int i,j,l;
	
	//#pragma omp parallel for private(i,j,l) shared(contoursIn,segmentationIn,skeIn,set) 
	// data preprocessing

	for (i=0; i<height; i++)
	{

		for (j=0; j<width; j++)
		{
			if (contoursIn[(j*height)+i] != -1)
			{
			l = contoursIn[(j*height)+i];
			(*set).pxC[l][((*set).nbPxC[l])*2] = i;
			(*set).pxC[l][((*set).nbPxC[l])*2+1] = j;
			(*set).nbPxC[l] +=1;
				
			}
			if ((segmentationIn[(j*height)+i] != -1) && (skeIn[(j*height)+i] != 0) )
			{
			l = segmentationIn[(j*height)+i];
			/*
			if( l==192)
			{
				printf("(%d,%d)\n",i,j);
				fflush(stdout);
				
			}*/
			(*set).pxS[l][((*set).nbPxS[l])*2] = i;
			(*set).pxS[l][((*set).nbPxS[l])*2+1] = j;
			(*set).nbPxS[l] +=1;

		
			}

		}
	}

	for(l = 0; l<nbLabels; l++)
	{
		(*set).pxS[l] = (int *) realloc((*set).pxS[l], (*set).nbPxS[l]*2*sizeof(int));
		(*set).pxC[l] = (int *) realloc((*set).pxC[l], (*set).nbPxC[l]*2*sizeof(int));

		if ((*set).nbPxS[l]==0)
		{
			printf("no skeleton for label at index %d \n",l);
		}

	}


}

void freeMemory(setImage *set)
{

	for(int l = 0; l<((*set).nbLabels); l++)
	{
		free((*set).pxS[l]);
		free((*set).pxC[l]);
	}

	free((*set).nbPxC);
	free((*set).nbPxS);

	free((*set).pxC);
	free((*set).pxS);
}

void fullTesselation(double * map, 
					unsigned long * labels, 
					const unsigned long nbLabels, 
					const unsigned long *nameLabels, 
					const double *angles, 
					const double *posMeans, 
					const double *angleInits, 
					const double *posMeansInit, 
					const long * segmentation, 
					const short * ske, 
					const long * contours, 
					const unsigned long height, 
					const unsigned long width,
					setImage *set)
{


	
	unsigned int iL;
	for (iL=0; iL<nbLabels; iL++) //nbLabels
		{
		//printf("id %d, label %ld", iL, nameLabels[iL]);
		fflush(stdout);	
		updateDistanceOfLabel(map, 
							labels, 
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
							segmentation, 
							ske, 
							contours, 
							height, 
							width);
		}
	

}

