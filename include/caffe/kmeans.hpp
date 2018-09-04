#ifndef KMEANS_HPP_
#define KMEANS_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <limits>
#include <string.h>


using namespace std;
template<typename Dtype>
void kmeans_cluster(/*vector<int> &*/int *cLabel, /*vector<Dtype> &*/Dtype *cCentro, Dtype *cWeights, int nWeights, int *mask/*vector<int> &mask*/, /*Dtype maxWeight, Dtype minWeight,*/  int nCluster,  int max_iter /* = 1000 */)
{
        //1. find min max from the weight, in order to generate initial centroid linearly
        Dtype maxWeight=numeric_limits<Dtype>::min(), minWeight=numeric_limits<Dtype>::max();
        for(int k = 0; k < nWeights; ++k)
        {
            //mask[k]=1 means weight value was not set to zero ----Solomon
            if(mask[k])
            {
                if(cWeights[k] > maxWeight)
                    maxWeight = cWeights[k];
                if(cWeights[k] < minWeight)
                    minWeight = cWeights[k];
            }
        }
	//2. generate initial centroids linearly based on the minWeight and maxWeight
	for (int k = 0; k < nCluster; k++)
		cCentro[k] = minWeight + (maxWeight - minWeight)*k / (nCluster - 1);

	//3. initialize all label to -1, cLabel means cluster index, and each Label need
        //   Log2(nCluster) bits to store.
	for (int k = 0; k < nWeights; ++k)
		cLabel[k] = -1;

	const Dtype float_max = numeric_limits<Dtype>::max();
	
	//4. initialize, nWeights means how many elements in sparse matrix
        //   and nCluster means the centroids number
	Dtype *cDistance = new Dtype[nWeights];
	int *cClusterSize = new int[nCluster];

	Dtype *pCentroPos = new Dtype[nCluster];
	int *pClusterSize = new int[nCluster];
	memset(pClusterSize, 0, sizeof(int)*nCluster);
	memset(pCentroPos, 0, sizeof(Dtype)*nCluster);
	Dtype *ptrC = new Dtype[nCluster];
	int *ptrS = new int[nCluster];

	int iter = 0;
	//Dtype tk1 = 0.f, tk2 = 0.f, tk3 = 0.f;
	double mCurDistance = 0.0;
	double mPreDistance = numeric_limits<double>::max();

	//5. clustering
	while (iter < max_iter)
	{
		// a. check convergence
		if (fabs(mPreDistance - mCurDistance) / mPreDistance < 0.01) break;
		mPreDistance = mCurDistance;
		mCurDistance = 0.0;

		// b. select nearest cluster, find the nearest cluster to each input weights
		for (int n = 0; n < nWeights; n++)
		{
			//elements which already be set to 0 will not participate clustering ----Solomon
			if (!mask[n])
				continue;
			Dtype distance;
			Dtype mindistance = float_max;
			int clostCluster = -1;
			for (int k = 0; k < nCluster; k++)
			{
				distance = fabs(cWeights[n] - cCentro[k]);
				if (distance < mindistance)
				{
					mindistance = distance;
					clostCluster = k;
				}
			}
			cDistance[n] = mindistance;
			//record the Nth weight is closest to clostCluster
			cLabel[n] = clostCluster;
		}


		//c. calc new distance/inertia, mCurDistance record the total distance which from all weights to their respective centers
		for (int n = 0; n < nWeights; n++)
		{
			if (mask[n])
				mCurDistance = mCurDistance + cDistance[n];
		}


	// generate new centroids
	// accumulation(private)

		for (int k = 0; k < nCluster; k++)
		{
			ptrC[k] = 0.f;
			ptrS[k] = 0;
		}

		for (int n = 0; n < nWeights; n++)
		{
			if (mask[n])
			{
				//prtC record a centroid bin was put total summation weight value
				//ptrS record a centroid bin was put how many weight in it
				ptrC[cLabel[n]] += cWeights[n];
				ptrS[cLabel[n]] += 1;
			}
		}

		for (int k = 0; k < nCluster; k++)
		{
			pCentroPos[ k] = ptrC[k];
			pClusterSize[k] = ptrS[k];
		}

		//reduction(global)
		for (int k = 0; k < nCluster; k++)
		{

			cCentro[k] = pCentroPos[k];
			cClusterSize[k] = pClusterSize[k];
	
			cCentro[k] /= cClusterSize[k];
		}

		iter++;
	//	cout << "Iteration: " << iter << " Distance: " << mCurDistance << endl;
	}
	//gather centroids
	//#pragma omp parallel for
	//for(int n=0; n<nNode; n++)
	//    cNodes[n] = cCentro[cLabel[n]];

	delete[] cDistance;
	delete[] cClusterSize;
	delete[] pClusterSize;
	delete[] pCentroPos;
	delete[] ptrC;
	delete[] ptrS;
}



#endif
