#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "cuda.h"
#include <thrust/sort.h>
#include <cublas_v2.h>

typedef struct{
  float *data;
  int leading_dim;
  int secondary_dim;
} knn_struct;


/*** 
 * Calculates the hash function for a set of points (tx,ty).
 * ***/

__device__ float hashCalculation_gpu(float x, int w, float r, float b){
	
	return floor( ( (r*x) + b ) / w );
}


/***    
 * This function calculates the hash value for all the dimensions, for a block[dimensions][subData*gridDimX] of points .
 * Number of threads, been executed == (dimensions * subData * gridDimX)
 * The result is a matrix with the calculated hashes. 
 * For every dimension it stores the hash values for every point, so assume that final matrix is [N x D]
 * N = Data.leading_dim
 * D = Dimensions
 * ***/

__global__ void hashFunction(int numObjects, int numAttributes, float *Data, float *hashes, int w, float* r, float* b){
	
	int tx = ( blockDim.x * blockIdx.x ) + threadIdx.x; 
	int ty = threadIdx.y; 
	
	hashes[ty*numObjects + tx] = hashCalculation_gpu(Data[ty*numObjects + tx], w, r[ty], b[ty]);
	__syncthreads();
}

/* Testing point function*/
void test(knn_struct *Data){
	
	int i,j;
	int tmp = Data->leading_dim-1;

	for( i=0; i<Data->leading_dim; i++ ){ //number of points
		for( j=0; j<Data->secondary_dim; j++ ){ //number of attributes
			//~ Data->data[ (i*Data->secondary_dim) + j] = floor((double)rand() * (320+1)/RAND_MAX) + i + j;
			Data->data[ (i*Data->secondary_dim) + j] = tmp;
			//~ Data->data[ (i*Data->secondary_dim) + j] = i;
		}
		tmp--;
	}
}

/* for debug only */
void print(knn_struct* data2print){

  int i,j = 0;
  int n = data2print->leading_dim;
  int m = data2print->secondary_dim;
  float *tmp_dataset = data2print->data;

  //~ for(i=0;i<n*m;i++){
		//~ printf("%f \t",tmp_dataset[i]);
	//~ }
  
  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      printf("%f ", tmp_dataset[j*n + i]);
    }
    printf("\n");
  }

  printf("\n");
  
}

void printHashes( int N, int Q, float *D ){
	
	int i,j;
	
	//~ 
	//~ for(i=0;i<N*Q;i++){
		//~ printf("%f \t",D[i]);
	//~ }
	
	for(i=0;i<N;i++){
		for(j=0;j<Q;j++){
		  printf("%f  ", D[ (j*N) + i ]);
		}
		printf("\n");
	}
}

void myPrint(int* data2print,int x,int y){

  int i, j = 0;
  int n = x;
  int m = y;
  int *tmp_dataset = data2print;

  //~ for(i=0;i<m*n;i++){
	//~ printf("%d   ",tmp_dataset[i]);
	//~ }
  
  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      printf("%d  \t  ", tmp_dataset[j*n + i]);
    }
    printf("\n");
  }

  printf("\n");
  
}

void save_int(int* data, char* file, int N, int M){
	
  FILE *outfile;
  
  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb")) == NULL){
    printf("Can't open output file");
  }

  fwrite(data, sizeof(int), N*M, outfile);

  fclose(outfile);

}

void save_float(float* data, char* file, int N, int M){
	
  FILE *outfile;
  
  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb")) == NULL){
    printf("Can't open output file");
  }

  fwrite(data, sizeof(float), N*M, outfile);

  fclose(outfile);

}

int main( int argc, char *argv[] ){
	
	srand(time(NULL));
	int w=720, i=0; //, j=0;
	
	//~ char *lsh_idx_file = "lsh_idx.bin" ;
	//~ char *hashes_file = "hashes.bin" ;
	
	// Host memory allocations
	knn_struct Data, subData;
	Data.leading_dim = pow(2,20);
	Data.secondary_dim = 30;
	subData.leading_dim = 16; // subData.leading_dim * Data.secondary MUST be < 512. The closer the better
	
	Data.data = (float*)malloc( (Data.leading_dim*Data.secondary_dim) * sizeof(float) );
	subData.data = (float*)malloc( (subData.leading_dim*Data.secondary_dim) * sizeof(float) );
	
	/**
	 * BE CAREFULL: Input data must be in form:
	 * Data[d(0)*N + d(1)*N + ... + d(D)*N]
	**/
	
	FILE *fp;
 	fp = fopen("lsh_sift_data.bin","rb");
	if (fp == NULL){
			printf("Unable to open lsh_data_sift.bin file! \n");
			return 1;
		}
	
	//Seek to the beginning of the file //
    fseek(fp, SEEK_SET, 0);
		
	Data.data = (float*)malloc((Data.secondary_dim*Data.leading_dim)*sizeof(float));
	
	fread((float*)Data.data, sizeof(float), Data.secondary_dim*Data.leading_dim, fp);
	fclose(fp);
	
	// Make the ID matrix
	//~ int* ID = (int*)malloc((Data.secondary_dim*Data.leading_dim)*sizeof(int));
	//~ for(i=0;i<Data.secondary_dim*Data.leading_dim;i++){
			//~ ID[i] = i;
	//~ }
	
	
	// Random point initialization for testing
	
	//~ knn_struct tmpData;
	//~ tmpData.leading_dim = Data.leading_dim;
	//~ tmpData.secondary_dim = Data.secondary_dim;
	
	//~ tmpData.data = (float*)malloc( (Data.leading_dim*Data.secondary_dim) * sizeof(float) );
	
	//~ test(&tmpData);
	
	// make input data in form of a [d(0)*N + d(1)*N + . . . + d(D)*N]
	//~ for(i=0;i<Data.secondary_dim;i++){
		//~ for(j=0;j<Data.leading_dim;j++){
			//~ Data.data[i*Data.leading_dim + j] = tmpData.data[j*Data.secondary_dim + i];
		//~ }
	//~ }
	
	
	//~ printf("Data: \n");
	//~ print(&Data);
	
	float *r, *b;
	r = (float*)malloc(Data.secondary_dim*sizeof(float));
	b = (float*)malloc(Data.secondary_dim*sizeof(float));
	
	// random vectors r[1:d] = rand(0,1) and b[1:d] = rand(0,w)
	for(i=0;i<Data.secondary_dim;i++){
		
		//for testing  -- > h(x(i)) = x(i)
		//~ r[i] = w;
		//~ b[i] = 0;
	
		r[i] = (double)rand() / (double)RAND_MAX ;
		b[i] = ((double)rand() * (w+1)/RAND_MAX);
	}
	
	float* hashes = (float*)malloc((Data.leading_dim*Data.secondary_dim) * sizeof(float)); 
	float* tmp_Hashes = (float*)malloc((subData.leading_dim*Data.secondary_dim) * sizeof(float));

	// compute the execution time
	float elapsedTime = 0 ;
	cudaEvent_t start, stop;
	
	// create event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//specify grid and block dimensions
	int gridDimX = 128; /** Be carefull IF gridDimX*subData.leading_dim*Data.secondary_dim exceeds Data.data matrix!!!  ( max gridDimX = 128, maxBLOCKSIZE = 512 )**/
	int gridDimY = 1;
	int blockDimX =  subData.leading_dim;
	int blockDimY =  Data.secondary_dim;
	dim3 dimBlock( blockDimX, blockDimY );
	dim3 dimGrid( gridDimX, gridDimY);
	
		
	printf("GRID: %d \t %d \n",gridDimX,gridDimY);
	printf("BLOCK: %d \t %d \n",blockDimX,blockDimY);
	
	// Device memory allocations
	float* d_Hashes, *d_Data, *d_r, *d_b;
	cudaMalloc( (void**)&d_Data, (subData.leading_dim*Data.secondary_dim*gridDimX) * sizeof(float));
	cudaMalloc( (void**)&d_Hashes, (subData.leading_dim*Data.secondary_dim*gridDimX) * sizeof(float));
	
	cudaMalloc( (void**)&d_r, (Data.secondary_dim) * sizeof(float));
	cudaMalloc( (void**)&d_b, (Data.secondary_dim) * sizeof(float));
	
	cudaMemcpy(d_r, r, (Data.secondary_dim) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, (Data.secondary_dim) * sizeof(float), cudaMemcpyHostToDevice);
	
	
	int loop1 = 0;
	loop1 = ceil((int)(Data.leading_dim/(subData.leading_dim * gridDimX)));
	printf("loop1 = %d \n", loop1);
	

	cudaEventRecord(start,0);
	
	for(i=0;i<loop1;i++){   // subData*gridDimX
	
		// Copy data to GPU
		cudaMemcpy(d_Data, &Data.data[i*subData.leading_dim*Data.secondary_dim*gridDimX], ((subData.leading_dim)*(Data.secondary_dim)*gridDimX) * sizeof(float), cudaMemcpyHostToDevice);
		
		// Kernel execution
		hashFunction<<<dimGrid,dimBlock>>> ((subData.leading_dim)*(gridDimX), Data.secondary_dim, d_Data, d_Hashes, w, d_r, d_b);
		
		// Copy data back to host
		cudaMemcpy( &hashes[i*subData.leading_dim*Data.secondary_dim*gridDimX], d_Hashes, (subData.leading_dim*Data.secondary_dim * gridDimX) * sizeof(float), cudaMemcpyDeviceToHost );
	}
	
	//~ printf("\nHashes b4 sort\n");
	//~ printHashes(Data.leading_dim, Data.secondary_dim, hashes);
	
	// Sort the hash values
	for(i=Data.secondary_dim-1;i>=0;i--){
		thrust::stable_sort_by_key(&hashes[i*Data.leading_dim], &hashes[i*Data.leading_dim] + (Data.leading_dim), &Data.data[i*Data.leading_dim]);
		}
		
	//~ printf("\nHashes after sort\n");
	//~ printHashes(Data.leading_dim, Data.secondary_dim, hashes);
	
	//~ printf("\n");
	
	//~ printf("Data After sort: \n");
	//~ print(&Data);
	
	//record event
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	printf("\n Time elapsed: %f ms \n", elapsedTime);
	
	//~ save_float(hashes, hashes_file, Data.leading_dim, Data.secondary_dim);
	//~ save_int(ID, lsh_idx_file, Data.leading_dim, Data.secondary_dim);
	
	/*==== clean host ====*/
	free(Data.data);
	free(subData.data);
	
	/*==== clean device===*/
	cudaFree(d_Data);
	cudaFree(d_Hashes);
	cudaFree(d_r);
	cudaFree(d_b);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaDeviceReset();
	
}
