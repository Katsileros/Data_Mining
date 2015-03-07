#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "cuda.h"
#include <thrust/sort.h>


typedef struct{
  float *data;
  int leading_dim;
  int secondary_dim;
} knn_struct;



/*** 
 * Calculates the euclidean distance for a set of points (tx,ty) for the given dimensions of points
 * ***/

__device__ float distance_gpu( int tx, int ty, float *Data, float *Queries, int numAttributes, int numQueries, int numObjects ){
	
	float dist = 0;
	int i;
	
	for(i=0; i<numAttributes; i++){
		float tmp = Data[ (numAttributes * tx) + i ] - Queries[ (numAttributes * ty) + i ];
		dist += ( tmp * tmp );
	}
	return dist;
}


/***    
 * This function calculates the distance from points to queries.
 * Number of threads, been executed == (subObjects * subQueris)
 * The result is the Matrix (QxN) with the calculated distances.
 * ***/

__global__ void gpuDistance(int numObjects, int numQueries, int numAttributes, float *Data, float *Queries, float *Dist){
	
	int tx = ( blockDim.x * blockIdx.x ) + threadIdx.x; // This thread takes ONE point from matrix Data 
	int ty = threadIdx.y; // This thread takes ONE point from matrix Queries
	
	Dist[ty*numObjects + tx] = distance_gpu( tx, ty, Data, Queries, numAttributes, numQueries, numObjects ); 
}

/**
 * This function fills the overall Dist matrix[QxN]
 * Q: Rows for each query
 * N: Columns for each point
 **/ 

void fillDist(float* Dist, float* tmpDist, int dataLeadingDim, int queriesLeadingDim, int N, int Q, int i, int j){
	int k,l;
	
	for(k=0;k<Q;k++){
		for(l=0;l<N;l++){
		  Dist[(((i*Q) + k)*dataLeadingDim) + ((j*N) + l)] = tmpDist[(k*N) + l];
		}
	}
}


void merge(float * list, int *pos, int left_start, int left_end, int right_start, int right_end)
{
	 //~ calculate temporary list sizes 
	int left_length = left_end - left_start;
	int right_length = right_end - right_start;
 
	 //~ declare temporary lists 
	float left_half[left_length];
	int left_half_pos[left_length];
	float right_half[right_length];
	int right_half_pos[right_length];
 
	int r = 0; // right_half index 
	int l = 0; // left_half index 
	int i = 0; // list index 
 
	//~ copy left half of list into left_half 
	for (i = left_start; i < left_end; i++, l++)
	{
		left_half[l] = list[i];
		left_half_pos[l] = pos[i];
	}
 
	//~ copy right half of list into right_half 
	for (i = right_start; i < right_end; i++, r++)
	{
		right_half[r] = list[i];
		right_half_pos[r] = pos[i];
	}
 
	//~ merge left_half and right_half back into list
	for ( i = left_start, r = 0, l = 0; l < left_length && r < right_length; i++)
	{
		if ( left_half[l] < right_half[r] ) {
				pos[i] = left_half_pos[l];
				list[i] = left_half[l++]; 
			}
		else { 
				pos[i] = right_half_pos[r];
				list[i] = right_half[r++]; 
			}
	}
 
	//~ Copy over leftovers of whichever temporary list hasn't finished
	for ( ; l < left_length; i++, l++) {
			list[i] = left_half[l]; 
			pos[i] = left_half_pos[l];
		}
	for ( ; r < right_length; i++, r++) {
			list[i] = right_half[r];
			pos[i] = right_half_pos[r]; 
		}	
}
 
void mergesort_r(int left, int right, float * list, int *pos)
{
	//~ Base case, the list can be no simpiler
	if (right - left <= 1)
	{
		return;
	}
 
	//~ set up bounds to slice array into
	int left_start  = left;
	int left_end    = (left+right)/2;
	int right_start = left_end;
	int right_end   = right;
 
	//~ sort left half
	mergesort_r( left_start, left_end, list, pos);
	//~ sort right half
	mergesort_r( right_start, right_end, list, pos);
 
	//~ merge sorted halves back together
	merge(list, pos, left_start, left_end, right_start, right_end);
}
 
void mergesort(float * list, int *pos, int length)
{
	for(int i=0;i<length;i++){
		pos[i] = i + 1;
	}
	mergesort_r(0, length, list, pos);
}



void fillNeighbours(float* neighbours, float* tmpNeighbours, int k, int Q){
	
	for(int i=0;i<k;i++){
		if(tmpNeighbours[i] < neighbours[(Q*k) + i]){
			neighbours[(Q*k) + i] = tmpNeighbours[i];
		}
	}
	
}

void fillPos(int* pos, int* tmpPos, int k, int Q){
	
	for(int i=0;i<k;i++){
			pos[(Q*k) + i] = tmpPos[i];
	}
	
}


/* Testing point function*/
void test(knn_struct *Data, knn_struct *Queries){
	
	int i,j;
	int tmp = Data->leading_dim-1;

	for( i=0; i<Data->leading_dim; i++ ){ //number of points
		for( j=0; j<Data->secondary_dim; j++ ){ //number of attributes
			Data->data[ (i*Data->secondary_dim) + j] = tmp;
			//~ Data->data[ (i*Data->secondary_dim) + j] = i;
		}
		tmp--;
	}
	
	for( i=0; i<Queries->leading_dim; i++ ){ //number of queries
		for( j=0; j<Queries->secondary_dim; j++ ){ //number of attributes
			Queries->data[ (i * Queries->secondary_dim) + j] = 0;
		}
	}
	
}

/* for debug only */
void print(knn_struct* data2print){

  int i, j = 0;
  int n = data2print->leading_dim;
  int m = data2print->secondary_dim;
  float *tmp_dataset = data2print->data;

  
  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      printf("%f ", tmp_dataset[i*n + j]);
    }
    printf("\n");
  }

  printf("\n");
  
}


void printDist( int N, int Q, float *D ){
	
	int i,j;
	
	for(i=0;i<Q;i++){
		printf("======================================================================================================================================================================================\n");
		printf("\n\n");
		for(j=0;j<N;j++){
		  printf("%f  ", D[ (i*N) + j ]);
		}
		printf("\n");
	}
	
}

void printPos( int N, int Q, int *D ){
	
	int i,j;
	
	for(i=0;i<Q;i++){
		printf("======================================================================================================================================================================================\n");
		printf("\n\n");
		for(j=0;j<N;j++){
		  printf("%d  ", D[ (i*N) + j ]);
		}
		printf("\n");
	}
	
}

void save_f(float* data, char* file, int N, int M){

  FILE *outfile;
  
  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb")) == NULL){
    printf("Can't open output file");
  }

  fwrite(data, sizeof(float), N*M, outfile);

  fclose(outfile);

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

int main( int argc, char *argv[] )
{
	knn_struct Data, Queries, subData, subQueries;	// host
	float *Dist, *tmpDist, *neighbours;	// host
	float *d_Data, *d_Queries, *d_Dist, *d_tmpDist, *d_Neighbours; // device
	int *tmp_pos, *position; // host
	int k=8; //number of neighbours
	
    //~ char *KNNdist_file = "KNNdist_cu.bin";
    //~ char *KNNidx_file = "KNNidx_cu.bin" ;

	
	//memory allocation for host 
	Data.leading_dim = pow(2,20);
	subData.leading_dim = 32;
	Data.secondary_dim = 128;
	Data.data = (float*)malloc( (Data.leading_dim * Data.secondary_dim) * sizeof(float) );
	
	Queries.leading_dim = 1024;
	subQueries.leading_dim = 16;
	Queries.secondary_dim = 128;
	Queries.data = (float*)malloc( (Queries.leading_dim * Queries.secondary_dim) * sizeof(float) );
	subQueries.data = (float*)malloc( (subQueries.leading_dim * Queries.secondary_dim) * sizeof(float) );
	
	printf("\n");	
	
	printf("Data: %d \n", Data.leading_dim);
	printf("Queries: %d \n", Queries.leading_dim);
	printf("k: %d \n", k);
	
	printf("\n");
		
		
	FILE *fp;
 	fp = fopen("data_sift.bin","rb");
	if (fp == NULL){
			printf("Unable to open tmp.bin file! \n");
			return 1;
		}
	
	//Seek to the beginning of the file //
    fseek(fp, SEEK_SET, 0);
		
	Data.data = (float*)malloc((Data.secondary_dim*Data.leading_dim)*sizeof(float));
	
	fread((float*)Data.data, sizeof(float), Data.secondary_dim*Data.leading_dim, fp);
	fclose(fp);
	
	
 	fp = fopen("queries_sift.bin","rb");
	if (fp == NULL){
			printf("Unable to open queries.bin file! \n");
			return 1;
		}
	
	//Seek to the beginning of the file //
    fseek(fp, SEEK_SET, 0);
		
	Queries.data = (float*)malloc((Queries.secondary_dim*Queries.leading_dim)*sizeof(float));
	
	fread((float*)Queries.data, sizeof(float), Queries.secondary_dim*Queries.leading_dim, fp);
	fclose(fp);
	
	
	
	Dist = (float*)malloc( (Data.leading_dim * Queries.leading_dim)*sizeof(float) );
	neighbours = (float*)malloc( (Queries.leading_dim*k)*sizeof(float) );
	position = (int*)malloc( (Queries.leading_dim*k)*sizeof(int) );
	
		
	for(int i=0;i<(Queries.leading_dim*k);i++){
		neighbours[i] = FLT_MAX;
		position[i] = i;
	}
	
	
	/* fill matrices with points*/
	//~ test( &Data, &Queries );
	
	//~ printf("Data: \n");
	//~ print(&Data);
	//~ printf("Queries: \n");
	//~ print(&Queries);

	// compute the execution time
	float elapsedTime = 0 ;
	cudaEvent_t start, stop;
	
	// create event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//memory allocation for device
	cudaMalloc( (void**)&d_Queries, (subQueries.leading_dim * Queries.secondary_dim)  * sizeof(float) );
	cudaMalloc( (void**)&d_Neighbours, (subQueries.leading_dim * k)  * sizeof(float) );

	
	//specify grid and block dimensions
	int gridDimX = 128; /** Be carefull IF gridDimX*subData.leading_dim*Data.secondary_dim exceeds Data.data matrix dimensions !!!  ( max gridDimX = 128 )**/
	int gridDimY = 1;
	int blockDimX =  subData.leading_dim;
	int blockDimY =  subQueries.leading_dim;
	
	int loop1 = 0,loop2 = 0;
	loop2 = (int)(Queries.leading_dim/(subQueries.leading_dim));
	
	// memory allocations depending on gridDimX
	subData.data = (float*)malloc( (subData.leading_dim * Data.secondary_dim * gridDimX) * sizeof(float) );
	tmpDist = (float*)malloc( (subData.leading_dim * subQueries.leading_dim * gridDimX)*sizeof(float) );
	loop1 = (int)(Data.leading_dim/(subData.leading_dim * gridDimX));
	
	tmp_pos = (int*)malloc((Data.leading_dim * subQueries.leading_dim)*sizeof(int));
	
	// memory allocation for device depending on gridDimX
	cudaMalloc( (void**)&d_Data, (subData.leading_dim * Data.secondary_dim * gridDimX) * sizeof(float) );
	cudaMalloc( (void**)&d_Dist,  (subData.leading_dim * subQueries.leading_dim * gridDimX) * sizeof(float) );	
	cudaMalloc( (void**)&d_tmpDist,  (subData.leading_dim * gridDimX) * sizeof(float) );	

	dim3 grid(gridDimX,gridDimY);
	dim3 block(blockDimX,blockDimY);
	
	
	printf("block: (%d,%d) \n", blockDimX, blockDimY);
	printf("grid: (%d,%d) \n", gridDimX, gridDimY);
	
	printf("\n");
	
	printf("loop1: %d \n",loop1);
	printf("loop2: %d \n",loop2);
	
	printf("\n");
	
	// record event
	cudaEventRecord(start,0);
	
	int i,j;
	 
	// Distance between the Q queries from each of the N total points
	// Result is the Dist[QxN] matrix
	for(i=0;i<loop2;i++){ 		// subQueries
		for(j=0;j<loop1;j++){   // subData*gridDimX
	
			cudaMemcpy(d_Data, &Data.data[j*subData.leading_dim*Data.secondary_dim*gridDimX], ((subData.leading_dim)*(Data.secondary_dim)*gridDimX) * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Queries, &Queries.data[i*subQueries.leading_dim*Queries.secondary_dim], ((subQueries.leading_dim) * (Queries.secondary_dim)) * sizeof(float), cudaMemcpyHostToDevice);			
			
			//launch the kernel on GPU
			gpuDistance<<<grid,block>>>( subData.leading_dim*gridDimX, subQueries.leading_dim, Data.secondary_dim, d_Data, d_Queries, d_Dist );
			
			//copy result back to host
			cudaMemcpy( tmpDist, d_Dist, (subData.leading_dim * subQueries.leading_dim * gridDimX) * sizeof(float), cudaMemcpyDeviceToHost );			
			
			fillDist(Dist,tmpDist,Data.leading_dim,Queries.leading_dim,subData.leading_dim*gridDimX,subQueries.leading_dim,i,j);
				
		}		
		// sorting distance matrix for each block of subQueries
			for(int l=0;l<subQueries.leading_dim;l++){
				mergesort(&Dist[l*(Data.leading_dim) + (i*subQueries.leading_dim*Data.leading_dim)], &tmp_pos[l*(Data.leading_dim)], (Data.leading_dim));
				fillNeighbours(neighbours,&Dist[l*(Data.leading_dim) + (i*subQueries.leading_dim*Data.leading_dim)], k, i*subQueries.leading_dim + l);
				fillPos(position,&tmp_pos[l*(Data.leading_dim)], k, i*subQueries.leading_dim + l);
			}
	}
	
	//record event
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&elapsedTime, start, stop);

	//~ printf("Dist: \n");
	//~ printDist(Data.leading_dim, Queries.leading_dim, Dist);
	//~ 
	//~ printf("Neighbours: \n");
	//~ printDist( k, Queries.leading_dim, neighbours);
	//~ printf("\n");
	//~ 
	//~ printf("Position: \n");
	//~ printPos( k, Queries.leading_dim, position);
	//~ printf("\n");
	
	printf("\n Time elapsed: %f ms \n", elapsedTime);


    //~ save_f(neighbours, KNNdist_file, k, Queries.leading_dim);
    //~ save_int(position, KNNidx_file, k, Queries.leading_dim);
	
	/*==== clean host ====*/
	free(Data.data);
	free(Queries.data);
	
	/*==== clean device===*/
	cudaFree(d_Data);
	cudaFree(d_Queries);
	cudaFree(d_Dist);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaDeviceReset();
}
