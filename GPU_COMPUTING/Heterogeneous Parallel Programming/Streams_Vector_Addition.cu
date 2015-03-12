#include	<wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<len)
	{
		out[i] = in1[i] + in2[i];
	}
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
	float * deviceInput1;
	float * deviceInput2;
	float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    //hostOutput = (float *) malloc(inputLength * sizeof(float));
	cudaHostAlloc((void **)&hostOutput,inputLength*sizeof(float),cudaHostAllocDefault);
    wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);

  	wbTime_start(GPU, "Allocating GPU memory.");
  	//@@ Allocate GPU memory here
	cudaMalloc((void**) &deviceInput1, (inputLength)*sizeof(float));
	cudaMalloc((void**) &deviceInput2, (inputLength)*sizeof(float));
	cudaMalloc((void**) &deviceOutput, (inputLength)*sizeof(float));

  	wbTime_stop(GPU, "Allocating GPU memory.");
	
	//@@ Initialize the grid and block dimensions here
	dim3 DimGrid(16, 1, 1);
	dim3 DimBlock(256, 1, 1);
	
	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	
	
  	wbTime_start(GPU, "Start CUDA Vector Addition with streams");
  	
	//@@ Async copies to GPU
		//@@ 1st stream
		cudaMemcpyAsync(&deviceInput1[0*(inputLength/4)], &hostInput1[0*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(&deviceInput2[0*(inputLength/4)], &hostInput2[0*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream0);
		//@@ 2nd stream
		cudaMemcpyAsync(&deviceInput1[1*(inputLength/4)], &hostInput1[1*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(&deviceInput2[1*(inputLength/4)], &hostInput2[1*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream1);
		//@@ 3rd stream
		cudaMemcpyAsync(&deviceInput1[2*(inputLength/4)], &hostInput1[2*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(&deviceInput2[2*(inputLength/4)], &hostInput2[2*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream2);
		//@@ 4th stream
		cudaMemcpyAsync(&deviceInput1[3*(inputLength/4)], &hostInput1[3*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(&deviceInput2[3*(inputLength/4)], &hostInput2[3*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyHostToDevice, stream3);
	
	//@@ Kernels
		//@@ 1st stream
		vecAdd<<<DimGrid,DimBlock,0,stream0>>>(&deviceInput1[0*(inputLength/4)],&deviceInput2[0*(inputLength/4)],deviceOutput,inputLength/4);
		//@@ 2nd stream
		vecAdd<<<DimGrid,DimBlock,0,stream1>>>(&deviceInput1[1*(inputLength/4)],&deviceInput2[1*(inputLength/4)],&deviceOutput[1*(inputLength/4)],inputLength/4);	
		//@@ 3rd stream
		vecAdd<<<DimGrid,DimBlock,0,stream2>>>(&deviceInput1[2*(inputLength/4)],&deviceInput2[2*(inputLength/4)],&deviceOutput[2*(inputLength/4)],inputLength/4);
		//@@ 4th stream
		vecAdd<<<DimGrid,DimBlock,0,stream3>>>(&deviceInput1[3*(inputLength/4)],&deviceInput2[3*(inputLength/4)],&deviceOutput[3*(inputLength/4)],inputLength/4);

	//@@ Async copies back to Host
		//@@ 1st stream
		cudaMemcpyAsync(&hostOutput[0*(inputLength/4)], &deviceOutput[0*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyDeviceToHost, stream0);
		//@@ 2nd stream	
		cudaMemcpyAsync(&hostOutput[1*(inputLength/4)], &deviceOutput[1*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyDeviceToHost, stream1);
		//@@ 3rd stream	
		cudaMemcpyAsync(&hostOutput[2*((inputLength/4))], &deviceOutput[2*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyDeviceToHost, stream2);
		//@@ 4th stream
		cudaMemcpyAsync(&hostOutput[3*(inputLength/4)], &deviceOutput[3*(inputLength/4)], (inputLength/4)*sizeof(float), cudaMemcpyDeviceToHost, stream3);

  	wbTime_stop(GPU, "Start CUDA Vector Addition with streams");

	//@@ Wait until all streams have finished their job
	cudaDeviceSynchronize();

  	//@@ Free the GPU memory here	
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);
	

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
