// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 16


//@@ insert code here
__global__ void histo_kernel(int *buffer, long size, int *histo)
{
	__shared__ int histo_private[256];
	
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	if(threadIdx.x < 256)
		histo_private[threadIdx.x] = 0;
	__syncthreads();
	
	
	while(i < size)
	{ 
		if(buffer[i] != 0)
		{
			atomicAdd(&(histo_private[buffer[i]]),(int)1);
		}
		i = i + stride;
	}
	
	// Wait for all threads in the block to finish
	__syncthreads();
	
	
	if( threadIdx.x < 256)
 		atomicAdd(&(histo[threadIdx.x]),(int)histo_private[threadIdx.x]);	
	__syncthreads();
}

__global__ void rgb2gray(unsigned char* ucharImage, int length, int* devGrayImage)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	
	while(i < (length / 3))
	{
		unsigned char r = ucharImage[3*i];
		unsigned char g = ucharImage[3*i + 1];
		unsigned char b = ucharImage[3*i + 2];
		
		//devGrayImage[i] = i;
		
		devGrayImage[i] = (int) ( (unsigned char)(0.21* r + 0.71* g + 0.07* b) );
		i = i + stride;
	}
}

__global__ void float2uChar(float* deviceInputImage, int length, unsigned char* deviceUcharImage)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	
	while(i < length)
	{
		deviceUcharImage[i] = (unsigned char) (255 * deviceInputImage[i]);
		i = i + stride;
	}
}

__global__ void applyHistoEq(float* deviceCorrectColor, int length, unsigned char* deviceUcharImage, float* deviceOutputImageData)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	
	while(i < length)
	{
		deviceUcharImage[i] = (unsigned char) (deviceCorrectColor[deviceUcharImage[i]]);
		deviceOutputImageData[i] = (float) ( (unsigned char) deviceUcharImage[i] / 255.0) ;
		i = i + stride;
	}
}

float clamp(float x)
{
	return (float) (min(max((x), 0.0), 255.0));
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;
	unsigned char * hostUcharImage;
    int * grayImage;
	//int * hostGrayImage;
	
    int length;
    int i;
	//int j;
	
    //@@ Insert more code here
	
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
	
	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
	
	
	//@@ Host Memory allocation
	length = imageWidth*imageHeight*imageChannels;
    hostUcharImage = (unsigned char*)malloc(length*sizeof(unsigned char));
    grayImage = (int*)malloc(length*sizeof(int));
	//hostGrayImage = (int*)malloc(length*sizeof(int));
    

	//@@ Cast image from float to unsigned char
	//for(i=0;i<length;i++){
	//	hostUcharImage[i] = (unsigned char) (255 * hostInputImageData[i]);
	//}
	
	//@@ Cast image from float to unsigned char CUDA Kernel
	float * deviceInputImage;
	unsigned char * deviceUcharImage;
	
	dim3 dimGrid1(TILE_WIDTH, 1);
    dim3 dimBlock1(TILE_WIDTH*TILE_WIDTH, 1, 1);
	
	cudaMalloc((void**) &deviceInputImage, length * sizeof(float));
	cudaMalloc((void**) &deviceUcharImage, length * sizeof(unsigned char));
	
	cudaMemcpy(deviceInputImage, hostInputImageData, length * sizeof(float), cudaMemcpyHostToDevice);

	float2uChar<<<dimGrid1, dimBlock1>>>(deviceInputImage, length, deviceUcharImage);
	cudaDeviceSynchronize(); 
		
	cudaMemcpy(&hostUcharImage[0], deviceUcharImage, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
/*	 
	//@@ Convert image from RGB to GrayScale
	for(i=0;i<imageHeight;i++)
	{
		for(j=0;j<imageWidth;j++)
		{
			int idx = i*imageWidth + j;
			unsigned char r = hostUcharImage[3*idx];
			unsigned char g = hostUcharImage[3*idx + 1];
			unsigned char b = hostUcharImage[3*idx + 2];
			
			hostGrayImage[idx] = idx;
			//hostGrayImage[idx] = (int) ( (unsigned char)(0.21* r + 0.71* g + 0.07* b) );
		}
	}
*/
	
	//@@ Convert image from RGB to GrayScale CUDA Kernel
	//unsigned char * deviceUcharImage;
	int * deviceGrayImage;
	
	dim3 dimGrid2(TILE_WIDTH, 1);
    dim3 dimBlock2(TILE_WIDTH*TILE_WIDTH, 1, 1);
	
	//cudaMalloc((void**) &deviceUcharImage, length * sizeof(unsigned char));
	cudaMalloc((void**) &deviceGrayImage, length * sizeof(int));
	
	cudaMemcpy(deviceUcharImage, hostUcharImage, length * sizeof(unsigned char), cudaMemcpyHostToDevice);

	rgb2gray<<<dimGrid2, dimBlock2>>>(deviceUcharImage, length, deviceGrayImage);
	cudaDeviceSynchronize(); 
		
	cudaMemcpy(&grayImage[0], deviceGrayImage, length * sizeof(int), cudaMemcpyDeviceToHost);

	//for(i=0;i<length;i++)
	//{
	//	if(grayImage[i] != hostGrayImage[i])
	//		std::cout << "(ok, wrong): " << "(" << hostGrayImage[i] << "," << grayImage[i] << ")." << std::endl;
	//}

	
	//int count = 0;
	//for(i=0;i<length;i++)
	//{
	//	if(grayImage[i] == 0)
	//		count++;
	//}
	//std::cout << "zero-count: " << count << std::endl;

	int* histogram;
	histogram = (int*)malloc(HISTOGRAM_LENGTH * sizeof(int));
	 
	for(i=0;i<256;i++)
	{
		histogram[i] = 0;
	}
	
/*	
	//@@ Compute the histogram grayImage
	
	for(i=0;i<imageHeight;i++)
	{
		for(j=0;j<imageWidth;j++)
		{
			int idx = i*imageWidth + j;
			histogram[grayImage[idx]]++;
		}
	}
*/	
	
		
	//@@ Cuda variables and memory allocation for histogram calculation
	int* deviceBuffer;
	int* deviceHisto;
	
	dim3 dimGrid3(TILE_WIDTH, 1);
    dim3 dimBlock3(TILE_WIDTH*TILE_WIDTH, 1, 1);

	
	//std::cout << "length: " << length << std::endl;
	//std::cout << "imageWidth: " << imageWidth << std::endl;
	//std::cout << "imageHeight: " << imageHeight << std::endl;
	
	//std::cout << "cuda-dimBlock: (" << dimBlock.x << "," << dimBlock.y << ")." << std::endl;
	//std::cout << "cuda-dimGrid: (" << dimGrid.x << "," << dimGrid.y << ")." << std::endl;
	
	cudaMalloc((void **) &deviceBuffer, length * sizeof(int));
	cudaMalloc((void**) &deviceHisto, 256 * sizeof(int));
	
	//@@ Compute the histogram grayImage CUDA Kernel
	cudaMemcpy(deviceHisto, histogram, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBuffer, grayImage, length * sizeof(int), cudaMemcpyHostToDevice);
	
	histo_kernel<<<dimGrid3, dimBlock3>>>(deviceBuffer, length, deviceHisto);
	cudaDeviceSynchronize(); 
	
	cudaMemcpy(&histogram[0], deviceHisto, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	
	//for(i=0;i<256;i++)
	//{
	//	std::cout << "histogram: " << histogram[i] << std::endl;	
	//}
	
	//std::cout << std::endl;
	 
	//@@ Compute cumulative Distribution of histogram
	float* cdf;
	cdf = (float*)malloc(256*sizeof(float));
	cdf[0] = ( (float) histogram[0] ) / (imageWidth*imageHeight);
	 
	for(i=1;i<256;i++)
	{
		cdf[i] = cdf[i-1] + ( ((float) histogram[i]) / (imageWidth*imageHeight) );
	}
	
	 
	//@@ Compute the minimum value of the CDF
	float cdfMin = cdf[0];
	
	//@@ Define the histogram equalization function
	float* correct_color;
	correct_color = (float*)malloc(256*sizeof(float));
	 
	for(i=0;i<256;i++)
	{
		float tmp =  ( ((float) (255*(cdf[i]-cdfMin) / (1-cdfMin))) );
		correct_color[i] = clamp(tmp);
	}
	
	
	//@@ Apply the histogram equalization function
	//for(i=0;i<length;i++)
	//{
	//	hostUcharImage[i] = (unsigned char) (correct_color[hostUcharImage[i]]);
	//}
	
	//for(i=0;i<length;i++)
	//{
	//	hostOutputImageData[i] = (float) ( (unsigned char) hostUcharImage[i] / 255.0) ;
	//}
	
	//@@ Apply the histogram equalization function CUDA Kernel
	float * deviceCorrectColor;
	float * deviceOutputImageData;
	cudaMalloc((void**) &deviceCorrectColor, 256 * sizeof(float));
	cudaMalloc((void**) &deviceOutputImageData, length * sizeof(float));
	
	dim3 dimGrid4(TILE_WIDTH, 1);
    dim3 dimBlock4(TILE_WIDTH*TILE_WIDTH, 1, 1);
	
	cudaMemcpy(deviceUcharImage, hostUcharImage, length * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceCorrectColor, correct_color, 256 * sizeof(float), cudaMemcpyHostToDevice);

	applyHistoEq<<<dimGrid4, dimBlock4>>>(deviceCorrectColor,length, deviceUcharImage, deviceOutputImageData);
	cudaDeviceSynchronize(); 
		
	cudaMemcpy(&hostOutputImageData[0], deviceOutputImageData, length * sizeof(float), cudaMemcpyDeviceToHost);
	

    wbSolution(args, outputImage);


    return 0;
}

