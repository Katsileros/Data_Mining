#include <wb.h> 

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float* restrict hostInput1;
  float* restrict hostInput2;
  float* restrict hostOutput;
	
  int i=0;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

/*
  wbTime_start(GPU, "Sequential FOR execution");

  for(i=0; i<inputLength; i++) {
	  hostOutput[i] = hostInput1[i] + hostInput2[i];
  }
  
  wbTime_stop(GPU, "Sequential FOR execution");
*/

  wbTime_start(GPU, "Parallel FOR execution");
  
  int n = inputLength;
	
  // sum component wise and save result into vector c
  #pragma acc parallel loop copyin(hostInput1[0:inputLength],hostInput2[0:inputLength]), copyout(hostOutput[0:inputLength])
  for(i=0; i<n; i++) {
	  hostOutput[i] = hostInput1[i] + hostInput2[i];
  }
  
  wbTime_stop(GPU, "Parallel FOR execution");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
