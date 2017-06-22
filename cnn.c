#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "cnn.h"
#include <CL/cl.h>
#include "kernel_cl.h"
struct timeval t1, t2;
	

inline void checkErr(cl_int err, const char * name) {
   if (err != CL_SUCCESS) {
      fprintf(stderr, "ERROR: %s (%d)\n", name, err);
      exit(EXIT_FAILURE);
   }
}



// Sequential CNN implementation
void conv(float Cout[NUM][OUTIMROW][OUTIMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
	static float C[NUM][IMROW][IMROW];

	for(int i = 0; i < NUM; i++) {
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w++)
				C[i][h][w] = bias[i];
		}
	}






// Use this to check the output of each API call
   cl_int status;  

   // Retrieve the number of platforms
   cl_uint numPlatforms = 0;
   status = clGetPlatformIDs(0, NULL, &numPlatforms);
   checkErr(status, "Retrieve the number of platforms");

   // Allocate enough space for each platform
   cl_platform_id *platforms = NULL;
   platforms = (cl_platform_id*)malloc(
         numPlatforms * sizeof(cl_platform_id));

   // Fill in the platforms
   status = clGetPlatformIDs(numPlatforms, platforms, NULL);
   checkErr(status, "Fill in the platforms");

gettimeofday(&t1, NULL);

   // Find CPU
   int platform_index = -1;
   for (int i = 0; i < numPlatforms; i++){
      char vendor[128];
      clGetPlatformInfo (platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
      char vendorF[7];
      memcpy((void*)vendorF, (void*)vendor, 6);
      vendorF[6] = '\0';
      fprintf(stderr, "%s\n", vendorF);
      if (strcmp(vendorF, "NVIDIA") == 0)
      {
         platform_index = i;
         break;
      }
   }
   if (platform_index == -1){
      printf("GPU platform not found!\n");
      exit(1);
   }

   // Retrieve the number of devices
   cl_uint numDevices = 0;
   status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, 
         NULL, &numDevices);
   checkErr(status, "Retrieve the number of devices");
   printf("#devices: %d, status %d\n", numDevices, status);

   // Allocate enough space for each device
   cl_device_id *devices;
   devices = (cl_device_id*)malloc(
         numDevices * sizeof(cl_device_id));

   // Fill in the devices 
   status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL,        
         numDevices, devices, NULL);
   checkErr(status, "Fill in the devices");

   // Create a context and associate it with the devices
   cl_context context;
   context = clCreateContext(NULL, numDevices, devices, NULL, 
         NULL, &status);
 

   // Create a command queue and associate it with the device 
   cl_command_queue cmdQueue;
   cmdQueue = clCreateCommandQueue(context, devices[0], 0, 
         &status);




/////////////////////////////////////////////////////////////////////
   cl_mem bufCin;
   bufCin = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM*INIMROW*INIMROW,                       
         NULL, &status);

   
   cl_mem bufWeight;
   bufWeight = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM*NUM*KERNEL*KERNEL,                        
         NULL, &status);

  
   cl_mem bufCzj;
   bufCzj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*NUM*IMROW*IMROW,
         NULL, &status); 

   status = clEnqueueWriteBuffer(cmdQueue, bufCin, CL_FALSE, 
         0, sizeof(float)*NUM*INIMROW*INIMROW, Cin, 0, NULL, NULL);
   checkErr(status, "Write buffer Cin");

   
   status = clEnqueueWriteBuffer(cmdQueue, bufWeight, CL_FALSE, 
         0, sizeof(float)*NUM*NUM*KERNEL*KERNEL, weight, 0, NULL, NULL);
   checkErr(status, "Write buffer weight");

   status = clEnqueueWriteBuffer(cmdQueue, bufCzj, CL_FALSE, 
         0, sizeof(float)*NUM*IMROW*IMROW, C, 0, NULL, NULL);
   checkErr(status, "Write buffer C");

   // Create a program with source code
   cl_program program = clCreateProgramWithSource(context, 1, 
         (const char**)&kernel_cl, NULL, &status);


   ////////////////////////////////////////////////////////////////

   // Build (compile) the program for the device
   status = clBuildProgram(program, numDevices, devices, 
         NULL, NULL, NULL);
   if(status == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0,
            NULL, &log_size);
      char *log = (char*)malloc(log_size);
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
            log_size, log, NULL);
      fprintf(stderr, "%s\n", log);
      exit(1);
   }
   //		checkErr(status, "Build program");

   // Create the vector addition kernel
   cl_kernel kernel;
   kernel = clCreateKernel(program, "Kernel", &status);

   // Associate the input and output buffers with the kernel 
   status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufCin);
   checkErr(status, "Set Arg 0");
   status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufWeight);
   checkErr(status, "Set Arg 1");
   status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufCzj);
   checkErr(status, "Set Arg 2");

   // Define an index space (global work size) of work 
   // items for execution. A workgroup size (local work size) 
   // is not required, but can be used.
   size_t local[3];
local[0]=112;
local[1]=1;
local[2]=1;
   size_t global[3];
global[0]=112;
global[1]=28;
global[2]=256;

   // Execute the kernel for execution
   status = clEnqueueNDRangeKernel(cmdQueue, kernel, 3, NULL, 
         (size_t*)&global, (size_t*)&local, 0, NULL, NULL);
   checkErr(status, "Execute kernel");



   // Read the device output buffer to the host output array
   clEnqueueReadBuffer(cmdQueue, bufCzj, CL_TRUE, 0, 
     sizeof(float)*NUM*IMROW*IMROW, C, 0, NULL, NULL);


// Max pooling
	for (int i = 0; i < NUM; i++) {
		for (int h = 0; h < OUTIMROW; h++) {
			for (int w = 0; w < OUTIMROW; w++) {
				float local_max = C[i][2 * h][2 * w];
				local_max = fmax(local_max, C[i][2 * h + 1][2 * w]);
				local_max = fmax(local_max, C[i][2 * h + 1][2 * w + 1]);
				local_max = fmax(local_max, C[i][2 * h][2 * w + 1]);
				Cout[i][h][w] = local_max;
			}
		}
	}


}



int main(){
	static float Cout[NUM][OUTIMROW][OUTIMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	fprintf(stderr, "Start cnn computation\n");
	

	// --- Please add OpenCL setup code below ---
   
   // Run the sequential implementation for now. 
   // You should replace this with a call to your kernel
	conv(Cout, Cin, weight, bias);	

   // --- Timing stuff
	gettimeofday(&t2, NULL);
	float elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
	fprintf(stderr, "time(s): %f\n", elapsed_time);
	fprintf(stderr, "GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time / 1e9);

   // Please disable the error check before handing in your submission
   // Reminder: We will be measuring your performance externally! (using a unix time invocation)
	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	return 0;
}