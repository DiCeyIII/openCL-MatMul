#include <CL/cl.h>
#include "constants.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>

// Define the size of the matrices



// Function to read a file into a string

void printMatrix(const float* A, int rows, int cols)
{

    for (int i{ 0 }; i < rows; ++i)
    {
        for (int j{ 0 }; j < cols; ++j)
        {
            std::cout << std::setprecision(2) << A[cols * i + j] << " ";
        }
        
        std::cout << "\n";
    }

    std::cout << "\n\n\n";

}
char* readFile(const char* filename)
{
    FILE* program_handle;
    char* program_buffer;
    size_t program_size;
    int err;
    //Read program file and place content into buffer
    program_handle = fopen(filename, "rb");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    return program_buffer;
}

std::string readFile(const std::string& filename) {
    std::fstream file(filename);

    std::string content{
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    };

    const char* kernelCharArray{ new char[content.size()] };
    kernelCharArray = content.c_str();

    return content.c_str();

}

int main() {
    cl_platform_id platformId = NULL;
    cl_device_id deviceId = NULL;
    cl_context context = NULL;
    cl_command_queue commandQueue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    clGetPlatformIDs(1, &platformId, NULL);
    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL);

    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, NULL);
    commandQueue = clCreateCommandQueue(context, deviceId, 0, NULL);

    float* A = new float[MATRIX_SIZE * MATRIX_SIZE];
    float* B = new float[MATRIX_SIZE * MATRIX_SIZE];
    float* C = new float[MATRIX_SIZE * MATRIX_SIZE];


    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    printMatrix(A, MATRIX_SIZE, MATRIX_SIZE);
    printMatrix(B, MATRIX_SIZE, MATRIX_SIZE);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, A, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, B, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);

   
    char* kernelSource{ readFile("matrix_multiply.cl") };

    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL);
  
    std::size_t logSize{};
    char* program_log{};

    clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    program_log = (char*)calloc(logSize + 1, sizeof(char));

    clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize + 1, program_log, NULL);

    kernel = clCreateKernel(program, "matrixMultiply", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    size_t globalWorkSize[] = { MATRIX_SIZE, MATRIX_SIZE };
    size_t localWorkSize[] = { WORK_SIZE, WORK_SIZE };
    clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, C, 0, NULL, NULL);
    
    printMatrix(C, MATRIX_SIZE, MATRIX_SIZE);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
