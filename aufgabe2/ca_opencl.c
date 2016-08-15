/*
 * simulate a cellular automaton with periodic boundaries (torus-like)
 * OpenCL version
 *
 * (c) 2016 Felix Kubicek (OpenCL port)
 * (c) 2016 Steffen Christgau (C99 port, modularization)
 * (c) 1996,1997 Peter Sanders, Ingo Boesnach (original source)
 *
 * command line arguments:
 * #1: Number of lines
 * #2: Number of iterations to be simulated
 *
 */

#include<CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ca_common.h"

#define GPU "Tesla K40c"
#define WORK_DIM 2 
#define WORK_GROUP_SIZE_X 32
#define WORK_GROUP_SIZE_Y 20

#define CL_TYPEDEF_CELL_T typedef unsigned char cl_cell_state_t
#define CL_TYPEDEF_LINE_T typedef cl_cell_state_t cl_line_t[XSIZE]

CL_TYPEDEF_CELL_T;
CL_TYPEDEF_LINE_T;   
/* --------------------- CA simulation -------------------------------- */

/* annealing rule from ChoDro96 page 34
 * the table is used to map the number of nonzero
 * states in the neighborhood to the new state
 */
static const cl_cell_state_t anneal[10] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 1};

#define CL_ERROR_CHECK(x)\
        do {\
            if ((x) != CL_SUCCESS)\
                {fprintf(stderr ,"OpenCL error (%d): %s: %u\n", x, __FILE__, __LINE__); exit (EXIT_FAILURE); }\
        } while (0)

const char* source = STR(CL_TYPEDEF_CELL_T;
                         CL_TYPEDEF_LINE_T;   
                         __kernel void simulate(__global cl_line_t *from, __global cl_line_t *to, const int lines, __constant cl_cell_state_t anneal[10])
                         {

                            int x = get_global_id(0);
                            int y = get_global_id(1);

                            int x_prev = ((x - 1) + XSIZE) % XSIZE;
                            int y_prev = ((y - 1) + lines) % lines;
                            int x_next = (x + 1) % XSIZE; 
                            int y_next = (y + 1) % lines;
                                             
                            to[y][x] = transition_cuda(from, x_prev, y_prev, x, y, x_next, y_next);
                         });

int main(int argc, char** argv)
{    
        int lines, its;
        line_t *verify_field;
        cl_line_t *from, *to;
        cl_mem from_d, to_d, anneal_d;

        cl_int err_num;
        cl_platform_id platform = NULL;
        cl_uint num_devices;
        cl_device_id *devices = NULL;
        cl_device_id gpu_dev;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;
        cl_int error;

        ca_init(argc, argv, &lines, &its);

        CL_ERROR_CHECK(error = clGetPlatformIDs(1, &platform, NULL));
        CL_ERROR_CHECK(error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices));

        MALLOC_ERROR_CHECK(devices = (cl_device_id *) calloc(num_devices, sizeof(cl_device_id)));
        CL_ERROR_CHECK(error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL));

        char device_name[1024];
        for(int i = 0; i < num_devices; i++)
        {
            gpu_dev = devices[i];
            CL_ERROR_CHECK(error = clGetDeviceInfo(gpu_dev, CL_DEVICE_NAME, sizeof(device_name), (void *) device_name, NULL));

            if (strstr(device_name, GPU) != NULL)
            {
                break;
            }
        }
        //printf("device %d: %s\n", gpu_dev, device_name);
        //printf("\n\n%s\n\n", source);
        
        context = clCreateContext(0, 1, &gpu_dev, NULL, NULL, &error);
        CL_ERROR_CHECK(error);

        queue = clCreateCommandQueue(context, gpu_dev, 0, &error);
        CL_ERROR_CHECK(error);
        
        MALLOC_ERROR_CHECK(from = (cl_line_t *) calloc(lines, sizeof(cl_line_t)));
	MALLOC_ERROR_CHECK(to = (cl_line_t *) calloc(lines, sizeof(cl_line_t)));
	MALLOC_ERROR_CHECK(verify_field = (line_t *) calloc((lines + 2), sizeof(line_t)));

        size_t buffer_size = lines * sizeof(cl_line_t);

        from_d = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &error); 
        CL_ERROR_CHECK(error);
        to_d = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &error); 
        CL_ERROR_CHECK(error);
        anneal_d = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(anneal), NULL, &error); 
        CL_ERROR_CHECK(error);

	ca_init_config_cuda(from, lines, 0);

        CL_ERROR_CHECK(error = clEnqueueWriteBuffer(queue, from_d, CL_TRUE, 0, buffer_size, from, 0, NULL, NULL));
        CL_ERROR_CHECK(error = clEnqueueWriteBuffer(queue, to_d, CL_TRUE, 0, buffer_size, to, 0, NULL, NULL));
        CL_ERROR_CHECK(error = clEnqueueWriteBuffer(queue, anneal_d, CL_TRUE, 0, sizeof(anneal), anneal, 0, NULL, NULL));

        program = clCreateProgramWithSource(context, 1, &source, NULL, &error);
        CL_ERROR_CHECK(error);

        CL_ERROR_CHECK(error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL)); 

        kernel = clCreateKernel(program, "simulate", &error);
        CL_ERROR_CHECK(error);

        const size_t local_work_size[WORK_DIM] = {WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y};
        const size_t global_work_size[WORK_DIM] = {XSIZE, lines};

        cl_mem *from_d_ptr = &from_d;
        cl_mem *to_d_ptr = &to_d;
        cl_mem *temp; 
        
        CL_ERROR_CHECK(error = clSetKernelArg(kernel, 2, sizeof(int), &lines));
        CL_ERROR_CHECK(error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &anneal_d));

        TIME_GET(sim_start);
        for (int i = 0; i < its; i++) 
        {        
            CL_ERROR_CHECK(error = clSetKernelArg(kernel, 0, sizeof(cl_mem), from_d_ptr));
            CL_ERROR_CHECK(error = clSetKernelArg(kernel, 1, sizeof(cl_mem), to_d_ptr));
            CL_ERROR_CHECK(error = clEnqueueNDRangeKernel(queue, kernel, WORK_DIM, NULL, global_work_size, local_work_size, 0, NULL, NULL)); 
        
            temp = from_d_ptr;
	    from_d_ptr = to_d_ptr;
	    to_d_ptr = temp;
        }   
            
        CL_ERROR_CHECK(error = clFinish(queue));
	TIME_GET(sim_stop);

        CL_ERROR_CHECK(error = clEnqueueReadBuffer(queue, *from_d_ptr, CL_TRUE, 0, buffer_size, from, 0, NULL, NULL));

        for(int y = 1; y <= lines; y++)
        {
            for(int x = 1; x<= XSIZE; x++)
            {
                verify_field[y][x] = from[y-1][x-1];
            }
        }
        //print_field(verify_field, lines);

	ca_hash_and_report(verify_field + 1, lines, TIME_DIFF(sim_start, sim_stop));

        clReleaseMemObject(from_d);
        clReleaseMemObject(to_d);
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

	free(from);
	free(to);
        free(verify_field);
        free(devices);

	return EXIT_SUCCESS;
}
