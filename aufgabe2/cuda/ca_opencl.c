/*
 * simulate a cellular automaton with periodic boundaries (torus-like)
 * serial version
 *
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

/* --------------------- CA simulation -------------------------------- */

/* annealing rule from ChoDro96 page 34
 * the table is used to map the number of nonzero
 * states in the neighborhood to the new state
 */
static const cell_state_t anneal[10] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 1};


int main(int argc, char** argv)
{
        cl_int err_num;
        cl_platform_id platform = NULL;
        cl_uint num_devices;
        cl_device_id *devices = NULL;
        cl_device_id gpu_dev;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;

        clGetPlatformIDs(1, &platform, NULL);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

        MALLOC_ERROR_CHECK(devices = (cl_device_id *) calloc(num_devices, sizeof(cl_device_id)));
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

        char device_name[1024];
        for(int i = 0; i < num_devices; i++)
        {
            gpu_dev = devices[i];
            clGetDeviceInfo(gpu_dev, CL_DEVICE_NAME, sizeof(device_name), (void *) device_name, NULL);

            if (strstr(device_name, GPU) != NULL)
            {
                break;
            }
        }

        //printf("device %d: %s\n", gpu_dev, device_name);
        context = clCreateContext(0, 1, &gpu_dev, NULL, NULL, NULL);

        queue = clCreateCommandQueue(context, gpu_dev, 0, NULL);


        free(devices);

	return EXIT_SUCCESS;
}
