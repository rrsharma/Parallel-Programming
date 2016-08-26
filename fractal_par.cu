/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

// Lab 9 : Rahil Sharma (Edited original)
//compile : nvcc -o fractal_par fractal_par.cu -lpthread -lglut -lGL
//run : ./fractal_par
//(also)run : double click on the object file created after compilation 

#include "book.h"
#include "cpu_bitmap.h"

#define DIM 500

 struct cuComplex {
 float r;
 float i;
 __device__ cuComplex( float a, float b ) : r(a), i(b) {}
 __device__ float magnitude2( void ) {
 return r * r + i * i;
 }
 __device__ cuComplex operator*(const cuComplex& a) {
 return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
 }
 __device__ cuComplex operator+(const cuComplex& a) {
 return cuComplex(r+a.r, i+a.i);
 }
};


__device__ int julia( int x, int y ) {
 const float scale = 1.5;
 float jx = scale * (float)(DIM/2 - x)/(DIM/2);
 float jy = scale * (float)(DIM/2 - y)/(DIM/2);
 cuComplex c(-0.8, 0.156);
 cuComplex a(jx, jy);
 int i = 0;
 for (i=0; i<200; i++) {
 a = a * a + c;
 if (a.magnitude2() > 1000)
 return 0;
 }
 return 1;
}

__global__ void kernel( unsigned char *ptr ) {
 int x = blockIdx.x;
 int y = blockIdx.y;
 int offset = x + y * gridDim.x;
 int juliaValue = julia( x, y );
    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = 255 *juliaValue;
    ptr[offset*4 + 2] = ((x*y)%255);
    ptr[offset*4 + 3] = 0;
}
int main( void ) {
 CPUBitmap bitmap( DIM, DIM );
 unsigned char *dev_bitmap;
 HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, 
 bitmap.image_size() ) );
 dim3 grid(DIM,DIM);
 kernel<<<grid,1>>>( dev_bitmap );
 HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost ) );
 bitmap.display_and_exit(NULL);
 HANDLE_ERROR( cudaFree( dev_bitmap ) );
}
