/* Bench code on the compression rate */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"
#include <cuda_runtime.h>

#define USAGE "Use %s \n"\
              "OPTION :\n"\
              "\t\t-x <int> dimension of the cube\n"\
              "\t\t-v <double> value used as parameter for compression\n"

int main(int argc, char* argv[])
{
  double val = 16.;
  int x = 100;
  int nx, ny, nz;
  zfp_type type;     /* array scalar type */
  zfp_field* field;  /* array meta data */
  zfp_stream* zfp;   /* compressed stream */
  size_t bufsize;    /* byte size of compressed buffer */
  bitstream* stream; /* bit stream to write to or read from */
  size_t zfpsize;    /* byte size of compressed stream */
  cudaError_t cuerr;

  float *d_array = NULL;
  void *d_buffer  = NULL;
  float *array   = NULL;
  float *array_original = NULL;

  if (argc > 1){
    for (int i = 1; i < argc; i++){
      if (!strcmp(argv[i], "-x"))
        x = atoi(argv[++i]);
      else if (!strcmp(argv[i], "-v"))
        val = atof(argv[++i]);
      else if (!strcmp(argv[i], "-h")){
        printf(USAGE, argv[0]);
        return 0;
      }
    }
  }

  nx=ny=nz=x;

  /* initialize array to be compressed */
  array = malloc(nx * ny * nz * sizeof(float));
  array_original = malloc(nx * ny * nz * sizeof(float));
  int i, j, k;
  for (k = 0; k < nz; k++)
    for (j = 0; j < ny; j++)
      for (i = 0; i < nx; i++) {
        double x = 2.0 * i / nx;
        double y = 2.0 * j / ny;
        double z = 2.0 * k / nz;
        array[i + nx * (j + ny * k)] = exp(-(x * x + y * y + z * z));
      }
  memcpy(array_original, array, nx * ny * nz * sizeof(float));


  /* Copy data to device */
  size_t arraysize = nx * ny * nz * sizeof(float);
  printf("[GPU] Rate compression (param %f) on data of size %dx%dx%d => %ld\n",
          val, nx, ny, nz, arraysize);

  cuerr = cudaMalloc((void**)&d_array, arraysize);
  if (cuerr != cudaSuccess){
    fprintf(stderr, "Unable to allocate d_Array on the GPU\n");
    return 1;
  }

  cuerr = cudaMemcpy(d_array, array, arraysize, cudaMemcpyHostToDevice);
  memset(array, 0, arraysize);

  /* allocate meta data for the 3D array a[nz][ny][nx] */
  type = zfp_type_float;
  field = zfp_field_3d(d_array, type, nx, ny, nz);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(NULL);

  /* set compression mode and parameter*/
  zfp_stream_set_rate(zfp, val, type, 3, 0);
  zfp_stream_set_execution(zfp, zfp_exec_cuda);

  /* allocate buffer for compressed data */
  bufsize = zfp_stream_maximum_size(zfp, field);
  cuerr = cudaMalloc(&d_buffer, bufsize);
  if (cuerr != cudaSuccess){
    fprintf(stderr, "Unable to allocate d_buffer on the GPU\n");
    cudaFree(d_array);
    return 1;
  }

  /* associate bit stream with allocated buffer */
  stream = stream_open(d_buffer, bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);


  /* compress array */
  zfpsize = zfp_compress(zfp, field);
  if (!zfpsize) {
    fprintf(stderr, "compression failed\n");
    cudaFree(d_buffer);
    cudaFree(d_array);
    return 1;
  }
  printf("%zu\n", zfpsize);
  
  zfp_stream_rewind(zfp);

  /* uncompress array*/
  zfpsize = zfp_decompress(zfp, field);
  if (!zfpsize) {
    fprintf(stderr, "decompression failed\n");
    cudaFree(d_buffer);
    cudaFree(d_array);
    return 1;
  }
  
  cuerr = cudaMemcpy(array, d_array, arraysize, cudaMemcpyDeviceToHost);
  
  double err = 0.;
  for (i = 0;i < nx * ny * nz; ++i){
    if (array[i] > array_original[i]){
      err += array[i] - array_original[i];
    } else{
      err += array_original[i] - array[i];
    }
  }
  printf("%.2f\n", err);

  /* clean up */
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  cudaFree(d_buffer);
  cudaFree(d_array);
  free(array);
  free(array_original);
}
