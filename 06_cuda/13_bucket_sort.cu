#include <cstdio>
#include <cstdlib>
#include <cstring>

// I'm using the same approach as I did for the Open MPI homework. That is, 
// 1. Count the frequency (fill the buckets)
// 2. Use the provided prefix sum on the buckets to calculate the offsets 
// 3. Remove bucket[i] from each offset since we want an exclusive prefix sum
//    Note: a *much* nicer approach would of course be to use an exclusive scan from the start
//          for instance this one found in Nvida's 'GPU Gems 3': 
//          https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
//          but I want to keep things simple so I understand what I'm doing :)
// 4. Finally, go through the list and sort the list in parallel using the offsets

// Fill the buckets by counting the frequency of each value in the array 'xs'
__global__ void frequency(int *keys, int *buckets, int len) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= len) {
    return; // we have enough threads already
  }

  atomicAdd(&buckets[keys[i]], 1);
}

// Prefix sum taken from the '08_scan.cu' example
__global__ void prefix_sum(int *a, int *b, int len) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j=1; j < len; j<<=1) {
    b[i] = a[i];
    __syncthreads();
    if (i>=j) {
      a[i] += b[i-j];
    }
    __syncthreads();
  }
}

__global__ void make_exclusive(int *xs, int *ys, int len) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= len) {
    return; // we have enough threads already
  }

  xs[i] -= ys[i];
}

// Use the offsets to sort the array 'keys'
__global__ void sort(int *keys, int *buckets, int *offsets, int len) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= len) {
    return; // we have enough threads already
  }

  // Go through 'keys' and fill it with the correct value using
  // 'j' as a cursor (starting at the correct offset)
  for (auto j = offsets[i]; buckets[i] > 0; buckets[i]--) {
      keys[j++] = i;
  }
}

int main() {
  const int n = 10;
  const int range = 5;
  
  int *keys, *buckets, *offsets;
  cudaMallocManaged(&keys,    n*sizeof(int));
  cudaMallocManaged(&buckets, range*sizeof(int));
  cudaMallocManaged(&offsets, range*sizeof(int));

  // Fill the the 'keys' array with random values within the range
  for (int i=0; i<n; i++) {
    keys[i] = rand() % range;
    printf("%d ",keys[i]);
  }
  printf("\n");

  // Set all elements in the buckets array to zero at the start
  for (int i=0; i<range; i++) {
    buckets[i] = 0;
  }

  cudaDeviceSynchronize();
  // Now, let's actualy start the bucket sort

  frequency<<<1,n>>>(keys, buckets, n); // TODO nr threads
  cudaDeviceSynchronize();

  int *temp;
  cudaMalloc(&temp, range * sizeof(int));
  std::memcpy(offsets, buckets, range * sizeof(int));
  prefix_sum<<<1, range>>>(offsets, temp, range);
  cudaDeviceSynchronize();

  make_exclusive<<<1, range>>>(offsets, buckets, range);
  cudaDeviceSynchronize();

  sort<<<1, range>>>(keys, buckets, offsets, range);
  cudaDeviceSynchronize();

  // Print the sorted array
  for (int i=0; i<n; i++) {
    printf("%d ",keys[i]);
  }
  printf("\n");
}
