#include <cstdio>
#include <cstdlib>
#include <vector>

// Parallel prefix sum (implementation taken from 11_scan.cpp)
template <class T>
std::vector<T> prefix_sum(std::vector<T> xs) {
  std::vector<T> copy(xs.size());
  auto N = xs.size();
  
  #pragma omp parallel
  for (auto j = 1; j < N; j <<= 1) {
#pragma omp for
    for (auto i = 0; i < N; i++) {
      copy[i] = xs[i];
    }
#pragma omp for
    for(auto i = j; i < N; i++) {
      xs[i] += copy[i-j];
    }
  }

  return xs;
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  // Further improvement: 
  // this loop could prob. also be done in parallel, for instance see:
  // https://stackoverflow.com/questions/29285110/parallel-incrementing-of-array-elements-with-openmp
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  
  auto offset = prefix_sum(bucket);
#pragma omp parallel for
  for (int i = 0; i < range; i++) {
    offset[i] -= bucket[i];
  }

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
