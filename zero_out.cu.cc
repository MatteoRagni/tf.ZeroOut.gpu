#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

__global__ void ZeroOutKernel(const Tensor* T_in, const int n, Tensor* T_out) {
  for (int i = 1; i < N; i++) t_out(i) = 0;
  t_out(0) = t_in(0);
}

void ZeroOutKernelLauncher(const Tensor* t_in, const int n, Tensor* t_out) {
  ZeroOutKernel<<<32, 256>>>(t_in, n, t_out);
}

#endif
