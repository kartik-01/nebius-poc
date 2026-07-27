// Bounded CUDA smoke: init every visible device, allocate a small buffer, run a
// deterministic saxpy, and check the result. Elapsed time is reported for
// diagnostics only — never compared against an absolute threshold here.
//
// On success, prints one JSON object to stdout and exits 0. On failure, prints
// a reason to stderr and exits nonzero.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kElements = 1 << 20;  // 1 Mi floats per device
constexpr float kAlpha = 2.0f;
constexpr int kMaxDevices = 16;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                   cudaGetErrorString(err__), __FILE__, __LINE__);             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void saxpy(int n, float alpha, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = alpha * x[i] + y[i];
  }
}

struct DeviceResult {
  int device;
  char name[256];
  float elapsed_ms;
  int mismatches;
};

int smoke_device(int device, DeviceResult* out) {
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::vector<float> host_x(kElements), host_y(kElements), host_out(kElements);
  for (int i = 0; i < kElements; ++i) {
    host_x[i] = static_cast<float>(i % 97) * 0.25f;
    host_y[i] = static_cast<float>(i % 53) * 0.5f;
  }

  float* device_x = nullptr;
  float* device_y = nullptr;
  CUDA_CHECK(cudaMalloc(&device_x, kElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_y, kElements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_x, host_x.data(), kElements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_y, host_y.data(), kElements * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  int threads = 256;
  int blocks = (kElements + threads - 1) / threads;
  saxpy<<<blocks, threads>>>(kElements, kAlpha, device_x, device_y);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(host_out.data(), device_y, kElements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int mismatches = 0;
  for (int i = 0; i < kElements; ++i) {
    float expected = kAlpha * host_x[i] + host_y[i];
    if (std::fabs(host_out[i] - expected) > 1e-4f) {
      ++mismatches;
    }
  }

  cudaFree(device_x);
  cudaFree(device_y);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  out->device = device;
  std::snprintf(out->name, sizeof(out->name), "%s", prop.name);
  out->elapsed_ms = elapsed_ms;
  out->mismatches = mismatches;

  if (mismatches != 0) {
    std::fprintf(stderr, "device %d (%s): %d mismatches\n", device, prop.name,
                 mismatches);
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return 1;
  }
  if (count > kMaxDevices) {
    std::fprintf(stderr, "refusing to smoke %d devices (max %d)\n", count,
                 kMaxDevices);
    return 1;
  }

  DeviceResult results[kMaxDevices];
  for (int device = 0; device < count; ++device) {
    if (smoke_device(device, &results[device]) != 0) {
      return 1;
    }
  }

  std::printf("{\"status\":\"PASS\",\"device_count\":%d,\"elements\":%d,\"devices\":[",
              count, kElements);
  for (int device = 0; device < count; ++device) {
    if (device > 0) {
      std::printf(",");
    }
    std::printf(
        "{\"device\":%d,\"name\":\"%s\",\"elapsed_ms\":%.3f,\"mismatches\":%d}",
        results[device].device, results[device].name, results[device].elapsed_ms,
        results[device].mismatches);
  }
  std::printf("]}\n");
  return 0;
}
