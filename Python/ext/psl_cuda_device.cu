// This file is part of PSL-Python.
// Copyright (c) 2021, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "unified_camera.cuh"
#include "warping.cuh"
#include "depth.cuh"

#include "psl_cuda_utils.hpp"

#include <torch/torch.h>

#define BLOCK_SIZE_RAYS 128
#define BLOCK_SIZE_PLANES 8

namespace UnifiedCameraModel
{
#include "psl_cuda_device_core.cu"
torch::Tensor getWarpedImageTensorCUDA(int ref, torch::Tensor image, torch::Tensor Ks, torch::Tensor Rs, torch::Tensor Ts, torch::Tensor rays, torch::Tensor Ps)
{
    return getWarpedImageTensorCUDA<5>(ref, image, Ks, Rs, Ts, rays, Ps);
}
torch::Tensor getRayTensorCUDA(int ref, torch::Tensor image, torch::Tensor Ks)
{
    return getRayTensorCUDA<5>(ref, image, Ks);
}
};

__global__ void getDepthCUDAKernel(
	float* __restrict__ output,
	const float3* __restrict__ rays,
	const long* __restrict__ indices,
	const float4* __restrict__ planes,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	const int index = indexX + indexY * width;
	const int indexP = indices[index];
	float4 plane = make_float4(planes[indexP].x, planes[indexP].y, planes[indexP].z, planes[indexP].w);
	output[index] = getDepthFromRayAndPlane(rays[index], plane);
}

torch::Tensor getDepthTensorCUDA(torch::Tensor rays, torch::Tensor indices, torch::Tensor Ps)
{
	const int width = rays.size(1), height = rays.size(0);
	auto DepthOptions =
	torch::TensorOptions()
		.dtype(torch::kFloat)
		.device(torch::kCUDA, 0);
	auto depths = torch::empty({height, width}, DepthOptions);

	float *p_depths = depths.data_ptr<float>();
	const float3 *p_rays = reinterpret_cast<float3 *>(rays.data_ptr<float>());
	const long *p_indices = indices.data_ptr<long>();
	const float4 *p_planes = reinterpret_cast<float4 *>(Ps.data_ptr<float>());
	{
		const dim3 threads(32, 32);
		const dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
		getDepthCUDAKernel<<<blocks, threads>>>(
                p_depths,
				p_rays,
                p_indices,
                p_planes,
				height, width);
	}

	return depths;
}
