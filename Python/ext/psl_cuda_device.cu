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

#include "psl_cuda_utils.hpp"

#include <torch/torch.h>

#include <vector>

__global__ void getRaysCUDAKernel(
	float3* __restrict__ output,
	Intrinsics kr,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	const int indexOutput = indexX + indexY * width;
	float3 xyz = unproject(make_float2(indexX, indexY), kr);
	output[indexOutput] =  xyz;
}

#define BLOCK_SIZE_RAYS 128
#define BLOCK_SIZE_PLANES 8

__global__ void getWarpedImageTensorCUDAKernel(
	unsigned char* __restrict__ output,
	cudaTextureObject_t tex,
	const float3* __restrict__ rays,
	Intrinsics ko,
	float3 r1,
	float3 r2,
	float3 r3,
	float3 t,
	const float* __restrict__ P,
	int nPlanes,
	int height,
	int width)
{
	const int indexXY = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexX = indexXY % width, indexY = indexXY / width;
	const int indexP = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height) || (indexP >= nPlanes))
	{
		return;
	}

    __shared__ float3 rays_shared[BLOCK_SIZE_RAYS];
    __shared__ float4 planes_shared[BLOCK_SIZE_PLANES];
	if (threadIdx.y == 0)
	{
		rays_shared[threadIdx.x] = rays[indexXY];
	}
	if (threadIdx.x == 0)
	{
		planes_shared[threadIdx.y].x = P[4 * indexP];
		planes_shared[threadIdx.y].y = P[4 * indexP + 1];
		planes_shared[threadIdx.y].z = P[4 * indexP + 2];
		planes_shared[threadIdx.y].w = P[4 * indexP + 3];
	}
	__syncthreads();

	// coordinate transform
	float3 n = make_float3(planes_shared[threadIdx.y].x, planes_shared[threadIdx.y].y, planes_shared[threadIdx.y].z);
	float d = planes_shared[threadIdx.y].w;
	float R[3][3];
	R[0][0] = r1.x;
	R[0][1] = r1.y;
	R[0][2] = r1.z;
	R[1][0] = r2.x;
	R[1][1] = r2.y;
	R[1][2] = r2.z;
	R[2][0] = r3.x;
	R[2][1] = r3.y;
	R[2][2] = r3.z;
	float H[3][3];
	computeHMatrix(R, t, n, d, H);
	float3 Hray = apply3x3Transformation(H, rays_shared[threadIdx.x]);
	float2 uvo = project(Hray, ko);
	const int wh = width * height;
	const int indexOutput = indexX + indexY * width + indexP * wh;
	output[indexOutput] =  static_cast<unsigned char>(255 * __saturatef(tex2D<float>(tex, uvo.x / width, uvo.y / height)));
	__syncthreads();
}

torch::Tensor getWarpedImageTensorCUDA(int ref, torch::Tensor images, torch::Tensor Ks, torch::Tensor xis, torch::Tensor Rs, torch::Tensor Ts, torch::Tensor rays, torch::Tensor Ps)
{
	const int nImages = images.size(0);
	const int nPlanes = Ps.size(0);
	const int width = images.size(2), height = images.size(1);
	auto WarpedImagesOptions =
	torch::TensorOptions()
		.dtype(torch::kByte)
		.device(torch::kCUDA, 0);
	auto warpedImages = torch::empty({nImages - 1, nPlanes, height, width}, WarpedImagesOptions);

	int l = 0;
	const unsigned char *p_images = images.data_ptr<unsigned char>();
	const float3 *p_rays = reinterpret_cast<float3 *>(rays.data_ptr<float>());
	const float *p_R_ref = Rs.data_ptr<float>() + 9 * ref;
	const float *p_T_ref = Ts.data_ptr<float>() + 3 * ref;
	const float *p_P = Ps.data_ptr<float>();
	for (int k = 0; k < nImages; k++)
	{
		const dim3 threads(BLOCK_SIZE_RAYS, BLOCK_SIZE_PLANES);
		const dim3 blocks(iDivUp(height * width, threads.x), iDivUp(nPlanes, threads.y));
		if (k == ref)
		{
			continue;
		}

		const float *p_K_other = Ks.data_ptr<float>() + 9 * k;
		const float *p_R_other = Rs.data_ptr<float>() + 9 * k;
		const float *p_T_other = Ts.data_ptr<float>() + 3 * k;
		const float *p_xi_other = xis.data_ptr<float>() + k;
		TextureObjectCreator<unsigned char> toc(p_images + k * width * height, width, height);
		cudaTextureObject_t tex = toc.getTextureObject();
		unsigned char *p_warpedImages = warpedImages.data_ptr<unsigned char>() + (l * width * height * nPlanes);
		// manual dispatch
		Intrinsics ko = getIntrinsics(p_K_other, *p_xi_other);
		float R[3][3];
		getRelativeRotation(p_R_ref, p_R_other, R);
		float3 r1 = make_float3(R[0][0], R[0][1], R[0][2]);
		float3 r2 = make_float3(R[1][0], R[1][1], R[1][2]);
		float3 r3 = make_float3(R[2][0], R[2][1], R[2][2]);
		float3 t = getRelativeTranslation(p_T_ref, p_T_other, R);
		getWarpedImageTensorCUDAKernel<<<blocks, threads>>>(
				p_warpedImages,
				tex,
				p_rays, ko,
				r1, r2, r3, t,
				p_P,
				nPlanes, height, width);
		l++;
	}

	return warpedImages;
}

torch::Tensor getRayTensorCUDA(int ref, torch::Tensor images, torch::Tensor Ks, torch::Tensor xis)
{
	const int width = images.size(2), height = images.size(1);
	auto RaysOptions =
	torch::TensorOptions()
		.dtype(torch::kFloat)
		.device(torch::kCUDA, 0);
	auto rays = torch::empty({height, width, 3}, RaysOptions);

	const float *p_K_ref = Ks.data_ptr<float>() + 9 * ref;
	const float *p_xi_ref = xis.data_ptr<float>() + ref;
	float3 *p_rays = reinterpret_cast<float3 *>(rays.data_ptr<float>());
	Intrinsics kr = getIntrinsics(p_K_ref, *p_xi_ref);
	{
		const dim3 threads(32, 32);
		const dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
		getRaysCUDAKernel<<<blocks, threads>>>(
				p_rays,
				kr,
				height, width);
	}

	return rays;
}
