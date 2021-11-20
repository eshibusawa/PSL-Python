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
	const float* __restrict__ K_ref,
	const float* __restrict__ xi_ref,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	Intrinsics kr = getIntrinsics(K_ref, *xi_ref);
	const int indexOutput = indexX + indexY * width;
	float3 xyz = unproject(make_float2(indexX, indexY), kr);
	output[indexOutput] =  xyz;
}

__global__ void getWarpedImageTensorCUDAKernel(
	unsigned char* __restrict__ output,
	cudaTextureObject_t tex,
	const float* __restrict__ K_ref,
	const float* __restrict__ K_other,
	const float* __restrict__ R_ref,
	const float* __restrict__ R_other,
	const float* __restrict__ T_ref,
	const float* __restrict__ T_other,
	const float* __restrict__ xi_ref,
	const float* __restrict__ xi_other,
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

	// coordinate transform
	Intrinsics kr = getIntrinsics(K_ref, *xi_ref);
	Intrinsics ko = getIntrinsics(K_other, *xi_other);
	float R[3][3];
	getRelativeRotation(R_ref, R_other, R);
	float3 t = getRelativeTranslation(T_ref, T_other, R);
	float3 n = make_float3(P[4 * indexP], P[4 * indexP + 1], P[4 * indexP + 2]);
	float d = P[4 * indexP + 3];
	float H[3][3];
	computeHMatrix(R, t, n, d, H);
	float2 uvr = make_float2(indexX, indexY);
	float2 uvo = unprojectHProject(kr, ko, H, uvr);

	const int wh = width * height;
	const int indexOutput = indexX + indexY * width + indexP * wh;
	output[indexOutput] =  static_cast<unsigned char>(255 * __saturatef(tex2D<float>(tex, uvo.x / width, uvo.y / height)));
}

std::vector<torch::Tensor> getWarpedImageTensorCUDA(int ref, torch::Tensor images, torch::Tensor Ks, torch::Tensor xis, torch::Tensor Rs, torch::Tensor Ts, torch::Tensor Ps)
{
	const int nImages = images.size(0);
	const int nPlanes = Ps.size(0);
	const int width = images.size(2), height = images.size(1);
	auto WarpedImagesOptions =
	torch::TensorOptions()
		.dtype(torch::kByte)
		.device(torch::kCUDA, 0);
	auto warpedImages = torch::empty({nImages - 1, nPlanes, height, width}, WarpedImagesOptions);

	auto RaysOptions =
	torch::TensorOptions()
		.dtype(torch::kFloat)
		.device(torch::kCUDA, 0);
	auto rays = torch::empty({height, width, 3}, RaysOptions);

	const float *p_K_ref = Ks.data_ptr<float>() + 9 * ref;
	const float *p_xi_ref = xis.data_ptr<float>() + ref;
	{
		const dim3 threads(32, 32);
		const dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
		float3 *p_rays = reinterpret_cast<float3 *>(rays.data_ptr<float>());
		getRaysCUDAKernel<<<blocks, threads>>>(
				p_rays,
				p_K_ref,
				p_xi_ref,
				height, width);
	}

	int l = 0;
	const unsigned char *p_images = images.data_ptr<unsigned char>();
	const float *p_R_ref = Rs.data_ptr<float>() + 9 * ref;
	const float *p_T_ref = Ts.data_ptr<float>() + 3 * ref;
	const float *p_P = Ps.data_ptr<float>();
	for (int k = 0; k < nImages; k++)
	{
		const dim3 threads(256, 4);
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
		getWarpedImageTensorCUDAKernel<<<blocks, threads>>>(
				p_warpedImages,
				tex,
				p_K_ref,
				p_K_other,
				p_R_ref,
				p_R_other,
				p_T_ref,
				p_T_other,
				p_xi_ref,
				p_xi_other,
				p_P,
				nPlanes, height, width);
		l++;
	}

	return {warpedImages, rays};
}
