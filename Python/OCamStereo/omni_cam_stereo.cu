// This file is part of PSL-Python.
// Copyright (c) 2022, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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

#define MATCH_WINDOW_RADIUS_H (MATCH_WINDOW_WIDTH / 2)
#define MATCH_WINDOW_RADIUS_V (MATCH_WINDOW_HEIGHT / 2)

__constant__ OmniCameraModel::Intrinsics g_K_other;
__constant__ float g_H[NUM_PLANES][9];
__constant__ float4 g_nd[NUM_PLANES];

__device__ float ZNCCCostFunction(const unsigned char *ref, const unsigned char *other)
{
	const int sz = (MATCH_WINDOW_WIDTH * MATCH_WINDOW_HEIGHT);
	float muRef = 0, muOther = 0;
	for (int k = 0; k < sz; k++)
	{
		muRef += ref[k];
		muOther += other[k];
	}
	muRef /= sz;
	muOther /= sz;
	float costNume = 0;
	float costDenom1 = 0;
	float costDenom2 = 0;
	float zmRef, zmOther;
	for (int k = 0; k < sz; k++)
	{
		zmRef = ref[k] - muRef;
		zmOther = other[k] - muOther;
		// (ref_k - mu_ref_k) * (other_k - mu_other_k)
		costNume = fmaf(zmRef, zmOther, costNume);
		// (ref_k - mu_ref_k)^2
		costDenom1 = fmaf(zmRef, zmRef, costDenom1);
		// (other_k - mu_other_k)^2
		costDenom2 = fmaf(zmOther, zmOther, costDenom2);
	}
	float zncc = costNume / (sqrtf(costDenom1 * costDenom2) + 1E-7);

	return (1 - zncc)/2;
}

float3 __device__ blockMatching(
	const int indexU,
	const int indexV,
	const float3* __restrict__ rays,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texMask,
	int height,
	int width
	)
{
	const int indexUV = indexU + indexV * width;
	unsigned char patch_ref[MATCH_WINDOW_WIDTH * MATCH_WINDOW_HEIGHT];
	unsigned char patch_other[MATCH_WINDOW_WIDTH * MATCH_WINDOW_HEIGHT];
	float minCost = (1UL << 31), cost = 0.f;
	int minIndexP = 0;
	for (int indexP = 0; indexP < NUM_PLANES; indexP++)
	{
		for (int j = -MATCH_WINDOW_RADIUS_V, k = 0; j <= MATCH_WINDOW_RADIUS_V; j++)
		{
			for (int i = -MATCH_WINDOW_RADIUS_H; i <= MATCH_WINDOW_RADIUS_H; i++)
			{
				if (tex2D<unsigned char>(texMask, indexU + i, indexV + j) == 0)
				{
					patch_ref[k] = 0;
					patch_other[k] = 255;
					k++;
					continue;
				}

				patch_ref[k] = tex2D<unsigned char>(texRef, indexU + i, indexV + j);
				int indexUV2 = indexU + i + (indexV + j) * width;
				float3 xyz_other2 = apply3x3Transformation(g_H[indexP], rays[indexUV2]);
				float2 uv_other2 = OmniCameraModel::project(xyz_other2, g_K_other);
				float val_other = tex2D<float>(texOther, uv_other2.x, uv_other2.y);
				patch_other[k] = static_cast<unsigned char>(255 * val_other);
				k++;
			}
		}
		cost = ZNCCCostFunction(patch_ref, patch_other);

		if (cost < minCost)
		{
			minCost = cost;
			minIndexP = indexP;
		}
	}

	float d = getDepthFromRayAndPlane(rays[indexUV], g_nd[minIndexP]);
	float3 output;
	output.x = d * rays[indexUV].x / rays[indexUV].z;
	output.y = d * rays[indexUV].y / rays[indexUV].z;
	output.z = d;
	return output;
}

extern "C" __global__ void getXYZ(
	float3* output,
	const float3* __restrict__ rays,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texMask,
	int height,
	int width
	)
{
	const int indexU = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexV = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexV >= height) || (indexU >= width))
	{
		return;
	}
	const int indexUV = indexU + indexV * width;
    if (tex2D<unsigned char>(texMask, indexU, indexV) == 0)
	{
		output[indexUV] = make_float3(0, 0, -1);
		return;
	}
	output[indexUV] = blockMatching(indexU, indexV, rays, texRef, texOther, texMask, height, width);
}

extern "C" __global__ void getXYZMaskIndexed(
	float3* output,
	const float3* __restrict__ rays,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texMask,
	cudaTextureObject_t texMaskIndexed,
	int height,
	int width,
	int heightMask,
	int widthMask
	)
{
	const int indexS = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexT = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexT >= heightMask) || (indexS >= widthMask))
	{
		return;
	}
	const short2 indexMask = tex2D<short2>(texMaskIndexed, indexS, indexT);
	const int indexU = indexMask.x, indexV = indexMask.y;
	if ((indexV < 0) || (indexU < 0))
	{
		return;
	}
	const int indexUV = indexU + indexV * width;
	output[indexUV] = blockMatching(indexU, indexV, rays, texRef, texOther, texMask, height, width);
}

extern "C" __global__ void getMaskFromIndexed(
	unsigned int* output,
	cudaTextureObject_t texMaskIndexed,
	int height,
	int width,
	int heightMask,
	int widthMask
	)
{
	const int indexS = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexT = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexT >= heightMask) || (indexS >= widthMask))
	{
		return;
	}
	const short2 indexMask = tex2D<short2>(texMaskIndexed, indexS, indexT);
	const int indexU = indexMask.x, indexV = indexMask.y;
	if ((indexV < 0) || (indexU < 0))
	{
		return;
	}
	const int indexUV = indexU + indexV * width;
	output[indexUV] = indexS + indexT * widthMask + 1;
}

extern "C" __global__ void getTableFromIndexed(
	short2* output,
	const float3* __restrict__ rays,
	cudaTextureObject_t texMaskIndexed,
	int height,
	int width,
	int heightMask,
	int widthMask
	)
{
	const int indexST = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexP = blockIdx.y * blockDim.y + threadIdx.y;
	const int indexS = indexST % widthMask, indexT = indexST / widthMask;
	if ((indexT >= heightMask) || (indexS >= widthMask) || (indexP >= NUM_PLANES))
	{
		return;
	}
	const short2 indexMask = tex2D<short2>(texMaskIndexed, indexS, indexT);
	const int indexU = indexMask.x, indexV = indexMask.y;
	if ((indexV < 0) || (indexU < 0))
	{
		return;
	}
	const int indexUV = indexU + indexV * width;
	float3 xyz_other = apply3x3Transformation(g_H[indexP], rays[indexUV]);
	float2 uv_other = OmniCameraModel::project(xyz_other, g_K_other);
	const int index = indexP + (indexS * NUM_PLANES) + (indexT * widthMask * NUM_PLANES);
	output[index].x = static_cast<short>(uv_other.x * 16);
	output[index].y = static_cast<short>(uv_other.y * 16);
}

float3 __device__ blockMatchingTable(
	const int indexU,
	const int indexV,
	const float3 ray,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texMaskLength,
	cudaTextureObject_t texMaskTable,
	int height,
	int width,
	int heightMask,
	int widthMask
	)
{
	unsigned char patch_ref[MATCH_WINDOW_WIDTH * MATCH_WINDOW_HEIGHT];
	unsigned char patch_other[MATCH_WINDOW_WIDTH * MATCH_WINDOW_HEIGHT];
	float minCost = (1UL << 31), cost = 0.f;
	int minIndexP = 0;
	for (int indexP = 0; indexP < NUM_PLANES; indexP++)
	{
		for (int j = -MATCH_WINDOW_RADIUS_V, k = 0; j <= MATCH_WINDOW_RADIUS_V; j++)
		{
			for (int i = -MATCH_WINDOW_RADIUS_H; i <= MATCH_WINDOW_RADIUS_H; i++)
			{
				const int indexL = tex2D<unsigned int>(texMaskLength, indexU + i, indexV + j);
				if (indexL == 0)
				{
					patch_ref[k] = 0;
					patch_other[k] = 255;
					k++;
					continue;
				}

				patch_ref[k] = tex2D<unsigned char>(texRef, indexU + i, indexV + j);
				const int indexS2 = (indexL - 1) % widthMask, indexT2 = (indexL - 1) / widthMask;
				short2 uv_other2 = tex2D<short2>(texMaskTable, NUM_PLANES * indexS2 + indexP, indexT2);
				float val_other = tex2D<float>(texOther, 0.0625f * uv_other2.x, 0.0625f * uv_other2.y);
				patch_other[k] = static_cast<unsigned char>(255 * val_other);
				k++;
			}
		}
		cost = ZNCCCostFunction(patch_ref, patch_other);

		if (cost < minCost)
		{
			minCost = cost;
			minIndexP = indexP;
		}
	}

	float d = getDepthFromRayAndPlane(ray, g_nd[minIndexP]);
	float3 output;
	output.x = d * ray.x / ray.z;
	output.y = d * ray.y / ray.z;
	output.z = d;
	return output;
}

extern "C" __global__ void getXYZMaskIndexedTable(
	float3* output,
	const float3* __restrict__ rays,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texMaskIndexed,
	cudaTextureObject_t texMaskLength,
	cudaTextureObject_t texMaskTable,
	int height,
	int width,
	int heightMask,
	int widthMask
	)
{
	const int indexS = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexT = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexT >= heightMask) || (indexS >= widthMask))
	{
		return;
	}
	const short2 indexMask = tex2D<short2>(texMaskIndexed, indexS, indexT);
	const int indexU = indexMask.x, indexV = indexMask.y;
	if ((indexV < 0) || (indexU < 0))
	{
		return;
	}
	const int indexUV = indexU + indexV * width;
	output[indexUV] = blockMatchingTable(indexU, indexV, rays[indexUV], texRef, texOther, texMaskLength, texMaskTable, height, width, heightMask, widthMask);
}
