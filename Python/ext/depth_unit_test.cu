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

__constant__ float g_planes[NUM_PLANES][4];
__constant__ bool g_planesHasNeighbor[NUM_PLANES];

extern "C" __global__ void getDepthTest(const float3 *rays, const long *indices, float *depth)
{
	const int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	const int indexY = blockDim.y * blockIdx.y + threadIdx.y;

	if ((indexX >= WIDTH) || (indexY >= HEIGHT))
	{
		return;
	}

	const int index = indexX + indexY * WIDTH;
	const int indexP = indices[index];
	float4 plane = make_float4(g_planes[indexP][0], g_planes[indexP][1], g_planes[indexP][2], g_planes[indexP][3]);
	depth[index] = getDepthFromRayAndPlane(rays[index], plane);
}

extern "C" __global__ void getDepthWithSubpixelDirectTest(const float3 *rays, const long *indices, const float *CV, float *depth)
{
	const int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	const int indexY = blockDim.y * blockIdx.y + threadIdx.y;

	if ((indexX >= WIDTH) || (indexY >= HEIGHT))
	{
		return;
	}

	const int index = indexX + indexY * WIDTH;
	const int indexP = indices[index];

	float4 plane = make_float4(g_planes[indexP][0], g_planes[indexP][1], g_planes[indexP][2], g_planes[indexP][3]);
	if (g_planesHasNeighbor[indexP])
	{
		const int wh = WIDTH * HEIGHT;
		const int indexC0 = (indexX + indexY * WIDTH) + indexP * wh;
		const int indexCm = (indexX + indexY * WIDTH) + (indexP - 1) * wh;
		const int indexCp = (indexX + indexY * WIDTH) + (indexP + 1) * wh;
		float3 cost = make_float3(CV[indexCm], CV[indexC0], CV[indexCp]);
		float3 plane_depth = make_float3(g_planes[indexP - 1][3], g_planes[indexP][3], g_planes[indexP + 1][3]);
		depth[index] = getDepthFromRayAndPlaneWithSubpixelDirect(rays[index], plane, plane_depth, cost);
	}
	else
	{
		depth[index] = getDepthFromRayAndPlane(rays[index], plane);
	}
}

extern "C" __global__ void getDepthWithSubpixelInverseTest(const float3 *rays, const long *indices, const float *CV, float *depth)
{
	const int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	const int indexY = blockDim.y * blockIdx.y + threadIdx.y;

	if ((indexX >= WIDTH) || (indexY >= HEIGHT))
	{
		return;
	}

	const int index = indexX + indexY * WIDTH;
	const int indexP = indices[index];

	float4 plane = make_float4(g_planes[indexP][0], g_planes[indexP][1], g_planes[indexP][2], g_planes[indexP][3]);
	if (g_planesHasNeighbor[indexP])
	{
		const int wh = WIDTH * HEIGHT;
		const int indexC0 = (indexX + indexY * WIDTH) + indexP * wh;
		const int indexCm = (indexX + indexY * WIDTH) + (indexP - 1) * wh;
		const int indexCp = (indexX + indexY * WIDTH) + (indexP + 1) * wh;
		float3 cost = make_float3(CV[indexCm], CV[indexC0], CV[indexCp]);
		float3 plane_depth = make_float3(g_planes[indexP - 1][3], g_planes[indexP][3], g_planes[indexP + 1][3]);
		depth[index] = getDepthFromRayAndPlaneWithSubpixelInverse(rays[index], plane, plane_depth, cost);
	}
	else
	{
		depth[index] = getDepthFromRayAndPlane(rays[index], plane);
	}
}
