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

__constant__ CAMERA_MODEL::Intrinsics g_K_ref;
__constant__ CAMERA_MODEL::Intrinsics g_K_other;
__constant__ float g_R[9];
__constant__ float g_t[3];
__constant__ float g_planes[NUM_PLANES][4];

extern "C" __global__ void warpingTest(float2 *xy)
{
	int indexXY = blockDim.x * blockIdx.x + threadIdx.x;
	int indexP = blockDim.y * blockIdx.y + threadIdx.y;
	int indexX = indexXY % WIDTH;
	int indexY = indexXY / WIDTH;

	if ((indexX >= WIDTH) || (indexY >= HEIGHT)  || (indexP >= NUM_PLANES))
	{
		return;
	}

	int index = indexX + indexY * WIDTH + (WIDTH * HEIGHT * indexP);

	float H[9];
	float3 t = make_float3(g_t[0], g_t[1], g_t[2]);
	float4 P = make_float4(g_planes[indexP][0],
			g_planes[indexP][1],
			g_planes[indexP][2],
			g_planes[indexP][3]);
	computeHMatrix(g_R, &t, &P,	H);
	xy[index] = CAMERA_MODEL::project(apply3x3Transformation(H, CAMERA_MODEL::unproject(make_float2(indexX, indexY), g_K_ref)), g_K_other);
}
