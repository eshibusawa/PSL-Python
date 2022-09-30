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

#ifndef EQUIDISTANT_CAM_CUH_
#define EQUIDISTANT_CAM_CUH_

namespace EquidistantCameraModel
{
struct Intrinsics
{
	float2 u0v0;
	float f;
};

extern "C" __global__ void getIntrinsicSize(int *sz)
{
	if ((blockDim.x * blockIdx.x + threadIdx.x != 0) || (blockDim.y * blockIdx.y + threadIdx.y != 0))
	{
		return;
	}

	Intrinsics K;
	sz[0] = sizeof(Intrinsics) / sizeof(float);
	sz[1] = (reinterpret_cast<char *>(&(K.u0v0)) - reinterpret_cast<char *>(&K)) / sizeof(float);
	sz[2] = (reinterpret_cast<char *>(&(K.f)) - reinterpret_cast<char *>(&K)) / sizeof(float);
}

inline __device__ float2 project(const float3 X, const Intrinsics K)
{
	float R = hypotf(X.x, X.y);
	float theta = atanf(R / X.z);
	float phi = atan2f(X.y, X.x);
	float2 uv;
	__sincosf(phi, &(uv.y), &(uv.x));
	uv.x = fmaf(K.f * theta, uv.x, K.u0v0.x);
	uv.y = fmaf(K.f * theta, uv.y, K.u0v0.y);
	return uv;
}

inline __device__ float3 unproject(const float2 x, const Intrinsics K)
{
	float2 xy = make_float2(x.x - K.u0v0.x, x.y - K.u0v0.y);
	float theta = hypotf(xy.x, xy.y) / K.f;
	float phi = atan2f(xy.y, xy.x);
	float3 X;
	__sincosf(phi, &(X.y), &(X.x));
	float2 sc;
	__sincosf(theta, &(sc.x), &(sc.y));
	X.x *= sc.x;
	X.y *= sc.x;
	X.z = sc.y;

	return X;
}
};

#endif // EQUIDISTANT_CAM_CUH_
