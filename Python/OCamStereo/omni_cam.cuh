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

#ifndef OMNI_CAM_CUH_
#define OMNI_CAM_CUH_

namespace OmniCameraModel
{
struct Intrinsics
{
	float2 u0v0;
	float affine[4];
	float inverseAffine[4];
	float inversePolynomial[OC_INVERSE_POLYNOMIAL_COEF_NUM]; // variable-length may cause alignment issue
	float polynomial[OC_POLYNOMIAL_COEF_NUM];
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
	sz[2] = (reinterpret_cast<char *>(&(K.affine)) - reinterpret_cast<char *>(&K)) / sizeof(float);
	sz[3] = (reinterpret_cast<char *>(&(K.inverseAffine)) - reinterpret_cast<char *>(&K)) / sizeof(float);
	sz[4] = (reinterpret_cast<char *>(&(K.inversePolynomial)) - reinterpret_cast<char *>(&K)) / sizeof(float);
	sz[5] = (reinterpret_cast<char *>(&(K.polynomial)) - reinterpret_cast<char *>(&K)) / sizeof(float);
}

inline __device__ float2 project(const float3 X, const Intrinsics K)
{
	float n = hypotf(X.x, X.y);
	float theta = atanf(-X.z / n);
	float rho = K.inversePolynomial[0];
	#pragma unroll (OC_INVERSE_POLYNOMIAL_COEF_NUM - 1)
	for (int i = 1; i < OC_INVERSE_POLYNOMIAL_COEF_NUM; i++)
	{
		rho = fmaf(rho, theta, K.inversePolynomial[i]);
	}
	float2 xy = make_float2(X.x / n * rho, X.y / n * rho);
	float2 uv;
	uv.x = fmaf(xy.x, K.affine[0], fmaf(xy.y, K.affine[1], K.u0v0.x));
	uv.y = fmaf(xy.x, K.affine[2], fmaf(xy.y, K.affine[3], K.u0v0.y));
	return uv;
}

inline __device__ float3 unproject(const float2 x, const Intrinsics K)
{
	float2 axy = make_float2(x.x - K.u0v0.x, x.y - K.u0v0.y);
	float2 xy;
	xy.x = fmaf(axy.x, K.inverseAffine[0], axy.y * K.inverseAffine[1]);
	xy.y = fmaf(axy.x, K.inverseAffine[2], axy.y * K.inverseAffine[3]);
	float r = hypotf(xy.x, xy.y);
	float rho = K.polynomial[0];
	#pragma unroll (OC_POLYNOMIAL_COEF_NUM - 1)
	for (int i = 1; i < OC_POLYNOMIAL_COEF_NUM; i++)
	{
		rho = fmaf(rho, r, K.polynomial[i]);
	}
	float3 X = make_float3(xy.x, xy.y, rho);
	float n = norm3df(X.x, X.y, X.z);
	X.x /= n;
	X.y /= n;
	X.z /= n;

	return X;
}
};

#endif // OMNI_CAM_CUH_
