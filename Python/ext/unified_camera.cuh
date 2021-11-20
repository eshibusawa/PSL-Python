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

#ifndef UNIFIED_CAMERA_CUH_
#define UNIFIED_CAMERA_CUH_

struct Intrinsics
{
	float2 fufv, u0v0;
	float xi;
};

inline __device__ Intrinsics getIntrinsics(const float *K, float xi)
{
	Intrinsics ret;
	ret.fufv = make_float2(K[0], K[4]);
	ret.u0v0 = make_float2(K[2], K[5]);
	ret.xi = xi;
	return ret;
}

inline __device__ float2 project(const float3 X, const Intrinsics K)
{
	float d = norm3df(X.x, X.y, X.z);
	float3 nX = make_float3(X.x / d, X.y / d, X.z / d);
	nX.z += K.xi;
	float2 rxy = make_float2(nX.x / nX.z, nX.y / nX.z);
	float2 xy = make_float2(K.fufv.x * rxy.x + K.u0v0.x, K.fufv.y * rxy.y + K.u0v0.y);
	return xy;
}

inline __device__ float3 unproject(const float2 x, const Intrinsics K)
{
	float2 rxy = make_float2((x.x - K.u0v0.x) / K.fufv.x, (x.y - K.u0v0.y) / K.fufv.y);
	float rsqr = rxy.x * rxy.x + rxy.y * rxy.y;
	float d = 1 + (1 - K.xi * K.xi) * rsqr;
	if (d < 0)
	{
		d = -d;
	}
	float F = (K.xi + sqrtf(d)) / (rsqr + 1);
	float3 X = make_float3(F * rxy.x, F * rxy.y, F - K.xi);
	return X;
}

#endif // UNIFIED_CAMERA_CUH_
