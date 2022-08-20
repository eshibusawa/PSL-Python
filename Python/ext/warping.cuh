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

#ifndef WARPING_CUH_
#define WARPING_CUH_

inline __device__ float3 apply3x3Transformation(const float *A, float3 x)
{
	return make_float3
	(
		A[0] * x.x + A[1] * x.y + A[2] * x.z,
		A[3] * x.x + A[4] * x.y + A[5] * x.z,
		A[6] * x.x + A[7] * x.y + A[8] * x.z
	);
}

inline __device__ void getRelativeRotation(const float *R_ref, const float *R_other, float *R)
{
	R[0] = R_other[0] * R_ref[0] + R_other[1] * R_ref[1] + R_other[2] * R_ref[2];
	R[1] = R_other[0] * R_ref[3] + R_other[1] * R_ref[4] + R_other[2] * R_ref[5];
	R[2] = R_other[0] * R_ref[6] + R_other[1] * R_ref[7] + R_other[2] * R_ref[8];
	R[3] = R_other[3] * R_ref[0] + R_other[4] * R_ref[1] + R_other[5] * R_ref[2];
	R[4] = R_other[3] * R_ref[3] + R_other[4] * R_ref[4] + R_other[5] * R_ref[5];
	R[5] = R_other[3] * R_ref[6] + R_other[4] * R_ref[7] + R_other[5] * R_ref[8];
	R[6] = R_other[6] * R_ref[0] + R_other[7] * R_ref[1] + R_other[8] * R_ref[2];
	R[7] = R_other[6] * R_ref[3] + R_other[7] * R_ref[4] + R_other[8] * R_ref[5];
	R[8] = R_other[6] * R_ref[6] + R_other[7] * R_ref[7] + R_other[8] * R_ref[8];
}

inline __device__ float3 getRelativeTranslation(const float *T_ref, const float *T_other, const float *R)
{
	return make_float3(
		R[0] * T_ref[0] + R[1] * T_ref[1] + R[2] * T_ref[2] - T_other[0],
		R[3] * T_ref[0] + R[4] * T_ref[1] + R[5] * T_ref[2] - T_other[1],
		R[6] * T_ref[0] + R[7] * T_ref[1] + R[8] * T_ref[2] - T_other[2]
	);
}

inline __device__ void computeHMatrix(const float *R, float3 *t, float4 *p, float *H)
{
	H[0] = R[0] + (t->x * p->x / p->w);
	H[1] = R[1] + (t->x * p->y / p->w);
	H[2] = R[2] + (t->x * p->z / p->w);
	H[3] = R[3] + (t->y * p->x / p->w);
	H[4] = R[4] + (t->y * p->y / p->w);
	H[5] = R[5] + (t->y * p->z / p->w);
	H[6] = R[6] + (t->z * p->x / p->w);
	H[7] = R[7] + (t->z * p->y / p->w);
	H[8] = R[8] + (t->z * p->z / p->w);
}

#endif // WARPING_CUH_
