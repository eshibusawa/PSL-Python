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

inline __device__ float3 apply3x3Transformation(const float A[3][3], float3 x)
{
	return make_float3
	(
		A[0][0] * x.x + A[0][1] * x.y + A[0][2] * x.z,
		A[1][0] * x.x + A[1][1] * x.y + A[1][2] * x.z,
		A[2][0] * x.x + A[2][1] * x.y + A[2][2] * x.z
	);
}

inline __host__ void getRelativeRotation(const float *R_ref, const float *R_other, float R[3][3])
{
	R[0][0] = R_other[0] * R_ref[0] + R_other[1] * R_ref[1] + R_other[2] * R_ref[2];
	R[0][1] = R_other[0] * R_ref[3] + R_other[1] * R_ref[4] + R_other[2] * R_ref[5];
	R[0][2] = R_other[0] * R_ref[6] + R_other[1] * R_ref[7] + R_other[2] * R_ref[8];
	R[1][0] = R_other[3] * R_ref[0] + R_other[4] * R_ref[1] + R_other[5] * R_ref[2];
	R[1][1] = R_other[3] * R_ref[3] + R_other[4] * R_ref[4] + R_other[5] * R_ref[5];
	R[1][2] = R_other[3] * R_ref[6] + R_other[4] * R_ref[7] + R_other[5] * R_ref[8];
	R[2][0] = R_other[6] * R_ref[0] + R_other[7] * R_ref[1] + R_other[8] * R_ref[2];
	R[2][1] = R_other[6] * R_ref[3] + R_other[7] * R_ref[4] + R_other[8] * R_ref[5];
	R[2][2] = R_other[6] * R_ref[6] + R_other[7] * R_ref[7] + R_other[8] * R_ref[8];
}

inline __host__ float3 getRelativeTranslation(const float *T_ref, const float *T_other, float R[3][3])
{
	return make_float3(
		R[0][0] * T_ref[0] + R[0][1] * T_ref[1] + R[0][2] * T_ref[2] - T_other[0],
		R[1][0] * T_ref[0]+ R[1][1] * T_ref[1] + R[1][2] * T_ref[2] - T_other[1],
		R[2][0] * T_ref[0] + R[2][1] * T_ref[1] + R[2][2] * T_ref[2] - T_other[2]
	);
}

inline __device__ void computeHMatrix(const float R[3][3], float3 t, float3 n, float d, float H[3][3])
{
	H[0][0] = R[0][0] + (t.x * n.x / d);
	H[0][1] = R[0][1] + (t.x * n.y / d);
	H[0][2] = R[0][2] + (t.x * n.z / d);
	H[1][0] = R[1][0] + (t.y * n.x / d);
	H[1][1] = R[1][1] + (t.y * n.y / d);
	H[1][2] = R[1][2] + (t.y * n.z / d);
	H[2][0] = R[2][0] + (t.z * n.x / d);
	H[2][1] = R[2][1] + (t.z * n.y / d);
	H[2][2] = R[2][2] + (t.z * n.z / d);
}

#endif // WARPING_CUH_
