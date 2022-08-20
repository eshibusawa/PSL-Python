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

extern "C" __global__ void getRaysKernelCAMERA_MODEL(
	float3* output,
	const float *Kr,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	CAMERA_MODEL::Intrinsics kr = CAMERA_MODEL::getIntrinsics(Kr);
	float3 xyz = CAMERA_MODEL::unproject(make_float2(indexX, indexY), kr);
	const int indexOutput = indexX + indexY * width;
	output[indexOutput] =  xyz;
}

extern "C" __global__ void getHomographisKernel(
	float* output,
	float* __restrict__ Rref,
	float* __restrict__ Rother,
	float* __restrict__ Tref,
	float* __restrict__ Tother,
    float4* __restrict__ P,
    int numPlanes)
{
	const int indexP = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexP >= numPlanes)
	{
		return;
	}
    float R[9];
    getRelativeRotation(Rref, Rother, R);
    float3 t = getRelativeTranslation(Tref, Tother, R);

	const int indexOutput = indexP * 9;
    computeHMatrix(R, &t, &(P[indexP]), &(output[indexOutput]));
}

extern "C" __global__ void getWarpedImageKernelCAMERA_MODEL(
	unsigned char* output,
	cudaTextureObject_t tex,
	const float3* __restrict__ rays,
	const float* __restrict__ Ko,
	const float* __restrict__ Hs,
	int height,
	int width,
	int numPlanes)
{
	const int indexXY = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexX = indexXY % width, indexY = indexXY / width;
	const int indexP = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height) || (indexP >= numPlanes))
	{
		return;
	}
	const int indexH = 9 * indexP;
	CAMERA_MODEL::Intrinsics ko = CAMERA_MODEL::getIntrinsics(Ko);
	float3 Hray = apply3x3Transformation(&(Hs[indexH]), rays[indexXY]);
	float2 uvo = CAMERA_MODEL::project(Hray, ko);
	const int wh = width * height;
	const int indexOutput = indexX + indexY * width + indexP * wh;

	// CuPy doesn't seems to cudaAddressModeWrap
	uvo.x = (uvo.x >= width) ? uvo.x - width : uvo.x;
	uvo.x = (uvo.x < 0) ? uvo.x + width : uvo.x;
	uvo.y = (uvo.y >= height) ? uvo.y - height : uvo.y;
	uvo.y = (uvo.y < 0) ? uvo.y + height : uvo.y;

	output[indexOutput] =  static_cast<unsigned char>(255 * __saturatef(tex2D<float>(tex, uvo.x, uvo.y)));
}

extern "C" __global__ void getDepthKernel(
	float* output,
	const float3* __restrict__ rays,
	const long* __restrict__ indices,
	const float4* __restrict__ Ps,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	const int index = indexX + indexY * width;
	const int indexP = indices[index];
	float4 plane = make_float4(Ps[indexP].x, Ps[indexP].y, Ps[indexP].z, Ps[indexP].w);
	output[index] = getDepthFromRayAndPlane(rays[index], plane);
}
