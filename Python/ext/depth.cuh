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

#ifndef DEPTH_CUH_
#define DEPTH_CUH_

#define DEPTH_DENOM_EPS (1E-5)

inline __device__ float getDepthFromRayAndPlane(const float3 ray, const float4 plane)
{
	float x = ray.x / ray.z, y = ray.y / ray.z;
	float ray_dot_plane = x * plane.x  + y * plane.y + plane.z;
	return -plane.w / ray_dot_plane;
}

inline __device__ float getDepthFromRayAndPlaneWithSubpixelDirect(const float3 ray, const float4 plane, const float3 plane_depth, const float3 cost)
{
	// [cost.x, cost.y, cost.z] = [C[-1], C[0], C[1]]
	float denom = cost.z + cost.x - cost.y;
	if (fabsf(denom) < DEPTH_DENOM_EPS)
	{
		return getDepthFromRayAndPlane(ray, plane);
	}

	float offset = (cost.x - cost.z)/(2 * denom);
	float step = 0;
	if (offset < 0)
	{
		step = plane_depth.y - plane_depth.x;
	}
	else
	{
		step = plane_depth.z - plane_depth.y;
	}
	float depth_sub = offset * step;

	float x = ray.x / ray.z, y = ray.y / ray.z;
	float ray_dot_plane = x * plane.x  + y * plane.y + plane.z;
	return -(plane_depth.y + depth_sub) / ray_dot_plane;
}

inline __device__ float getDepthFromRayAndPlaneWithSubpixelInverse(const float3 ray, const float4 plane, const float3 plane_depth, const float3 cost)
{
	// [cost.x, cost.y, cost.z] = [C[-1], C[0], C[1]]
	float denom = cost.z + cost.x - cost.y;
	if (fabsf(denom) < DEPTH_DENOM_EPS)
	{
		return getDepthFromRayAndPlane(ray, plane);
	}

	float offset = (cost.x - cost.z)/(2 * denom);
	float step = 0;
	if (offset < 0)
	{
		step = (1 / plane_depth.y) - (1 / plane_depth.z);
	}
	else
	{
		step = (1 / plane_depth.x) - (1 / plane_depth.y);
	}
	float depth_sub = offset * step;

	float x = ray.x / ray.z, y = ray.y / ray.z;
	float ray_dot_plane = x * plane.x  + y * plane.y + plane.z;
	return -1/(1/plane_depth.y - depth_sub) / ray_dot_plane;
}
#endif // DEPTH_CUH_
