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

__constant__ CAMERA_MODEL::Intrinsics g_K;

extern "C" __global__ void unprojectTest(float3 *xyz)
{
    int indexX = blockDim.x * blockIdx.x + threadIdx.x;
    int indexY = blockDim.y * blockIdx.y + threadIdx.y;
    if ((indexX >= WIDTH) || (indexY >= HEIGHT))
    {
        return;
    }

    int index = indexX + indexY * WIDTH;
    xyz[index] = CAMERA_MODEL::unproject(make_float2(indexX, indexY), g_K);
}

extern "C" __global__ void projectTest(const float3 *xyz, float2 *xy)
{
    int indexX = blockDim.x * blockIdx.x + threadIdx.x;
    int indexY = blockDim.y * blockIdx.y + threadIdx.y;
    if ((indexX >= WIDTH) || (indexY >= HEIGHT))
    {
        return;
    }

    int index = indexX + indexY * WIDTH;
    xy[index] = CAMERA_MODEL::project(make_float3(xyz[index].x, xyz[index].y, xyz[index].z), g_K);
}
