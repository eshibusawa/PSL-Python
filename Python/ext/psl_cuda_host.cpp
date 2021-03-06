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

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// CUDA forward declarations
namespace UnifiedCameraModel
{
torch::Tensor getWarpedImageTensorCUDA(int ref, torch::Tensor image, torch::Tensor Ks, torch::Tensor Rs, torch::Tensor Ts, torch::Tensor rays, torch::Tensor Ps);
torch::Tensor getRayTensorCUDA(int ref, torch::Tensor image, torch::Tensor Ks);
};

torch::Tensor getDepthTensorCUDA(torch::Tensor rays, torch::Tensor indices, torch::Tensor Ps);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

torch::Tensor getWarpedImageTensor(int ref, torch::Tensor images, torch::Tensor Ks, torch::Tensor Rs, torch::Tensor Ts, torch::Tensor rays, torch::Tensor Ps)
{
	CHECK_INPUT(images);
	CHECK_INPUT_CPU(Ks);
	CHECK_INPUT_CPU(Rs);
	CHECK_INPUT_CPU(Ts);
	CHECK_INPUT(rays);
	CHECK_INPUT(Ps);

	if ((images.sizes().size() != 3) || (images.scalar_type() != torch::kByte) ||
		(Ks.sizes().size() != 2) || (Ks.scalar_type() != torch::kFloat) ||
		(Rs.sizes().size() != 3) || (Rs.scalar_type() != torch::kFloat) ||
		(Ts.sizes().size() != 3) || (Ts.scalar_type() != torch::kFloat) ||
		(rays.sizes().size() != 3) || (rays.scalar_type() != torch::kFloat) ||
		(Ps.sizes().size() != 2) || (Ps.scalar_type() != torch::kFloat))
	{
		std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
		return torch::Tensor();
	}

	return UnifiedCameraModel::getWarpedImageTensorCUDA(ref, images, Ks, Rs, Ts, rays, Ps);
}

torch::Tensor getRayTensor(int ref, torch::Tensor images, torch::Tensor Ks)
{
	CHECK_INPUT(images);
	CHECK_INPUT_CPU(Ks);

	if ((images.sizes().size() != 3) || (images.scalar_type() != torch::kByte) ||
		(Ks.sizes().size() != 2) || (Ks.scalar_type() != torch::kFloat))
	{
		std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
		return torch::Tensor();
	}

	return UnifiedCameraModel::getRayTensorCUDA(ref, images, Ks);
}

torch::Tensor getDepthTensor(torch::Tensor rays, torch::Tensor indices, torch::Tensor Ps)
{
	CHECK_INPUT(rays);
	CHECK_INPUT(indices);
	CHECK_INPUT(Ps);

	if ((rays.sizes().size() != 3) || (rays.scalar_type() != torch::kFloat) ||
		(indices.sizes().size() != 2) || (indices.scalar_type() != torch::kLong) ||
		(Ps.sizes().size() != 2) || (Ps.scalar_type() != torch::kFloat))
	{
		std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
		return torch::Tensor();
	}

	return getDepthTensorCUDA(rays, indices, Ps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_ray_tensor", &getRayTensor, "compute ray (CUDA)");
	m.def("get_warped_image_tensor", &getWarpedImageTensor, "warp images (CUDA)");
	m.def("get_depth_tensor", &getDepthTensor, "compute depth (CUDA)");
}
