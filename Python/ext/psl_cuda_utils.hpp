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

#ifndef PSL_CUDA_UTILS_HPP_
#define PSL_CUDA_UTILS_HPP_

#include <cuda_runtime.h>
#include <iostream>

template <typename T>
inline void check(T result, char const *const func, const char *const file, int const line) {
	if (cudaSuccess != result)
	{
		std::cerr << "CUDA Error: " << file << ":" << line;
		std::cerr << " code=" << static_cast<unsigned int>(result);
		std::cerr << "(" << cudaGetErrorString(result) << ") \"" << func << "\"" << std::endl;
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline int iDivUp(int a, int b)
{
	return static_cast<int>(std::ceil(static_cast<float>(a)/b));
}

template <typename ScalarType>
class TextureObjectCreator
{
public:
	TextureObjectCreator(const ScalarType *p, int w, int h):
		m_pDevice(p)
		, m_width(w)
		, m_height(h)
		, m_texObject(0)
	{
	}

	~TextureObjectCreator()
	{
		checkCudaErrors(cudaDestroyTextureObject(m_texObject));
	}

	cudaTextureObject_t getTextureObject()
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8 * sizeof(ScalarType), 0, 0, 0, cudaChannelFormatKindUnsigned);
		memset(&m_texRes, 0, sizeof(cudaResourceDesc));
		m_texRes.resType = cudaResourceTypePitch2D;
		m_texRes.res.pitch2D.devPtr = const_cast<ScalarType *>(m_pDevice);
		m_texRes.res.pitch2D.desc = channelDesc;
		m_texRes.res.pitch2D.width = m_width;
		m_texRes.res.pitch2D.height = m_height;
		m_texRes.res.pitch2D.pitchInBytes = m_width * sizeof(ScalarType);

		memset(&m_texDescr, 0, sizeof(cudaTextureDesc));
		m_texDescr.normalizedCoords = true;
		m_texDescr.filterMode = cudaFilterModeLinear;
		m_texDescr.addressMode[0] = cudaAddressModeWrap;
		m_texDescr.addressMode[1] = cudaAddressModeWrap;
		m_texDescr.readMode = cudaReadModeNormalizedFloat;
		checkCudaErrors(cudaCreateTextureObject(&m_texObject, &m_texRes, &m_texDescr, NULL));

		return m_texObject;
	}

private:
	const ScalarType *m_pDevice;
	int m_width, m_height;
	cudaTextureObject_t m_texObject;
	cudaTextureDesc m_texDescr;
	cudaResourceDesc m_texRes;
};

#endif // PSL_CUDA_UTILS_HPP_