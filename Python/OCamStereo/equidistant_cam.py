# This file is part of PSL-Python.
# Copyright (c) 2022, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import math

import numpy as np
import cupy as cp

import add_path
from util_cuda import upload_constant
import texture

class EquidistantCam():
    def __init__(self, p):
        self.f = p[0]
        self.u0 = p[1]
        self.v0 = p[2]
        self.gpu_module = None
        self.rays = None
        self.mask = None
        self.sz = None

    def get_intrincs_array(self):
        p = list()
        p.append(self.u0)
        p.append(self.v0)
        p.append(self.f)
        return np.hstack(p)

    def compile_module(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'equidistant_cam.cuh'))
        fnl.append(os.path.join(dn, 'omni_cam.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        cuda_source = cuda_source.replace('OmniCameraModel', 'EquidistantCameraModel')

        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def check_alignment(self):
        offset_array = cp.empty(3, dtype=cp.int32)
        gpu_func = self.gpu_module.get_function('getIntrinsicSize')
        sz_block = 1, 1
        sz_grid = 1, 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                offset_array,
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        offset_array = offset_array.get()
        o = 0
        assert o == offset_array[1]
        o += 2
        assert o == offset_array[2]

    def setup_module(self):
        if self.gpu_module is None:
            self.compile_module()
        self.check_alignment()
        upload_constant(self.gpu_module, self.get_intrincs_array(), 'g_K_ref')

    def get_rays(self, sz = None):
        if sz is None:
            sz = self.sz
        rays = cp.empty((sz[0], sz[1], 3), dtype=cp.float32)
        assert rays.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('getRays')
        sz_block = 32, 32
        sz_grid = math.ceil(rays.shape[1] / sz_block[0]), math.ceil(rays.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                rays,
                rays.shape[0],
                rays.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.rays = rays

    def get_thetas(self):
        if self.rays is None:
            self.get_rays()

        thetas = cp.empty_like(self.rays)
        assert thetas.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('getThetas')
        sz_block = 32, 32
        sz_grid = math.ceil(thetas.shape[1] / sz_block[0]), math.ceil(thetas.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                thetas,
                self.rays,
                thetas.shape[0],
                thetas.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.thetas = thetas

    def create_default_mask(self, sz = None):
        if sz is None:
            sz = self.sz
        mask = cp.full((sz[0], sz[1]), 1, dtype=cp.uint8)
        to_mask = texture.create_texture_object(cp.asarray(mask),
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)
        self.mask = mask
        self.to_mask = to_mask

    def set_mask(self, mask):
        to_mask = texture.create_texture_object(cp.asarray(mask),
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)
        self.mask = mask
        self.to_mask = to_mask

    def get_xyz_from_distance(self, D):
        if self.rays is None:
            self.get_rays(D.shape[:2])
        if self.mask is None:
            self.create_default_mask(D.shape[:2])
        assert D.shape == self.rays.shape[:2]

        xyz = cp.empty_like(self.rays)
        assert xyz.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('getXYZFromDistance')
        sz_block = 32, 32
        sz_grid = math.ceil(xyz.shape[1] / sz_block[0]), math.ceil(xyz.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                xyz,
                self.rays,
                D,
                self.to_mask,
                xyz.shape[0],
                xyz.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return xyz
