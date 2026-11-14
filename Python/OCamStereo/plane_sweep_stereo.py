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

import cv2
import numpy as np
import cupy as cp

import add_path
from psl import plane_sweep
from util_cuda import upload_constant
import texture

class plane_sweep_stereo():
    def __init__(self):
        self.match_window_width = 9
        self.match_window_height = 9
        self.near_z = 0.4
        self.far_z = 2
        self.num_planes = 64
        self.plane_generation_mode_ = {'uniform_depth':0, 'uniform_disparity':1}
        self.plane_generation_mode = self.plane_generation_mode_['uniform_disparity']
        self.gpu_module = None
        self.rays = None
        self.mask = None
        self.cams = None
        self.use_mask_indexed = False
        self.use_mask_table = False
        self.use_ocam = True

    def setup_calibration(self, cams, Rs, ts):
        self.cams = cams
        R = np.dot(Rs[1], Rs[0].T)
        t = -np.dot(R, ts[0]) + ts[1]
        self.R = R
        self.t = -t

    def compile_module(self):
        assert (self.match_window_width % 2) == 1
        assert (self.match_window_height % 2) == 1
        assert self.cams is not None

        dn = os.path.dirname(__file__)
        dn_psl = os.path.join(os.path.dirname(dn), 'ext')

        fnl = list()
        fnl.append(os.path.join(dn_psl, 'warping.cuh'))
        fnl.append(os.path.join(dn_psl, 'depth.cuh'))
        if self.use_ocam:
            fnl.append(os.path.join(dn, 'omni_cam.cuh'))
        else:
            fnl.append(os.path.join(dn, 'equidistant_cam.cuh'))
        fnl.append(os.path.join(dn, 'omni_cam_stereo.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        if self.use_ocam:
            cuda_source = cuda_source.replace('OC_POLYNOMIAL_COEF_NUM', str(self.cams[1].polynomial.shape[0]))
            cuda_source = cuda_source.replace('OC_INVERSE_POLYNOMIAL_COEF_NUM', str(self.cams[1].inverse_polynomial.shape[0]))
        else:
            cuda_source = cuda_source.replace('OmniCameraModel', 'EquidistantCameraModel')

        cuda_source = cuda_source.replace('MATCH_WINDOW_WIDTH', str(self.match_window_width))
        cuda_source = cuda_source.replace('MATCH_WINDOW_HEIGHT', str(self.match_window_height))
        cuda_source = cuda_source.replace('NUM_PLANES', str(self.num_planes))

        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def setup_transformations(self):
        # upload intrinsics
        upload_constant(self.gpu_module, self.cams[1].get_intrincs_array(), 'g_K_other')

        # setup homographies
        ds = plane_sweep.generate_depth(self.near_z, self.far_z, self.num_planes, self.plane_generation_mode)
        ds = ds[::-1]
        n = np.array([0, 0, -1], dtype=np.float32)[:,np.newaxis]
        Hs = list()
        for d in ds:
            H = self.R + np.dot(self.t, n.T)/d
            Hs.append(H)

        # upload homographies
        upload_constant(self.gpu_module, np.array(Hs).reshape(-1), 'g_H')
        # upload planes
        Ps = np.empty((ds.shape[0], 4), dtype=np.float32)
        Ps[:,0:3] = n.T
        Ps[:,3] = ds.T
        upload_constant(self.gpu_module, Ps.reshape(-1), 'g_nd')

    def get_table(self):
        # compute rays
        if self.rays is None:
            self.cams[0].get_rays()
            self.rays = self.cams[0].rays
        if self.to_mask is None:
            return None

        table = cp.empty((self.rays.shape[0], self.rays.shape[1], self.num_planes, 2), dtype=cp.int16)
        assert table.flags.c_contiguous
        sz_block = 32, 32
        sz_grid = math.ceil(table.shape[1] / sz_block[0]), math.ceil(table.shape[0] / sz_block[1])
        gpufunc = self.gpu_module.get_function('getTable')
        gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                table,
                self.rays,
                self.to_mask,
                table.shape[0],
                table.shape[1],
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.table = table
        return table

    def set_mask(self, mask):
        self.mask = mask
        to_mask = texture.create_texture_object(mask,
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords=0)
        self.to_mask = to_mask
        self.use_mask_indexed = False
        self.use_mask_table = False

    def set_mask_indexed(self, mask):
        self.set_mask(mask)
        index = np.empty((mask.shape[0], mask.shape[1], 2), dtype=np.int16)
        index[:,:,0] = np.arange(0, mask.shape[1])[np.newaxis,:]
        index[:,:,1] = np.arange(0, mask.shape[0])[:,np.newaxis]

        index1d = index[mask != 0]
        sz1 = math.ceil(math.sqrt(index1d.shape[0] / self.num_planes) / 32) * 32
        sz0 = math.ceil(index1d.shape[0] / sz1)
        index1d_ceiled = np.empty((sz0 * sz1, 2), dtype=np.int16)
        index1d_ceiled[0:index1d.shape[0],:] = index1d
        index1d_ceiled[index1d.shape[0]:,:] = -1
        mask_indexed = index1d_ceiled.reshape(sz0, sz1, 2)
        to_mask_indexed = texture.create_texture_object(mask_indexed,
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords=0)
        self.to_mask_indexed = to_mask_indexed
        self.mask_indexed = mask_indexed
        self.use_mask_indexed = True
        self.use_mask_table = False

    def set_mask_indexed_table(self, mask):
        # compute rays
        if self.rays is None:
            self.cams[0].get_rays()
            self.rays = self.cams[0].rays

        self.set_mask_indexed(mask)
        mask_length = cp.zeros(mask.shape, dtype=cp.uint32)
        assert mask_length.flags.c_contiguous
        sz_block = 32, 32
        sz_grid = math.ceil(self.mask_indexed.shape[1] / sz_block[0]), math.ceil(self.mask_indexed.shape[0] / sz_block[1])
        gpufunc = self.gpu_module.get_function('getMaskFromIndexed')
        gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                mask_length,
                self.to_mask_indexed,
                mask_length.shape[0],
                mask_length.shape[1],
                self.mask_indexed.shape[0],
                self.mask_indexed.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.to_mask_length = texture.create_texture_object(mask_length,
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords=0)

        table = cp.empty((self.mask_indexed.shape[0], self.mask_indexed.shape[1] * self.num_planes, 2), dtype=cp.int16)
        assert table.flags.c_contiguous
        sz_block = 64, 16
        sz_grid = math.ceil(self.mask_indexed.shape[0] * self.mask_indexed.shape[1]/ sz_block[0]), math.ceil(self.num_planes / sz_block[1])
        gpufunc = self.gpu_module.get_function('getTableFromIndexed')
        gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                table,
                self.rays,
                self.to_mask_indexed,
                mask_length.shape[0],
                mask_length.shape[1],
                self.mask_indexed.shape[0],
                self.mask_indexed.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.to_table = texture.create_texture_object(table,
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords=0)
        self.use_mask_indexed = False
        self.use_mask_table = True

    def get_xyz(self, img_ref, img_other):
        # compute rays
        if self.rays is None:
            self.cams[0].get_rays()
            self.rays = self.cams[0].rays
        if self.mask is None:
            self.cams[0].create_default_mask()
            self.mask = self.cams[0].mask
            self.to_mask = self.cams[0].to_mask

        # upload images
        to_ref = texture.create_texture_object(img_ref,
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords=0)
        to_other = texture.create_texture_object(img_other,
            filterMode=cp.cuda.runtime.cudaFilterModeLinear,
            readMode=cp.cuda.runtime.cudaReadModeNormalizedFloat,
            normalizedCoords=0)

        # compute point cloud
        xyz = cp.zeros((img_ref.shape[0], img_ref.shape[1], 3), dtype=cp.float32)
        if self.use_mask_table:
            sz_block = 32, 8
            sz_grid = math.ceil(self.mask_indexed.shape[1] / sz_block[0]), math.ceil(self.mask_indexed.shape[0] / sz_block[1])
            gpufunc = self.gpu_module.get_function('getXYZMaskIndexedTable')
            gpufunc(
                block=sz_block,
                grid=sz_grid,
                args=(
                    xyz,
                    self.rays,
                    to_ref,
                    to_other,
                    self.to_mask_indexed,
                    self.to_mask_length,
                    self.to_table,
                    xyz.shape[0],
                    xyz.shape[1],
                    self.mask_indexed.shape[0],
                    self.mask_indexed.shape[1]
                )
            )
        elif self.use_mask_indexed:
            sz_block = 16, 16
            sz_grid = math.ceil(self.mask_indexed.shape[1] / sz_block[0]), math.ceil(self.mask_indexed.shape[0] / sz_block[1])
            gpufunc = self.gpu_module.get_function('getXYZMaskIndexed')
            gpufunc(
                block=sz_block,
                grid=sz_grid,
                args=(
                    xyz,
                    self.rays,
                    to_ref,
                    to_other,
                    self.to_mask,
                    self.to_mask_indexed,
                    xyz.shape[0],
                    xyz.shape[1],
                    self.mask_indexed.shape[0],
                    self.mask_indexed.shape[1]
                )
            )
        else:
            sz_block = 16, 16
            sz_grid = math.ceil(xyz.shape[1] / sz_block[0]), math.ceil(xyz.shape[0] / sz_block[1])
            gpufunc = self.gpu_module.get_function('getXYZ')
            gpufunc(
                block=sz_block,
                grid=sz_grid,
                args=(
                    xyz,
                    self.rays,
                    to_ref,
                    to_other,
                    self.to_mask,
                    xyz.shape[0],
                    xyz.shape[1]
                )
            )
        cp.cuda.runtime.deviceSynchronize()

        return xyz

def draw_trajectories(imgs, table_gpu, points):
    if len(imgs[0].shape) == 3:
        imgs_draw = np.copy(imgs[0]), np.copy(imgs[1])
    else:
        imgs_draw = cv2.cvtColor(imgs[0], cv2.COLOR_GRAY2BGR), cv2.cvtColor(imgs[1], cv2.COLOR_GRAY2BGR)

    for p in points:
        cv2.circle(imgs_draw[0], (int(p[0]), int(p[1])), 5, (255, 255, 0))
        trajectory = table_gpu[int(p[1]), int(p[0]), :, :].get()
        pt_prev = None
        for pt in trajectory:
            if (pt[0] < 0) or (pt[1] < 0):
                pt_prev = None
                continue
            pt = int(pt[0]/16), int(pt[1]/16)
            # cv2.circle(imgs_draw[1], (pt[0], pt[1]), 15, (255, 255, 0), 1)
            if pt_prev is not None:
                cv2.line(imgs_draw[1], (pt_prev[0], pt_prev[1]), (pt[0], pt[1]), (127, 127, 0), 1)
            pt_prev = pt

    return imgs_draw
