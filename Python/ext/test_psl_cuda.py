# This file is part of PSL-Python.
# Copyright (c) 2021, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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

from unittest import TestCase
from nose.tools import ok_

import numpy as np
import cupy as cp
import torch

import add_path
from unified_camera import unified_camera as ucm
from psl import plane_sweep as psl

class UCMTestCase(TestCase):
    def setUp(self):
        self.eps = 1E-6
        self.eps_reproj = 5E-4
        self.intrinsics = np.array([447.77409157, 447.37213065, 324.56178482, 199.10951743, 1.1584422115726898], dtype=np.float32)
        self.sz = 400, 640
        self.num_planes = 16

        K = np.eye(3, dtype=np.float32)
        K[0,0] = self.intrinsics[0]
        K[1,1] = self.intrinsics[1]
        K[0,2] = self.intrinsics[2]
        K[1,2] = self.intrinsics[3]
        self.ucm_ref = ucm(K, self.intrinsics[4])
        self.ucm_other = self.ucm_ref

        self.R_ref = np.array([[-0.17524842, -0.9828536, 0.05733053],
            [-0.67348578, 0.07720413, -0.73515742],
            [ 0.71812596, -0.16744648, -0.67546783]], dtype=np.float32)
        self.R_other = np.array([[-0.17543485, -0.98288089, 0.05628303],
        [-0.67267369, 0.07793009, -0.73582404],
        [ 0.71884124, -0.16694929, -0.67482976]], dtype=np.float32)
        self.T_ref = np.array([[-48.98402717],
            [-16.88003147],
            [ 14.19639506]], dtype=np.float32)
        self.T_other = np.array([[-49.20218616],
            [-16.82320663],
            [ 14.2248999 ]], dtype=np.float32)

        dn = os.path.dirname(os.path.realpath(__file__))
        sources = list()
        sources.append('unified_camera.cuh')
        sources.append('warping.cuh')
        sources.append('project_unproject_unit_test.cu')
        sources.append('warping_unit_test.cu')
        cuda_source = None
        for fn in sources:
            fpfn = os.path.join(dn, fn)
            # load raw kernel
            with open(fpfn, 'r') as f:
                if cuda_source is None:
                    cuda_source = f.read()
                else:
                    cuda_source += f.read()
        cuda_source = cuda_source.replace("HEIGHT", str(self.sz[0]))
        cuda_source = cuda_source.replace("WIDTH", str(self.sz[1]))
        cuda_source = cuda_source.replace("NUM_PLANES", str(self.num_planes))
        cuda_source = cuda_source.replace("CAMERA_MODEL", "UnifiedCameraModel")
        cuda_source = cuda_source.replace("__host__", "__device__")
        self.module = cp.RawModule(code=cuda_source)

    def tearDown(self):
        pass

    def unproject_project_test(self):
        xy_ref = np.empty((2, self.sz[0], self.sz[1]), dtype=np.float32)
        xy_ref[0, :,:] = np.arange(0, self.sz[1])[np.newaxis,:]
        xy_ref[1, :,:] = np.arange(0, self.sz[0])[:,np.newaxis]
        xyz_ref = self.ucm_ref.unproject(xy_ref.reshape(2, -1)) # xyz[3, h*w]

        # upload intrinsics
        K_ptr = self.module.get_global("g_K")
        intrinsics_gpu = cp.ndarray(self.intrinsics.shape, cp.float32, K_ptr)
        intrinsics_gpu[:] = cp.array(self.intrinsics)

        xyz_gpu = torch.zeros((self.sz[0], self.sz[1], 3), dtype=torch.float32, device='cuda')
        assert xyz_gpu.is_contiguous()
        sz_block = 32, 32
        sz_grid = math.ceil(self.sz[1] / sz_block[1]), math.ceil(self.sz[0] / sz_block[0])
        # call the kernel
        test_unproject_gpufunc = self.module.get_function("unprojectTest")
        test_unproject_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                xyz_gpu.data_ptr(),
            )
        )
        # download the result
        xyz_t = xyz_gpu.cpu().numpy() # xyz[h, w, 3]
        xyz = xyz_t.transpose(2, 0, 1).reshape(3, -1)
        # evaluate error
        err = np.abs(xyz - xyz_ref)
        ok_(np.max(err) < self.eps)

        xy_gpu = torch.zeros((self.sz[0], self.sz[1], 2), dtype=torch.float32, device='cuda')
        assert xy_gpu.is_contiguous()
        # call the kernel
        test_project_gpufunc = self.module.get_function("projectTest")
        test_project_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                xyz_gpu.data_ptr(),
                xy_gpu.data_ptr(),
            )
        )
        # download the result
        xy_t = xy_gpu.cpu().numpy() # xy[h, w, 2]
        xy = xy_t.transpose(2, 0, 1)
        # evaluate error
        err = np.abs(xy - xy_ref)
        ok_(np.max(err) < self.eps_reproj)

    def compute_warping_map_test(self):
        # compute reference map
        ps = psl()
        ps.num_planes = self.num_planes
        ps.add_image(self.ucm_ref, self.R_ref, self.T_ref, None)
        ps.add_image(self.ucm_ref, self.R_ref, self.T_other, None)
        ps.generate_planes()
        rays = self.ucm_ref.unproject_rays(self.sz).astype(np.float32)
        # relative pose
        R = np.dot(self.R_other, self.R_ref.T)
        t = np.dot(R, self.T_ref) - self.T_other
        ps.transform_planes(R, t)
        xy_ref = list()
        for l in range(0, self.num_planes):
            xyz = np.dot(ps.Hs[l], rays)

            xy_ref.append(self.ucm_other.project(xyz).reshape(2,*self.sz).astype(np.float32))

        # upload intrinsics
        K_ptr_ref = self.module.get_global("g_K_ref")
        intrinsics_gpu_ref = cp.ndarray(self.intrinsics.shape, cp.float32, K_ptr_ref)
        intrinsics_gpu_ref[:] = cp.array(self.intrinsics)
        K_ptr_other = self.module.get_global("g_K_other")
        intrinsics_gpu_other = cp.ndarray(self.intrinsics.shape, cp.float32, K_ptr_other)
        intrinsics_gpu_other[:] = cp.array(self.intrinsics)
        # upload extrinsics
        R_ptr = self.module.get_global("g_R")
        vR = R.reshape(-1)
        R_gpu = cp.ndarray(vR.shape, cp.float32, R_ptr)
        R_gpu[:] = cp.array(vR)
        t_ptr = self.module.get_global("g_t")
        vt = t.reshape(-1)
        t_gpu = cp.ndarray(vt.shape, cp.float32, t_ptr)
        t_gpu[:] = cp.array(vt)
        # upload planes
        planes_ptr = self.module.get_global("g_planes")
        vplanes = ps.planes.T.reshape(-1)
        planes_gpu = cp.ndarray(vplanes.shape, cp.float32, planes_ptr)
        planes_gpu[:] = cp.array(vplanes)

        xy_gpu = torch.zeros((self.num_planes, self.sz[0], self.sz[1], 2), dtype=torch.float32, device='cuda')
        assert xy_gpu.is_contiguous()
        sz_block = 64, 16
        sz_grid = math.ceil((self.sz[0] * self.sz[1]) / sz_block[0]), math.ceil(self.num_planes / sz_block[1])
        # call the kernel
        test_warping_gpufunc = self.module.get_function("warpingTest")
        test_warping_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                xy_gpu.data_ptr(),
            )
        )
        # download the result
        xy_t = xy_gpu.cpu().numpy() # xy[p, h, w, 2]
        xy = xy_t.transpose(0, 3, 1, 2)
        # evaluate error
        err = np.abs(xy - np.array(xy_ref))
        ok_(np.max(err) < self.eps_reproj)

class DepthEstimationTestCase(TestCase):
    def setUp(self):
        self.eps = 1E-4
        self.intrinsics = np.array([447.77409157, 447.37213065, 324.56178482, 199.10951743, 1.1584422115726898], dtype=np.float32)
        self.sz = 400, 640
        self.num_planes = 16

        K = np.eye(3, dtype=np.float32)
        K[0,0] = self.intrinsics[0]
        K[1,1] = self.intrinsics[1]
        K[0,2] = self.intrinsics[2]
        K[1,2] = self.intrinsics[3]
        self.ucm_ref = ucm(K, self.intrinsics[4])

        dn = os.path.dirname(os.path.realpath(__file__))
        sources = list()
        sources.append('depth.cuh')
        sources.append('depth_unit_test.cu')
        cuda_source = None
        for fn in sources:
            fpfn = os.path.join(dn, fn)
            # load raw kernel
            with open(fpfn, 'r') as f:
                if cuda_source is None:
                    cuda_source = f.read()
                else:
                    cuda_source += f.read()
        cuda_source = cuda_source.replace("HEIGHT", str(self.sz[0]))
        cuda_source = cuda_source.replace("WIDTH", str(self.sz[1]))
        cuda_source = cuda_source.replace("NUM_PLANES", str(self.num_planes))
        cuda_source = cuda_source.replace("__host__", "__device__")
        self.module = cp.RawModule(code=cuda_source)

    def tearDown(self):
        pass

    def depth_estimation_test(self):
        ps = psl()
        ps.num_planes = self.num_planes
        ps.generate_planes()
        ps.planes = ps.planes.astype(np.float32)
        ps.add_image(self.ucm_ref, None, None, None)
        ps.rays = self.ucm_ref.unproject_rays(self.sz).astype(np.float32)
        ps.CV = (np.random.rand(*self.sz, self.num_planes) + 1).astype(np.float32)
        indices = (np.random.rand(*self.sz) * self.num_planes).astype(np.int64)
        np.put_along_axis(ps.CV, indices[:,:,np.newaxis], 0, axis=2)

        # upload planes
        planes_ptr = self.module.get_global("g_planes")
        vplanes = ps.planes.T.reshape(-1)
        planes_gpu = cp.ndarray(vplanes.shape, cp.float32, planes_ptr)
        planes_gpu[:] = cp.array(vplanes)

        planes_has_neighbor_ptr = self.module.get_global("g_planesHasNeighbor")
        planes_has_neighbor_gpu = cp.ndarray(ps.planes_has_neighbor.shape, cp.bool, planes_has_neighbor_ptr)
        planes_has_neighbor_gpu[:] = cp.array(ps.planes_has_neighbor)

        D_gpu = torch.zeros((self.sz[0], self.sz[1]), dtype=torch.float32, device='cuda')
        assert D_gpu.is_contiguous()
        sz_block = 32, 32
        sz_grid = math.ceil(self.sz[1] / sz_block[1]), math.ceil(self.sz[0] / sz_block[0])

        # upload indices
        indices_gpu = torch.tensor(indices, device=torch.device('cuda'))
        assert indices_gpu.is_contiguous()
        rays_t = ps.rays.reshape(3, *self.sz)
        rays_gpu = torch.tensor(rays_t.transpose(1, 2, 0), device=torch.device('cuda')).contiguous()
        assert rays_gpu.is_contiguous()

        # call the kernel
        test_depth_gpufunc = self.module.get_function("getDepthTest")
        test_depth_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                rays_gpu.data_ptr(),
                indices_gpu.data_ptr(),
                D_gpu.data_ptr(),
            )
        )
        ps.sub_pixel_enabled = False
        D_ref = ps.get_depth_from_planes(indices)
        D = D_gpu.to('cpu').numpy()
        err = np.abs(D - D_ref)
        ok_(np.max(err) < self.eps)

        # upload CV
        CV_gpu = torch.tensor(ps.CV.transpose(2, 0, 1), device=torch.device('cuda')).contiguous()
        assert CV_gpu.is_contiguous()

        # call the kernel
        test_depth_gpufunc = self.module.get_function("getDepthWithSubpixelDirectTest")
        test_depth_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                rays_gpu.data_ptr(),
                indices_gpu.data_ptr(),
                CV_gpu.data_ptr(),
                D_gpu.data_ptr(),
            )
        )
        ps.sub_pixel_enabled = True
        ps.sub_pixel_interpolation_mode = ps.sub_pixel_interpolation_mode_['direct']
        D_ref = ps.get_depth_from_planes(indices)
        D = D_gpu.to('cpu').numpy()
        err = np.abs(D - D_ref)
        ok_(np.max(err) < self.eps)

        # call the kernel
        test_depth_gpufunc = self.module.get_function("getDepthWithSubpixelInverseTest")
        test_depth_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                rays_gpu.data_ptr(),
                indices_gpu.data_ptr(),
                CV_gpu.data_ptr(),
                D_gpu.data_ptr(),
            )
        )
        ps.sub_pixel_enabled = True
        ps.sub_pixel_interpolation_mode = ps.sub_pixel_interpolation_mode_['inverse']
        D_ref = ps.get_depth_from_planes(indices)
        D = D_gpu.to('cpu').numpy()
        err = np.abs(D - D_ref)
        ok_(np.max(err) < self.eps)
