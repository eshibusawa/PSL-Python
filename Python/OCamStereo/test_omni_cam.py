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

from unittest import TestCase
from nose.tools import ok_

import numpy as np
import cupy as cp

import omni_cam
import add_path
from util_cuda import upload_constant

class OmniCamTestCase(TestCase):
    def setUp(self):
        self.eps_projection = 1E-4
        self.eps_back_projection = 1E-5

        # https://github.com/zhangzichao/omni_cam/blob/master/test/ocam_param.txt
        reference_param = np.array([640, 480, -69.6915, 0.000000e+00, 5.4772e-4, 2.1371e-5, -8.7523e-9, 320.0, 240.0, 1.0, 0.0, 0.0, 142.7468, 104.8486, 7.3973, 17.4581, 12.6308, -4.3751, 6.9093, 10.9703, -0.6053, -3.9119, -1.0675, 0.0], dtype=np.float32)
        oc = omni_cam.OmniCam(reference_param)

        dn = os.path.dirname(os.path.realpath(__file__))
        dn_psl = os.path.join(os.path.dirname(dn), 'ext')
        sources = list()
        sources.append(os.path.join(dn, 'omni_cam.cuh'))
        sources.append(os.path.join(dn_psl, 'project_unproject_unit_test.cu'))
        cuda_source = None
        for fpfn in sources:
            # load raw kernel
            with open(fpfn, 'r') as f:
                if cuda_source is None:
                    cuda_source = f.read()
                else:
                    cuda_source += f.read()
        cuda_source = cuda_source.replace('HEIGHT', str(reference_param[1]))
        cuda_source = cuda_source.replace('WIDTH', str(reference_param[0]))
        cuda_source = cuda_source.replace('CAMERA_MODEL', 'OmniCameraModel')
        cuda_source = cuda_source.replace('OC_POLYNOMIAL_COEF_NUM', str(oc.polynomial.shape[0]))
        cuda_source = cuda_source.replace('OC_INVERSE_POLYNOMIAL_COEF_NUM', str(oc.inverse_polynomial.shape[0]))
        cuda_source = cuda_source.replace('__host__', '__device__')
        self.module = cp.RawModule(code=cuda_source)
        self.module.compile()

        # concat intrinsics
        self.sz = oc.sz
        self.intrinsics = oc.get_intrincs_array()

        # upload intrinsics
        upload_constant(self.module, self.intrinsics, 'g_K')

    def tearDown(self):
        pass

    def unproject_project_test(self):
        xyz_gpu = cp.zeros((self.sz[0], self.sz[1], 3), dtype=cp.float32)
        sz_block = 32, 32
        sz_grid = math.ceil(self.sz[1] / sz_block[0]), math.ceil(self.sz[0] / sz_block[1])
        test_unproject_gpufunc = self.module.get_function("unprojectTest")
        test_unproject_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                xyz_gpu,
            )
        )
        xyz0_ref = np.array([0.733011271294813, 0.549758453471110, 0.400574735838165])
        xyz0 = xyz_gpu[300, 400, :]
        err = np.abs(xyz0_ref - xyz0.get())
        ok_(np.max(err) < self.eps_back_projection)

        xyz_gpu = cp.array([1.0, 1.0, -1.0], dtype=cp.float32)[cp.newaxis, cp.newaxis, :]
        xy_gpu = cp.zeros((xyz_gpu.shape[0], xyz_gpu.shape[1], 2), dtype=cp.float32)
        sz_block = 1, 1
        sz_grid = math.ceil(xyz_gpu.shape[1] / sz_block[0]), math.ceil(xyz_gpu.shape[0] / sz_block[1])
        test_project_gpufunc = self.module.get_function("projectTest")
        test_project_gpufunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                xyz_gpu,
                xy_gpu,
            )
        )
        xy_ref = np.array([4.729118411664447e+02, 3.929118411664447e+02], dtype=np.float32)[np.newaxis, np.newaxis, :]
        err = np.abs(xy_ref - xy_gpu.get())
        ok_(np.max(err) < self.eps_projection)
