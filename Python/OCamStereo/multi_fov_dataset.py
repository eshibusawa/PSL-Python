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
import numpy as np
import cv2

from functools import lru_cache

def get_rotation_from_quaternion(q):
    q0_2 = 2 * q[0]
    q1_2 = 2 * q[1]
    q2_2 = 2 * q[2]
    q03_2 = q0_2 * q[3]
    q13_2 = q1_2 * q[3]
    q23_2 = q2_2 * q[3]
    q00_2 = q0_2 * q[0]
    q01_2 = q0_2 * q[1]
    q02_2 = q0_2 * q[2]
    q11_2 = q1_2 * q[1]
    q12_2 = q1_2 * q[2]
    q22_2 = q2_2 * q[2]
    R = np.empty((3, 3), dtype=q.dtype)
    R[0,0] = 1 - (q11_2 + q22_2)
    R[0,1] = q01_2 - q23_2
    R[0,2] = q02_2 + q13_2
    R[1,0] = q01_2 + q23_2
    R[1,1] = 1 - (q00_2 + q22_2)
    R[1,2] = q12_2 - q03_2
    R[2,0] = q02_2 - q13_2
    R[2,1] = q12_2 + q03_2
    R[2,2] = 1 - (q00_2 + q11_2)
    return R

class dataset_loader():
    def __init__(self, base_path):
        self.base_path = base_path
        self.info_path = os.path.join(base_path, 'info')
        self.data_path = os.path.join(base_path, 'data')

    def get_calibration(self):
        fpfn = os.path.join(self.info_path, 'intrinsics.txt')
        with open(fpfn) as f:
            l = f.readline().split()
        p = np.array([float(i) for i in l], dtype=np.float32)
        return p

    @staticmethod
    @lru_cache(maxsize=None)
    def get_row_from_id(fpfn, id):
        id = str(id)
        try:
            with open(fpfn, mode='r') as f:
                while True:
                    l = f.readline().split()
                    if not l:
                        raise ValueError
                    if (l[0] == id):
                        break
        except ValueError:
            return None
        return l

    @staticmethod
    def load_pose(fpfn, id):
        l = dataset_loader.get_row_from_id(fpfn, id)
        pose = np.array([float(i) for i in l], dtype=np.float32)
        # Xw = R*Xc + t
        t = pose[1:4][::,np.newaxis]
        R = get_rotation_from_quaternion( pose[4:])
        # Xc = R2*Xw + t2
        R2 = R.T
        t2 = -np.dot(R.T, t)
        return R2, t2

    def load_deptharray(self, id):
        fpfn_depth_list = os.path.join(self.info_path, 'depthmaps.txt')
        l = dataset_loader.get_row_from_id(fpfn_depth_list, id)
        fpfn = os.path.join(self.data_path, l[1])
        d = np.loadtxt(fpfn, dtype=np.float32)
        return d

    def load_image(self, id):
        fpfn_image_list = os.path.join(self.info_path, 'images.txt')
        l = dataset_loader.get_row_from_id(fpfn_image_list, id)
        fpfn = os.path.join(self.data_path, l[2])
        img = cv2.imread(fpfn)
        return img, float(l[1])

    def load_data(self, id):
        fpfn_pose = os.path.join(self.info_path, 'groundtruth.txt')
        R, t = dataset_loader.load_pose(fpfn_pose, id)
        d = self.load_deptharray(id)
        img, timestamp = self.load_image(id)
        return img, timestamp, R, t, d.reshape(img.shape[:2])
