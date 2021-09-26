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
import numpy as np
import cv2

from unified_camera import unified_camera

class psl_data():
    def __init__(self, ds_path):
        self.ds_path = ds_path

    def read_calibration(self):
        fn = 'calib.txt'
        fpfn_calib = os.path.join(self.ds_path, fn)
        with open(fpfn_calib, 'r') as f:
            line = f.readline().split(' ')
            K = np.eye(3)
            K[0,0] = line[0]
            K[0,1] = line[1]
            K[0,2] = line[2]
            K[1,1] = line[3]
            K[1,2] = line[4]
            xi = float(line[5])
            ks = np.array([line[6], line[7]], dtype=np.float)
            ps = np.array([line[8], line[9]], dtype=np.float)

            line = f.readline().split(' ')
            R = np.empty((3, 3), dtype=np.float)
            C = np.empty((3, 1), dtype=np.float)
            for j in range(0, 3):
                line = f.readline().split(' ')
                R[j, 0] = line[0]
                R[j, 1] = line[1]
                R[j, 2] = line[2]
                C[j, 0] = line[3]

        self.K = K
        self.R = R
        self.C = C
        self.xi = xi
        self.ks = ks
        self.ps = ps
        return K, R, C, xi, ks, ps

    def read_system_poses(self):
        fn = 'system_poses.txt'
        fpfn_calib = os.path.join(self.ds_path, fn)
        with open(fpfn_calib, 'r') as f:
            lines = f.readlines()

        R = list()
        T = list()
        t = list()
        for line in lines:
            if line[0] == '#':
                continue
            l = line.split(' ')

            t.append(l[0])
            Rz, _ = cv2.Rodrigues(np.array([0, 0, float(l[1])]))
            Ry, _ = cv2.Rodrigues(np.array([0, float(l[2]), 0]))
            Rx, _ = cv2.Rodrigues(np.array([float(l[3]), 0, 0]))
            R.append(np.dot(Rz, np.dot(Ry, Rx)))
            T.append(np.array([l[4], l[5], l[6]], dtype=np.float))

        system_R = np.array(R)
        system_T = np.array(T)[:,:,np.newaxis]
        timestamps = np.array(t, np.uint64)
        self.system_R = system_R
        self.system_T = system_T
        self.timestamps = timestamps
        return system_R, system_T, timestamps

    def get_world_to_camera_pose(self):
        # Xc = Rc * Xg - Tc
        self.Rs = np.empty((self.system_R.shape[0], 3, 3), dtype=np.float)
        self.Ts = np.empty((self.system_T.shape[0], 3, 1), dtype=np.float)
        for R, T, k in zip(self.system_R, self.system_T, range(0, self.system_R.shape[0])):
            self.Rs[k] = np.dot(self.R.T, R.T)
            self.Ts[k] = -np.dot(self.Rs[k], T) - np.dot(self.R.T, self.C)
        return self.Rs, self.Ts

    def get_relative_pose(self, ref, other):
        R12 = np.dot(self.Rs[other], self.Rs[ref].T)
        t12 = np.dot(R12, self.Ts[ref]) - self.Ts[other]
        return R12, t12

    def read_image_file_list(self):
        fn = 'images.txt'
        fpfn_calib = os.path.join(self.ds_path, fn)
        with open(fpfn_calib, 'r') as f:
            lines = f.readlines()

            image_file_list = list()
            for l in lines:
                image_file_list.append(l.replace('\n',''))
        self.image_file_list = image_file_list
        return image_file_list

    def read_image(self, id, is_gray = True):
        fn = self.image_file_list[id]
        t = fn[0:-4][10:]
        if int(t) != self.timestamps[id]:
            return None
        fpfn = os.path.join(self.ds_path, fn)

        if is_gray:
            read_type = cv2.IMREAD_GRAYSCALE
        else:
            read_type = cv2.IMREAD_COLOR

        return cv2.imread(fpfn, read_type)

    def get_unified_camera(self):
        return unified_camera(self.K, self.xi)
