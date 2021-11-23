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

import cv2

import psl as py_psl
from psl_data import psl_data
from unified_camera import unified_camera as ucm

if __name__ == '__main__':
    psl_path = '/path/to/PSL'
    ds_path = psl_path + '/data/fisheyeCamera/left/'
    use_gpu = True

    psld = psl_data(ds_path)
    K, R, C, xi, ks, ps = psld.read_calibration()
    psld.read_system_poses()
    psld.read_image_file_list()
    uc = psld.get_unified_camera()
    Rs, Ts = psld.get_world_to_camera_pose()
    img = psld.read_image(0)
    ud_map, K_ud = uc.get_undistort_map((img.shape[1], img.shape[0]), (0.5, 1), ks, ps)
    uc_new = ucm(K_ud, xi)

    imgs = list()
    for k in range(0, 5):
        img = psld.read_image(k)
        img_ud = cv2.remap(img, ud_map[0], ud_map[1], cv2.INTER_LINEAR, None, cv2.BORDER_WRAP)
        imgs.append(img_ud)

    pyps = py_psl.plane_sweep()
    pyps.num_planes = 256
    if use_gpu:
        pyps.use_gpu = True
        pyps.sub_pixel_enabled = False
    else:
        pyps.use_gpu = False
        pyps.sub_pixel_enabled = True

    for k in range(0, len(imgs)):
        pyps.add_image(uc_new, Rs[k], Ts[k], imgs[k])

    D = pyps.get_depth(0)
    Dimg = py_psl.depth_to_colormap(D, pyps.near_z, pyps.far_z)
    cv2.imwrite('depth.png', Dimg)
