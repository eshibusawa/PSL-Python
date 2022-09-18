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

import numpy as np
import cv2

import cupy as cp
import open3d as o3d

import add_path
from omni_cam import OmniCam
from multi_fov_dataset import dataset_loader
from plane_sweep_stereo import plane_sweep_stereo
from colormap import depth_to_colormap

loader = dataset_loader('/path/to/vfr_fisheye')
id_img_ref = 1
id_img_other = 3
show_point_cloud = True

# reference camera
p = loader.get_calibration()
ocam_ref = OmniCam(p)
img_ref, timestamp_ref, R_ref, t_ref, D_ref = loader.load_data(id_img_ref)
ocam_ref.setup_module()
# other camera (motion stereo)
ocam_other = ocam_ref
img_other, timestamp_other, R_other, t_other, D_other = loader.load_data(id_img_other)

xyz_ref = ocam_ref.get_xyz_from_distance(cp.array(D_ref)).get()
cv2.imwrite('img_ref.png', img_ref)

if show_point_cloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_ref.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.fliplr(img_ref.reshape(-1, 3)/255))
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud('vfr_fisheye_{:04d}.pcd'.format(id_img_ref), pcd)

ps = plane_sweep_stereo()
ps.match_window_width = 9
ps.match_window_height = 9
ps.near_z = 0.4
ps.far_z = 2
ps.num_planes = 64

ps.setup_calibration((ocam_ref, ocam_other), (R_ref, R_other), (t_ref, t_other))
ps.compile_module()
ps.setup_transformations()

# compute mask HFoV 180 deg
ocam_ref.get_thetas()
thetas = ocam_ref.thetas
rays = ocam_ref.rays
mask_theta = ((cp.abs(thetas[:,:,2]) < np.deg2rad(180//2))).get()
mask = np.zeros(rays.shape[:2], dtype=np.uint8)
mask[mask_theta] = 255
cv2.imwrite('mask.png', mask)

# set mask
ps.set_mask_indexed(mask)

# do plane sweep
gimg_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
gimg_other = cv2.cvtColor(img_other, cv2.COLOR_BGR2GRAY)
xyz = ps.get_xyz(gimg_ref, gimg_other)

# write depth color map
Dimg = depth_to_colormap(xyz[:,:,2].get(), ps.near_z, ps.far_z)
cv2.imwrite('depth.png', Dimg)
Dimg_ref = depth_to_colormap(xyz_ref[:,:,2], ps.near_z, ps.far_z)
cv2.imwrite('depth_ref.png', Dimg_ref)
