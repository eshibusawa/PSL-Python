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

import numpy as np
import cv2

from colormap import colormap as jet
from pixel_cost import absolute_difference as ad_cost
from pixel_cost import zero_mean_normalized_cross_correlation as zncc_cost

def depth_to_colormap(depth, min_z, max_z):
    mask = depth > 0
    delta = 1/min_z - 1/max_z
    index = np.round(np.maximum(0, np.minimum(1/(depth + 1E-6) - 1/max_z, delta) / delta) * 255).astype(np.int)
    img = (jet[index] * 255).astype(np.uint8)
    img[~mask] = (0,0,0)
    return img

class plane_sweep():
    def __init__(self):
        self.scale = 1.0
        self.near_z = 0.4
        self.far_z = 5000
        self.match_window_width = 7
        self.match_window_height = 7
        self.num_planes = 256
        self.plane_generation_mode_ = {'uniform_depth':0, 'uniform_disparity':1}
        self.plane_generation_mode = self.plane_generation_mode_['uniform_disparity']
        self.matching_costs_ = {'sad':0, 'zncc':1}
        self.matching_costs = self.matching_costs_['sad']
        self.sub_pixel_interpolation_mode_ = {'direct':0, 'inverse':1}
        self.sub_pixel_interpolation_mode = self.sub_pixel_interpolation_mode_['inverse']
        self.sub_pixel_enabled = True
        self.box_filer_ad_enabled = True
        self.imgs = list()
        self.cams = list()
        self.Rs = list()
        self.Ts = list()
        self.CV = None
        self.rays = None

    def generate_planes(self):
        self.planes = np.empty((4, self.num_planes), dtype=np.float)
        self.planes[0:2, :] = 0
        self.planes[2, :] = -1
        if self.plane_generation_mode == 1:
            minD = 1/self.far_z
            maxD = 1/self.near_z
            dstep = (maxD - minD)/(self.num_planes - 1)
            self.planes[3, :] = 1 / (np.arange(maxD, minD - dstep, -dstep))
        else:
            step = (self.far_z - self.near_z)/(self.num_planes - 1)
            self.planes[3, :] = np.arange(self.near_z, self.far_z + step, step)

    def transform_planes(self, R, t):
        self.Hs = np.empty((self.planes.shape[1], 3, 3), dtype=self.planes.dtype)
        for k in range(0, self.Hs.shape[0]):
            p = self.planes[:,k]
            n = (p[0:3][:,np.newaxis])
            self.Hs[k] = R + np.dot(t, n.T)/p[3]

    def add_image(self, cam, R, t, img):
        # Xc = Rc*X - Tc
        self.cams.append(cam)
        self.Rs.append(R)
        self.Ts.append(t)
        self.imgs.append(img)

    def get_cost_volume(self, ref):
        self.generate_planes()

        n_planes = self.planes.shape[1]
        n_imgs = len(self.imgs)
        sz_img = self.imgs[ref].shape

        accumlation_scale = 1.0 / (n_imgs - 1)
        if self.matching_costs == 1:
            cost_function = zncc_cost(self.imgs[ref], n_planes, (self.match_window_width, self.match_window_height), scale=accumlation_scale)
        else:
            cost_function = ad_cost(self.imgs[ref], n_planes, (self.match_window_width, self.match_window_height), self.box_filer_ad_enabled, scale=accumlation_scale)

        if n_imgs == 2: # stereo case
            self.mask = np.zeros((sz_img[0], sz_img[1], n_planes), np.bool)

        rays = self.cams[ref].unproject_rays(sz_img)
        for k in range(0, n_imgs):
            if k == ref:
                continue
            # relative pose
            R = np.dot(self.Rs[k], self.Rs[ref].T)
            t = np.dot(R, self.Ts[ref]) - self.Ts[k]
            self.transform_planes(R, t)
            img_other = self.imgs[k]
            cam_other = self.cams[k]
            for l in range(0, n_planes):
                xyz = np.dot(self.Hs[l], rays)
                xy = cam_other.project(xyz).astype(np.float32)
                warp = cv2.remap(img_other, xy[0].reshape(sz_img), xy[1].reshape(sz_img), cv2.INTER_LINEAR, None, cv2.BORDER_WRAP, 255)
                cost_function.accumulate(warp, l)
                if n_imgs == 2: # stereo case
                    mask = (xy[0,:] >= 0) & (xy[1,:] >= 0) & (xy[0,:] < sz_img[1]) & (xy[1,:] < sz_img[0])
                    mask = mask.reshape(sz_img)
                    self.mask[:,:,l] = mask

        self.CV = cost_function.get_cost_volume()
        self.rays = rays

    def get_depth_from_planes(self, indices):
        if (self.planes is None) or (self.rays is None):
            return None

        xy = np.empty((2, self.rays.shape[1]), dtype=np.float)
        xy[0] = self.rays[0,:] / self.rays[2,:]
        xy[1] = self.rays[1,:] / self.rays[2,:]

        indices_0 = indices.reshape(-1)
        plane_depth = self.planes[3,indices_0]
        plane_dot = xy[0] * self.planes[0, indices_0] + xy[1] * self.planes[1, indices_0] + self.planes[2, indices_0]

        if self.sub_pixel_enabled:
            indices_mask = (indices != 0)
            indices_mask &= (indices != self.CV.shape[2] - 1)
            indices_m = np.copy(indices)
            indices_m[indices_mask] -= 1
            indices_p = np.copy(indices)
            indices_p[indices_mask] += 1
            cost_0 = np.take_along_axis(self.CV, indices[:,:,np.newaxis], 2).reshape(-1)
            cost_p = np.take_along_axis(self.CV, indices_p[:,:,np.newaxis], 2).reshape(-1)
            cost_m = np.take_along_axis(self.CV, indices_m[:,:,np.newaxis], 2).reshape(-1)
            offset_denom = cost_p + cost_m - (cost_0 * 2)
            offset_denom_mask = np.abs(offset_denom) < 1E-5
            offset_denom[offset_denom_mask] = 1
            offset = (cost_m - cost_p)/(2 * offset_denom)
            offset[offset_denom_mask] = 0

            if self.sub_pixel_interpolation_mode == 0:
                step_p = self.planes[3, indices_p.reshape(-1)] - self.planes[3, indices_0]
                step_m = self.planes[3, indices_0] - self.planes[3, indices_m.reshape(-1)]
                depth_sub = np.copy(offset)
                depth_sub[offset < 0] = (offset*step_m)[offset < 0]
                depth_sub[offset > 0] = (offset*step_p)[offset > 0]
                D = -(plane_depth + depth_sub) / plane_dot
            else:
                step_p = 1/(self.planes[3, indices_0]) - 1/(self.planes[3, indices_p.reshape(-1)])
                step_m = 1/(self.planes[3, indices_m.reshape(-1)]) - 1/(self.planes[3, indices_0])

                depth_sub = np.copy(offset)
                depth_sub[offset < 0] = (offset*step_m)[offset < 0]
                depth_sub[offset > 0] = (offset*step_p)[offset > 0]
                D = -1/(1/plane_depth - depth_sub) / plane_dot
        else:
            D = -plane_depth / plane_dot

        D = D.reshape(indices.shape)
        return D

    def get_depth(self):
        if self.CV is None:
            return None
        if False:
            best_plane_indices = np.argmin(self.CV, axis=2)
        else:
            fliped_indices = np.flip(np.arange(0, self.CV.shape[2], 1), axis=0)
            # select far depth in the case of same cost value
            best_plane_indices = fliped_indices[np.argmin(np.flip(self.CV, axis=2), axis=2)]

        D = self.get_depth_from_planes(best_plane_indices)
        return D