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
        self.near_y = 0.4
        self.far_y = 1.5
        self.num_ground = 16
        self.plane_generation_mode_ = {'uniform_depth':0, 'uniform_disparity':1}
        self.plane_generation_mode = self.plane_generation_mode_['uniform_disparity']
        self.matching_costs_ = {'sad':0, 'zncc':1}
        self.matching_costs = self.matching_costs_['sad']
        self.sub_pixel_interpolation_mode_ = {'direct':0, 'inverse':1}
        self.sub_pixel_interpolation_mode = self.sub_pixel_interpolation_mode_['inverse']
        self.sub_pixel_enabled = True
        self.box_filer_ad_enabled = True
        self.write_debug_warping_enabled = False
        self.ground_plane_enabled = False
        self.use_gpu = False
        self.clear_images()
        self.CV = None
        self.rays = None

    @staticmethod
    def generate_depth(near, far, num, mode):
        if mode == 1:
            minD = 1/far
            maxD = 1/near
            dstep = (maxD - minD)/(num - 1)
            depth = 1 / (np.arange(maxD, minD - dstep, -dstep))
        else:
            step = (far - near)/(num - 1)
            depth  = np.arange(near, far + step, step)
        return np.flip(depth)[:num]

    def generate_planes(self):
        # front parallel
        fp_planes = np.zeros((4, self.num_planes), dtype=np.float)
        fp_planes[2, :] = -1
        fp_planes_has_neighbor = np.full(self.num_planes, True, dtype=np.bool)
        fp_planes_has_neighbor[0] = fp_planes_has_neighbor[-1] = False
        fp_planes[3, :] = plane_sweep.generate_depth(self.near_z, self.far_z, self.num_planes, self.plane_generation_mode)
        self.planes = fp_planes
        self.planes_has_neighbor = fp_planes_has_neighbor

        # ground
        if self.ground_plane_enabled:
            g_planes = np.zeros((4, self.num_ground), dtype=np.float)
            g_planes[1, :] = -1
            g_planes[3, :] = plane_sweep.generate_depth(self.near_y, self.far_y, self.num_ground, self.plane_generation_mode)
            g_planes_has_neighbor = np.full(self.num_ground, True, dtype=np.bool)
            g_planes_has_neighbor[0] = g_planes_has_neighbor[-1] = False
            self.planes = np.hstack((self.planes, g_planes))
            self.planes_has_neighbor = np.hstack((self.planes_has_neighbor, g_planes_has_neighbor))

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

    def clear_images(self):
        self.imgs = list()
        self.cams = list()
        self.Rs = list()
        self.Ts = list()

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

                if self.write_debug_warping_enabled:
                    cv2.imwrite('debug_warping_{:04d}_{:04d}.png'.format(k, l), warp)

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
            indices_mask = self.planes_has_neighbor[indices]
            indices_m = np.copy(indices)
            indices_m[indices_mask] += 1
            indices_p = np.copy(indices)
            indices_p[indices_mask] -= 1
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

    def get_depth(self, ref = 0):
        try:
            import torch
            import psl_cuda as py_psl_cuda
            import kornia.filters
        except ImportError:
            self.use_gpu = False

        if not self.use_gpu:
            if self.CV is None:
                self.get_cost_volume(ref)
            best_plane_indices = np.argmin(self.CV, axis=2)
            D = self.get_depth_from_planes(best_plane_indices)
        else:
            D = self.get_depth_gpu(ref)

        return D

    def get_depth_gpu(self, ref = 0):
        if self.matching_costs != 0:
            raise ValueError("Only SAD is supported!")
        if self.sub_pixel_enabled:
            raise ValueError("Subpixel estimation is not supported!")

        try:
            import torch
            import psl_cuda as py_psl_cuda
            import kornia.filters
        except ImportError:
            return None

        self.generate_planes()
        n_imgs = len(self.imgs)

        # upload tensor
        Ks = [C.K for C in self.cams]
        xis = [C.xi for C in self.cams]
        t_imgs = torch.tensor(np.array(self.imgs), device=torch.device('cuda'))
        t_Ks = torch.tensor(np.array(Ks), dtype=torch.float32, device=torch.device('cpu'))
        t_xis = torch.tensor(np.array(xis), dtype=torch.float32, device=torch.device('cpu'))
        t_Rs = torch.tensor(np.array(self.Rs), dtype=torch.float32, device=torch.device('cpu'))
        t_Ts = torch.tensor(np.array(self.Ts), dtype=torch.float32, device=torch.device('cpu'))
        t_Ps = torch.tensor(self.planes.T, dtype=torch.float32, device=torch.device('cuda'))

        # compute warped images
        warped_images_gpu, rays_gpu = py_psl_cuda.get_warped_image_tensor(ref, t_imgs, t_Ks, t_xis, t_Rs, t_Ts, t_Ps)
        if self.write_debug_warping_enabled:
            warped_images = warped_images_gpu.to('cpu').numpy()
            for k in range(0, n_imgs - 1):
                for l in range(0, self.planes.shape[1]):
                    cv2.imwrite('debug_warping_{:04d}_{:04d}.png'.format(k, l), warped_images[k, l])
        ad_gpu = torch.abs(t_imgs[ref].to(torch.float16) - warped_images_gpu)
        del warped_images_gpu
        torch.cuda.empty_cache()

        # sad
        box_filer_size = (self.match_window_width, self.match_window_height)
        sad = torch.mean(kornia.filters.box_blur(ad_gpu, box_filer_size, border_type='replicate', normalized=False), dim=0)
        del ad_gpu
        torch.cuda.empty_cache()

        indices = torch.argmin(sad, dim=0)
        self.rays = rays_gpu.to('cpu').numpy().transpose(2, 0, 1).reshape(3, -1)
        D = self.get_depth_from_planes(indices.to('cpu').numpy())

        del sad, rays_gpu, indices
        del t_imgs, t_Ks, t_xis, t_Rs, t_Ts, t_Ps
        torch.cuda.empty_cache()

        return D
