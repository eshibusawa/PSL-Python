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

class absolute_difference():
    def __init__(self, img_ref, n_planes, kernel_size = (7, 7), box_filter_enabled = True, scale = 1.0):
        self.CV = np.zeros((img_ref.shape[0], img_ref.shape[1], n_planes), np.float32)
        self.img_ref = img_ref
        self.box_filter_enabled = box_filter_enabled
        self.kernel_size = kernel_size
        self.scale = scale
        self.box_filter = (lambda x: cv2.boxFilter(x, ddepth = -1, ksize=self.kernel_size, normalize=False, borderType=cv2.BORDER_REPLICATE))

    def accumulate(self, warp, index):
        cost = np.abs(self.img_ref.astype(self.CV.dtype) - warp.astype(self.CV.dtype))
        self.CV[:,:,index] += (self.scale * cost)

    def get_cost_volume(self):
        if self.box_filter_enabled:
            for l in range(0, self.CV.shape[2]):
                self.CV[:,:,l] = self.box_filter(self.CV[:,:,l])
        return self.CV

class zero_mean_normalized_cross_correlation():
    def __init__(self, img_ref, n_planes, kernel_size = (7, 7), scale = 1.0):
        self.CV = np.zeros((img_ref.shape[0], img_ref.shape[1], n_planes), np.float32)
        self.img_ref = img_ref
        self.kernel_size = kernel_size
        self.scale = scale
        self.eps = 1E-7
        self.box_filter = (lambda x, f: cv2.boxFilter(x, ddepth = cv2.CV_32F, ksize=self.kernel_size, normalize=f, borderType=cv2.BORDER_REPLICATE))
        img_box = self.img_ref - self.box_filter(self.img_ref, True)
        img_sqr_box = self.box_filter(img_box * img_box, False)
        self.img_box = img_box
        self.img_sqr_box = img_sqr_box

    def accumulate(self, warp, index):
        warp_box = warp - self.box_filter(warp, True)
        warp_sqr_box = self.box_filter(warp_box * warp_box, False)
        cross = warp_box * self.img_box
        nume = self.box_filter(cross, False)
        denom = warp_sqr_box * self.img_sqr_box
        cost = (1 - (nume/(np.sqrt(denom) + self.eps)))/2
        self.CV[:,:,index] += (self.scale * cost)

    def get_cost_volume(self):
        return self.CV
