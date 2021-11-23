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

import torch
import kornia.filters

class absolute_difference():
    def __init__(self, img_ref, kernel_size = (7, 7), box_filter_enabled = True, use_batch = True):
        self.img_ref = img_ref
        self.use_batch = use_batch
        self.kernel_size = kernel_size
        self.box_filter_enabled = box_filter_enabled
        self.precision = torch.float16
        self.box_filter = (lambda x: kornia.filters.box_blur(x, self.kernel_size, border_type='replicate', normalized=False))

    def get_cost_volume(self, warped_images):
        if self.use_batch:
            ad = torch.abs(self.img_ref.to(self.precision) - warped_images)
            if self.box_filter_enabled:
                CV = torch.mean(self.box_filter(ad), dim = 0)
            else:
                CV = torch.mean(ad, dim = 0)
        else:
            sad = None
            scale = 1.0 / warped_images.shape[0]
            for warped_image in warped_images:
                ad = torch.abs(self.img_ref.to(self.precision) - warped_image)
                if self.box_filter_enabled:
                    ad = self.box_filter(ad[None, :])[0, :]
                if sad is None:
                    sad = ad
                else:
                    sad += ad
            CV = sad * scale
            del sad
        del ad
        torch.cuda.empty_cache()
        return CV

class zero_mean_normalized_cross_correlation():
    def __init__(self, img_ref, kernel_size = (7, 7), use_batch = True):
        self.kernel_size = kernel_size
        self.use_batch = use_batch
        self.eps = 1E-7
        self.precision = torch.float32
        self.box_filter = (lambda x, f: kornia.filters.box_blur(x, self.kernel_size, border_type='replicate', normalized=f))
        img_ref2 = img_ref[None, None, :]
        self.img_box = img_ref2 - self.box_filter(img_ref2.to(self.precision), True)
        self.img_sqr_box = self.box_filter(self.img_box * self.img_box, False)

    def get_cost_volume(self, warped_images):
        if self.use_batch:
            warp_box = warped_images - self.box_filter(warped_images.to(self.precision), True)
            warp_sqr_box = self.box_filter(warp_box * warp_box, False)
            cross = warp_box * self.img_box
            nume = self.box_filter(cross, False)
            denom = warp_sqr_box * self.img_sqr_box
            CV = (1 - torch.mean(nume/(torch.sqrt(denom) + self.eps), dim = 0))/2
        else:
            zncc = None
            scale = 1.0 / warped_images.shape[0]
            for warped_image in warped_images:
                w = warped_image[None, :]
                warp_box = w - self.box_filter(w.to(self.precision), True)
                warp_sqr_box = self.box_filter(warp_box * warp_box, False)
                cross = warp_box * self.img_box
                nume = self.box_filter(cross, False)
                denom = warp_sqr_box * self.img_sqr_box
                if zncc is None:
                    zncc = (nume/(torch.sqrt(denom) + self.eps))
                else:
                    zncc += (nume/(torch.sqrt(denom) + self.eps))
            CV = (1 - (zncc[0] * scale))/2
            del zncc
        del warp_box, warp_sqr_box, cross, nume, denom
        torch.cuda.empty_cache()
        return CV