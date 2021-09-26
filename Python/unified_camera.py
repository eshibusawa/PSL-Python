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

class unified_camera():
    def __init__(self, K, xi):
        self.K = K
        self.xi = xi
        self.undistort_map = None
        self.iK = np.array([[1/self.K[0,0], -self.K[0,2]/self.K[0,0]], [1/self.K[1,1], -self.K[1,2]/self.K[1,1]]])

    def unproject(self, x):
        if self.K[0, 1] != 0:
            return None
        # rectify
        rxy = self.iK[:,0][:,np.newaxis]*x + self.iK[:,1][:,np.newaxis]
        rsqr = np.sum(rxy * rxy, axis=0)
        # scale
        Ds = 1 + (1 - self.xi * self.xi) * rsqr
        mask = Ds < 0
        Ds[mask] = -Ds[mask]
        Fs = (self.xi + np.sqrt(Ds))/(rsqr + 1)
        # unproject
        X = np.empty((3, x.shape[1]), dtype=x.dtype)
        X[0,:] = Fs * rxy[0,:]
        X[1,:] = Fs * rxy[1,:]
        X[2,:] = Fs - self.xi
        return X

    def project(self, X):
        ds = np.linalg.norm(X, axis=0)
        nX = X / ds
        nX[2,:] += self.xi
        rxy = np.empty((2, X.shape[1]), dtype=X.dtype)
        rxy[0,:] = nX[0,:]/nX[2,:]
        rxy[1,:] = nX[1,:]/nX[2,:]

        xy = np.empty((2, X.shape[1]), dtype=X.dtype)
        xy[0,:] = self.K[0, 0] * rxy[0, :] + self.K[0, 2]
        xy[1,:] = self.K[1, 1] * rxy[1, :] + self.K[1, 2]
        return xy

    def unproject_rays(self, sz):
        xy = np.empty((2, sz[0], sz[1]), dtype=np.float)
        xy[0,:,:] = np.arange(0, sz[1])[np.newaxis,:]
        xy[1,:,:] = np.arange(0, sz[0])[:,np.newaxis]
        xy = xy.reshape(2, -1)
        return self.unproject(xy)

    def get_undistort_map(self, sz, ss, ks, ps):
        sz_new = int(np.round(ss[0] * sz[0])), int(np.round(ss[0] * sz[1])) # [w, h]
        K_new = np.copy(self.K)
        K_new[0, 0] *= (ss[0])*(ss[1])
        K_new[1, 1] *= (ss[0])*(ss[1])
        K_new[0, 2] = (K_new[0, 2] + 0.5) * (ss[0]) - 0.5
        K_new[1, 2] = (K_new[1, 2] + 0.5) * (ss[0]) - 0.5

        xy2 = np.empty((2, sz_new[1], sz_new[0]), np.float32)
        xy2[0, :, :] = np.arange(0, sz_new[0])[np.newaxis,:]
        xy2[1, :, :] = np.arange(0, sz_new[1])[:,np.newaxis]
        xy = xy2.reshape(2, -1)
        iK_new = np.array([[1/K_new[0,0], -K_new[0,2]/K_new[0,0]], [1/K_new[1,1], -K_new[1,2]/K_new[1,1]]])
        mxy = np.empty_like(xy)
        mxy[0] = xy[0] * iK_new[0,0] + iK_new[0,1]
        mxy[1] = xy[1] * iK_new[1,0] + iK_new[1,1]

        mxmx = mxy[0] * mxy[0]
        mxmy = mxy[0] * mxy[1]
        mymy = mxy[1] * mxy[1]
        rho_sqr = mxmx + mymy
        # radial
        ratio_radial = ks[0] * rho_sqr + ks[1] * np.square(rho_sqr)
        dxy = np.empty_like(xy)
        dxy[0] = mxy[0] * ratio_radial
        dxy[1] = mxy[1] * ratio_radial
        # tangential
        dxy[0] += 2 * ps[0] * mxmy + ps[1] * (rho_sqr + 2 * mxmx)
        dxy[1] += ps[0] * (rho_sqr + 2 * mymy) + 2 * ps[1] * mxmy
        # distorted
        udx = mxy[0] + dxy[0]
        udy = mxy[1] + dxy[1]
        xd = self.K[0, 0]*udx + self.K[0, 2]
        yd = self.K[1, 1]*udy + self.K[1, 2]
        ud_map = np.empty((2, sz_new[1], sz_new[0]), np.float32)
        ud_map[0] = xd.reshape(sz_new[1], sz_new[0])
        ud_map[1] = yd.reshape(sz_new[1], sz_new[0])

        return ud_map, K_new
