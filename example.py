# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:12:17 2023

@author: runep
"""
import numpy as np
import nibabel as nib
from os.path import join
from dosma.core import MedicalVolume
from dosma.tissues import FemoralCartilage, TibialCartilage
from copy import deepcopy
from scipy.spatial.distance import cdist
from extract_bone_and_cartilage_regions import collect_bone_and_cartilage, progress

import matplotlib.pyplot as plt
#%%
class segmentation:
    def __init__(self, path):
        self.file = nib.load(path)
        self.header = self.file.header
        self.pixel_spacing = self.header['pixdim'][1]
        self.slice_thickness = self.header['pixdim'][3]   


seg = segmentation('data/S01A_seg.nii.gz')
seg.A = seg.file.get_fdata()
seg.A = seg.A[...,118:]

bone_thickness_threshold = 3 # mm
dist_thresh = round(bone_thickness_threshold / seg.pixel_spacing) # Pixels
#%%
cart = deepcopy(seg.A)

# cart[cart != 2] = 0
# cart[(cart != 3) & (cart != 4)] = 0
# cart[cart != 1] = 0

# locs_cart = np.nonzero(np.logical_or(seg.A == 3, seg.A == 4))
locs_cart = np.where((cart == 3) | (cart == 4))
# locs_cart = np.nonzero(cart)
locs_cart = tuple(np.unique(locs) for locs in locs_cart)

ex_pts = [(np.min(locs_cart[1]),np.max(locs_cart[1])), 
          (np.min(locs_cart[0]),np.max(locs_cart[0])),
          (np.min(locs_cart[2]),np.max(locs_cart[2]))]

pad = 0
bounding_box = np.s_[ex_pts[1][0]-pad:ex_pts[1][1]+pad, 
                     ex_pts[0][0]-pad:ex_pts[0][1]+pad,
                     ex_pts[2][0]-pad:ex_pts[2][1]+pad]

x_lim, y_lim, z_lim = [bounding_box[0].start, bounding_box[1].start, bounding_box[2].start]

test = seg.A[bounding_box]
# test[test != 7] = 0
# test[test != 8] = 0
# test[test != 9] = 0
cart = cart[bounding_box]
# cart[cart != 0] = 1

#%%
import time
final_mask = np.zeros(seg.A.shape)
t1 = time.time()
for z in range(cart.shape[2]):
    # locs_cart_sub_area = np.nonzero(cart[...,z])
    locs_cart_sub_area = np.where((cart[...,z] == 3) | (cart[...,z] == 4))
    locs_cart_sub_area = np.stack(locs_cart_sub_area).T
    # locs_bone_sub_area = np.nonzero(test[...,z])
    locs_bone_sub_area = np.where(test[...,z] == 8)
    locs_bone_sub_area = np.stack(locs_bone_sub_area).T
    
    dist = cdist(locs_bone_sub_area, locs_cart_sub_area)
    dist[dist > dist_thresh] = 0
    closest_ind = np.where((dist < dist_thresh) & (dist > 0))
    
    for i in closest_ind[0]:
        x,y = locs_bone_sub_area[i]
        final_mask[x+x_lim, y+y_lim, z+z_lim] = 2

print("Time taken %d" % (time.time()-t1))

# sub_mask = np.zeros((36,36))
# final_mask[np.where(z < dist_thresh)] = 2
# for i in range(len(z)):
    # sub_mask[locs_bone_sub_area[i,0], locs_bone_sub_area[i,1]] = z[i,0]
#%%
cor_ind = 46//2
sag_ind = 50
ax_ind  = 33

fig, ax = plt.subplots(1,2)
ax[0].imshow(test[...,ax_ind], cmap='gray')
ax[0].imshow(cart[...,ax_ind], alpha=.1*cart[...,ax_ind])
ax[0].plot(sag_ind, cor_ind, '+b', markersize=15)

# ax[1].imshow(bone_sub_area[...,ax_ind], cmap='gray')
# ax[1].imshow(cart_sub_area[...,ax_ind], alpha=.8*cart_sub_area[...,ax_ind])
# ax[1].plot(sub_pad, sub_pad, '+b', markersize=15)
# ax[1].plot(17,23, '+r', markersize=15)
# sub_mask[sub_mask < dist_thresh] = 0
# im = ax[2].imshow(sub_mask)
# ax[2].imshow(cart_sub_area[...,ax_ind], alpha=.8*cart_sub_area[...,ax_ind])
# ax[2].plot(sub_pad, sub_pad, '+b', markersize=15)
# ax[2].plot(17,23, '+r', markersize=15)

ax[1].imshow(seg.A[...,ax_ind+z_lim], cmap='gray')
ax[1].imshow(final_mask[...,ax_ind+z_lim], alpha=.5*final_mask[...,ax_ind+z_lim], cmap='hot')
# ax[1].imshow(cart[...,ax_ind], alpha=.1*cart[...,ax_ind], cmap='hot')
# cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
# fig.colorbar(im, cax=cax, orientation='horizontal')
plt.tight_layout()
plt.show()
#%%
name = 'tibia'

fake_right = np.zeros_like(final_mask)
zzzzz = np.array((final_mask, fake_right)).reshape(512,512,236)
finallll = np.zeros((512,512,236))

finallll[...,118:] = final_mask

final_mask[(cart == 3) | (cart == 4)] = 1
nib.save(nib.Nifti1Image(finallll, seg.file.affine), 
         name + '.nii.gz')

nib.save(nib.Nifti1Image(cart, seg.file.affine), 
         name + '_cart.nii.gz')

nib.save(nib.Nifti1Image(test, seg.file.affine), 
         name + '_bone.nii.gz')
#%%
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
           'figure.figsize': (16, 14),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
del params


fig, ax = plt.subplots(1,3)

ax[0].imshow(test[cor_ind,:,:], cmap='gray')
ax[0].imshow(cart[cor_ind,:,:], alpha=.8*cart[cor_ind,:,:])
ax[0].plot(ax_ind, sag_ind, '+b', markersize=15)

ax[1].imshow(test[:,sag_ind,:], cmap='gray')
ax[1].imshow(cart[:,sag_ind,:], alpha=.8*cart[:,sag_ind,:])
ax[1].plot(ax_ind, cor_ind, '+b', markersize=15)

ax[2].imshow(test[:,:,ax_ind], cmap='gray')
ax[2].imshow(cart[:,:,ax_ind], alpha=.8*cart[:,:,ax_ind])
ax[2].plot(sag_ind, cor_ind, '+b', markersize=15)

# ax.plot(40,20, '+b', markersize=15)
# ax.plot(20,40, '+b', markersize=15)
# zzz = np.zeros_like(cart[...,33])
# zzz[20,40] = 1
# ax.imshow(zzz, alpha=zzz)
# for i, j in inner:
#     ax.plot(j, i, '+b', markersize=10)

ax[0].set_aspect(seg.pixel_spacing/seg.slice_thickness)
ax[1].set_aspect(seg.pixel_spacing/seg.slice_thickness)
plt.tight_layout()
plt.show()

#%% Random
inner = []
for i in range(cart.shape[0]):
    mask = np.r_[np.equal(cart[i,:,33], 1)]
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    if len(idx) > 0:
        inner.append((i, np.min(idx)))
