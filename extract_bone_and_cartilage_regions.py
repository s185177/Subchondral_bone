# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:21:44 2023


@author: runep
"""
import time
import numpy as np
import nibabel as nib
from os.path import join
from dosma.core import MedicalVolume
from dosma.tissues import FemoralCartilage, TibialCartilage
from scipy.spatial.distance import cdist

class segmentation:
    def __init__(self, path):
        self.file = nib.load(path)
        self.header = self.file.header
        self.pixel_spacing = self.header['pixdim'][1]
        self.slice_thickness = self.header['pixdim'][3]
        
    def constrain_image(self, locs_cart, tissue):
        locs_cart = tuple(np.unique(locs) for locs in locs_cart)
        
        ex_pts = [(np.min(locs_cart[1]),np.max(locs_cart[1])), 
                  (np.min(locs_cart[0]),np.max(locs_cart[0])),
                  (np.min(locs_cart[2]),np.max(locs_cart[2]))]

        pad = 0
        bounding_box = np.s_[ex_pts[1][0]-pad:ex_pts[1][1]+pad, 
                             ex_pts[0][0]-pad:ex_pts[0][1]+pad,
                             ex_pts[2][0]-pad:ex_pts[2][1]+pad]
        
        self.tissue = tissue
        self.constrained = self.A[bounding_box]
        self.x_lim = bounding_box[0].start
        self.y_lim = bounding_box[1].start
        self.z_lim = bounding_box[2].start

def define_subchondral_bone(bone_seg, tissue, dist_thresh):
    final_mask = np.zeros(bone_seg.A.shape)
    t1 = time.time()
    for z in range(bone_seg.constrained.shape[2]):
        if tissue == 'Femur':
            locs_bone_sub_area = np.where(bone_seg.constrained[...,z] == 7)
            locs_cart_sub_area = np.where(bone_seg.constrained[...,z] == 2)
        elif tissue == 'Tibia':
            locs_bone_sub_area = np.where(bone_seg.constrained[...,z] == 8)
            locs_cart_sub_area = np.where((bone_seg.constrained[...,z] == 3) | 
                                          (bone_seg.constrained[...,z] == 4))
        elif tissue == 'Patella':
            locs_bone_sub_area = np.where(bone_seg.constrained[...,z] == 9)
            locs_cart_sub_area = np.where(bone_seg.constrained[...,z] == 1)
        
        locs_cart_sub_area = np.stack(locs_cart_sub_area).T
        locs_bone_sub_area = np.stack(locs_bone_sub_area).T
        
        dist = cdist(locs_bone_sub_area, locs_cart_sub_area)
        closest_ind = np.where((dist < dist_thresh) & (dist > 0))
        
        for i in closest_ind[0]:
            x,y = locs_bone_sub_area[i]
            final_mask[x+bone_seg.x_lim, 
                       y+bone_seg.y_lim, 
                       z+bone_seg.z_lim] = 2
    
    if tissue == 'Femur':
        final_mask[bone_seg.A == 2] = 1
    elif tissue == 'Tibia':
        final_mask[(bone_seg.A == 3) | (bone_seg.A == 4)] = 1
    elif tissue == 'Patella':
        final_mask[bone_seg.A == 1] = 1
    
    print("Time taken %d seconds\n" % (time.time()-t1))
    bone_seg.mask = final_mask

def get_femoral_regions(mask, thickness_divisor = 1.0):
    fc = FemoralCartilage()
    fc.set_mask(mask)
    
    # Only interested in regions_mask
    # Thickness devisor is default at 0.5, but was changed, so it doesn't
    # crop the edges of the cartilage mask    
    regions_mask, _,_,_ = fc.split_regions(mask.A, thickness_divisor)
    
    # Just used to show the label legend
    regions_map = {}
    final_mask = np.zeros_like(regions_mask)
    
    c = 1
    for i, sag in enumerate(fc._SAGITTAL_KEYS):
        for j, cor in enumerate(fc._CORONAL_KEYS):
            roi = (sag | cor | 32)
            curr_mask = fc.__binarize_region_mask__(regions_mask, roi)
            
            final_mask[curr_mask != 0] = c
            regions_map[c] = 'fem_cart_' + fc._SAGITTAL_NAMES[i] + '_' + fc._CORONAL_NAMES[j]
            regions_map[c+6] = 'fem_bone_' + fc._SAGITTAL_NAMES[i] + '_' + fc._CORONAL_NAMES[j]
            
            c += 1
    return final_mask, dict(sorted(regions_map.items()))

def get_tibial_regions(mask, med_lat):
    tc = TibialCartilage()
    tc.set_mask(mask)
    tc_regions_mask = tc.regions_mask[...,2]
    
    if med_lat == 'left':
        tc_mask = np.array(np.where((tc_regions_mask == 0), 1,2))
    elif med_lat == 'right':
        tc_mask = np.array(np.where((tc_regions_mask == 0), 2,1))
    
    return tc_mask

def run_region_extraction(bone_thickness_threshold=3):
    # User input
    # seg_bone_p = input("\nChoose bone segmentation path (NIFTI):\n")
    seg_bone_p = 'data/S01A_seg.nii.gz'
    
    # User input
    # out_dir = input("\nChoose output directory:\n")
    out_dir = 'data'
    
    file_name = seg_bone_p.split('/')[1].split('_')[0] + '_regional_mask.nii.gz'
    print("\nFinal mask will be saved as - {}\n".format(file_name))

    # Used for user input    
    # choice = input("Femoral bone thickness is set at 3 mm, press y to proceed or n to change\n")
    # if choice == 'n':
    #     bone_thickness_threshold = int(input("\nAssign new threshold value [mm]\n"))
    # print()
    
    seg = segmentation(seg_bone_p)
    seg.A = seg.file.get_fdata()
    img_dims = seg.A.shape
    dist_thresh = round(bone_thickness_threshold / seg.pixel_spacing) # Pixels

    print('____________Femur____________')
    # Get femoral cartilage from bone segmentation
    # Use bone = 7 for femur
    seg.constrain_image(np.where(seg.A == 2), 7)
    define_subchondral_bone(seg, 'Femur', dist_thresh)
    
    # Region extraction requires medical volume
    nifti_mask = nib.Nifti1Image(seg.mask, seg.file.affine)
    seg_med_vol = MedicalVolume.from_nib(nifti_mask)

    print('Splitting femoral regions bi-lateral\n')    
    # Get regions for bi-lateral knees
    fc_left,  regions_map = get_femoral_regions(seg_med_vol[...,:img_dims[2]//2])
    fc_right, _ = get_femoral_regions(seg_med_vol[...,img_dims[2]//2:])
    fc_mask = np.dstack((fc_left, fc_right))
    # Remove excess bone & make difference between bone and cartilage
    fc_mask[seg.mask == 0] = 0
    fc_mask[seg.mask == 2] += 6
    del fc_left, fc_right, nifti_mask
    
    print('____________Tibia____________')
    # Use bone = 8 for femur
    seg.constrain_image(np.where((seg.A == 3) | (seg.A == 4)), 8)
    define_subchondral_bone(seg, 'Tibia', dist_thresh)
    
    bone_med_vol = MedicalVolume.from_nib(seg.file)

    tc_left  = get_tibial_regions(bone_med_vol[...,:img_dims[2]//2], 'left')
    tc_right = get_tibial_regions(bone_med_vol[...,img_dims[2]//2:], 'right')
    tc_mask = np.dstack((tc_left, tc_right))
    tc_mask[seg.mask == 0] = 0
    tc_mask[seg.mask == 1] += 12
    tc_mask[seg.mask == 2] += 14

    del tc_left, tc_right

    print('____________Patella__________')
    # 1 for cartilage and 9 for bone
    seg.constrain_image(np.where(seg.A == 1), 9)
    define_subchondral_bone(seg, 'Patella', dist_thresh)

    pc_mask = np.zeros_like(seg.A)
    pc_mask[seg.mask == 1] = 17
    pc_mask[seg.mask == 2] = 18
    
    print("Collecting masks & saving regions")
    final_mask = np.zeros(img_dims)
    final_mask += fc_mask
    final_mask += tc_mask
    final_mask += pc_mask

    # Save as NIFTI
    img = nib.Nifti1Image(final_mask, seg.file.affine)
    nib.save(img, join(out_dir, file_name))
    
    regions_map[13] = "tib_cart_medial"
    regions_map[14] = "tib_cart_lateral"
    regions_map[15] = "tib_bone_medial"
    regions_map[16] = "tib_bone_lateral"
    regions_map[17] = "pat_cart"
    regions_map[18] = "pat_bone"
    print('___________________________')
    print("Regions legend:")
    print(regions_map)    

if __name__ == "__main__":
    run_region_extraction()