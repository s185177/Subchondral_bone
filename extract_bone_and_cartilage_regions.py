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
from copy import deepcopy

def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)

def collect_bone_and_cartilage(seg_bone, seg_cart, bone, femoral_bone_thick):
    slices = list(set(np.where(seg_cart != 0)[-1]))
    out_mask = np.zeros_like(seg_bone.A)
    # Padding was only used for plotting 
    # to show more of the image around the mask
    pad = 0
    
    print("Defining sub-bone mask")    
    for n, ind in enumerate(slices):
        progress(n)
        time.sleep(0.1)
        # Maybe some erosion could be needed as to remove outlying mask pixels
        # but not sure whether it was necessary
        # slice_cart = erosion(slice_cart, disk(1))

        # Get cartilage locations     
        locs_cart = np.nonzero(seg_cart[:,:,ind])
        
        # Get corner points of the cartilage
        ex_pts = [(np.min(locs_cart[1]),np.max(locs_cart[1])), 
                  (np.min(locs_cart[0]),np.max(locs_cart[0]))]
        
        # Define bounding box used to narrow the search window
        bounding_box = np.s_[ex_pts[1][0]-pad:ex_pts[1][1]+pad, 
                             ex_pts[0][0]-pad:ex_pts[0][1]+pad]
        
        x_lim, y_lim = [bounding_box[0].start, bounding_box[1].start]
        
        # Get bone mask and use only femoral bone labeled with 7
        slice_bone = seg_bone.A[...,ind]
        slice_bone = slice_bone[bounding_box]
        slice_bone[slice_bone != bone] = 0
        locs_bone = np.nonzero(slice_bone)
        
        # Define the femoral subchondral bone
        for coor in range(len(locs_bone[0])):
            x = locs_bone[0][coor] - (locs_cart[0] - x_lim)
            y = locs_bone[1][coor] - (locs_cart[1] - y_lim)
            dist = np.sqrt(x**2 + y**2)
            # Actually think this line defines that 
            # we're looking at x coordinates only on the sagittal plane
            # so I think this is the place to make changes, by either making it
            # work 3-dimensionally or change the plane.
            if len(np.where(dist <= femoral_bone_thick)[0]) >= 1:
                out_mask[locs_bone[0][coor] + x_lim,locs_bone[1][coor] + y_lim, ind] = 2
    
    # Label all from cartilage mask as cartilage
    out_mask[seg_cart != 0] = 1
    return out_mask 

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
            # 32 is an axial key used for superficial, 
            # the others 64 & 128 didn't give much else
            roi = (sag | cor | 32)
            curr_mask = fc.__binarize_region_mask__(regions_mask, roi)
            
            final_mask[curr_mask != 0] = c
            regions_map[c] = 'fem_cart_' + fc._SAGITTAL_NAMES[i] + '_' + fc._CORONAL_NAMES[j]
            regions_map[c+6] = 'fem_bone_' + fc._SAGITTAL_NAMES[i] + '_' + fc._CORONAL_NAMES[j]
            
            c += 1

    final_mask[mask == 0] = 0
    final_mask[mask == 2] += 6 
    
    return final_mask, dict(sorted(regions_map.items()))

def get_tibial_regions(mask):
    tc = TibialCartilage()
    tc.set_mask(mask)
    # Create regional mask, remove the 2 
    # at the end if you want [ant_cen_pos] and [sup_inf]
    tc_regions_mask = tc.regions_mask[...,2]
    
    final_mask = np.array(np.where((tc_regions_mask == 0), 2,1))
    final_mask[mask == 0] = 0
    # Use this if you want to cut some of the inferior tibial bone off
    # final_mask[tc_regions_mask[...,0] != 0] = 0
    
    # 4 is tibial cartilage and 8 is bone
    final_mask[mask == 3] = 13
    final_mask[mask == 4] = 14
    final_mask[mask == 8] += 14
    return final_mask

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

    seg_bone_h = nib.load(seg_bone_p)
    seg_bone_img = MedicalVolume.from_nib(seg_bone_h)
    img_dims = seg_bone_img.shape

    dist_thresh = round(bone_thickness_threshold / seg_bone_h.header['pixdim'][1]) # Pixels
    
    # Get femoral cartilage from bone segmentation
    fc_cart = deepcopy(seg_bone_img) #._partial_clone()
    fc_cart[fc_cart != 2] = 0
    print('___________________________')
    t1 = time.time()
    # Use bone = 7 for femur
    out_mask_fc = collect_bone_and_cartilage(seg_bone_img, fc_cart.A, 7, dist_thresh)
    print()
    print("Time taken %d seconds" % (time.time() - t1))
    del fc_cart
    print('___________________________')
    print('Splitting femoral regions bi-lateral')
    
    # Region extraction requires medical volume
    nifti_mask = nib.Nifti1Image(out_mask_fc, seg_bone_h.affine)
    seg_all_img = MedicalVolume.from_nib(nifti_mask)
    
    # Get regions for bi-lateral knees
    fc_left,  regions_map = get_femoral_regions(seg_all_img[...,:img_dims[-1]//2])
    fc_right, _ = get_femoral_regions(seg_all_img[...,img_dims[-1]//2:])
    fc_mask  = np.dstack((fc_left, fc_right))
    del fc_left, fc_right, nifti_mask, seg_all_img
    print()
    print('____________Tibia____________')
    # Think that 4 is cartilage but maybe it's more meniscus, not sure
    # 8 is tibial bone
    bone_copy = deepcopy(seg_bone_img) #._partial_clone()
    bone_copy[(bone_copy != 8) & (bone_copy != 4) & (bone_copy != 3)] = 0
    print('Splitting regions')
    tc_left  = get_tibial_regions(bone_copy[...,:img_dims[-1]//2])
    tc_right = get_tibial_regions(bone_copy[...,img_dims[-1]//2:])
    tc_mask = np.dstack((tc_left, tc_right))
    del tc_left, tc_right, bone_copy

    print("Getting subchondral bone\n")
    tc_cart = deepcopy(seg_bone_img) #._partial_clone()
    tc_cart[(tc_cart != 3) & (tc_cart != 4)] = 0
    out_mask_tc = collect_bone_and_cartilage(seg_bone_img, tc_cart.A, 8, dist_thresh)
    tc_mask[out_mask_tc == 0] = 0
    del tc_cart, out_mask_tc
    print()
    print('____________Patella__________')
    print('Splitting regions')
    # 1 for cartilage and 9 for bone
    pc_mask = np.zeros_like(seg_bone_img.A)
    pc_mask[seg_bone_img == 1] = 17
    pc_mask[seg_bone_img == 9] = 18
    
    print("Getting subchondral bone\n")
    pc_cart = deepcopy(seg_bone_img) #._partial_clone()
    pc_cart[pc_cart != 1] = 0
    out_mask_pc = collect_bone_and_cartilage(seg_bone_img, pc_cart.A, 9, dist_thresh)
    pc_mask[out_mask_pc == 0] = 0
    del pc_cart, out_mask_pc
    
    print("Collecting masks & saving regions")
    final_mask = np.zeros(img_dims)
    final_mask += fc_mask
    final_mask += tc_mask
    final_mask += pc_mask

    # Save as NIFTI
    img = nib.Nifti1Image(final_mask, seg_bone_img.affine)
    nib.save(img, join(out_dir, file_name))
    
    regions_map[13] = "tib_cart_medial"
    regions_map[14] = "tib_cart_lateral"
    regions_map[15] = "tib_bone_lateral"
    regions_map[16] = "tib_bone_medial"
    regions_map[17] = "pat_cart"
    regions_map[18] = "pat_bone"
    print('___________________________')
    print("Regions legend:")
    print(regions_map)    

if __name__ == "__main__":
    run_region_extraction()