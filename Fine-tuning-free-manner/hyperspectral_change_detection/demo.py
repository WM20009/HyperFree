import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import numpy as np
import torch

from HyperFree.utils.spectral_process_utils import read_img, write_img
from HyperFree import SamAutomaticMaskGenerator, HyperFree_model_registry
from prompt_mask_feature_interaction import Hyperspectral_CD, Evaluator, show_anns, set_random_seed, enhance_contrast_histogram
import cv2

"""
HyperFree
Hyperspectral Change Detection in Tuning-free Manner
"""

device = 'cuda:1'
set_random_seed(111)
img1_paths = ['./../../Data/hyperspectral_change_detection/Hermiston/val/time1/img1_1.tif', \
           './../../Data/hyperspectral_change_detection/Hermiston/val/time1/img1_2.tif']
img2_paths = ['./../../Data/hyperspectral_change_detection/Hermiston/val/time2/img2_1.tif', \
           './../../Data/hyperspectral_change_detection/Hermiston/val/time2/img2_2.tif']
mask_paths = ['./../../Data/hyperspectral_change_detection/Hermiston/val/label/label_1.tif', \
              './../../Data/hyperspectral_change_detection/Hermiston/val/label/label_2.tif']

wavelengths=[427, 437, 447, 457, 467, 477, 487, 497, 507, 517, 527, 537, 547, 557, 567, 577, 587, 597, 607, 617, 627, 637, 647, 657, 667, 677, 687, 697, 707, 717, 727, 737, 
747, 757, 767, 777, 787, 797, 807, 817, 827, 837, 847, 857, 867, 877, 887, 897, 907, 917, 927, 937, 947, 957, 967, 977, 987, 997, 1007, 1017, 1027, 1037, 1047, 
1057, 1067, 1077, 1087, 1097, 1107, 1117, 1127, 1137, 1147, 1157, 1167, 1177, 1187, 1197, 1207, 1217, 1227, 1237, 1247, 1257, 1267, 1277, 1287, 1297, 1307, 1317, 
1327, 1337, 1347, 1357, 1367, 1377, 1387, 1397, 1407, 1417, 1427, 1437, 1447, 1457, 1467, 1477, 1487, 1497, 1507, 1517, 1527, 1537, 1547, 1557, 1567, 1577, 1587, 
1597, 1607, 1617, 1627, 1637, 1647, 1657, 1667, 1677, 1687, 1697, 1707, 1717, 1727, 1737, 1747, 1757, 1767, 1777, 1787, 1797, 1807, 1817, 1827, 1837, 1847, 1857, 
1867, 1877, 1887, 1897, 1907, 1917, 1927, 1937, 1947, 1957]

band_subset_start, band_subset_end = 30, 90
wavelengths = wavelengths[band_subset_start:band_subset_end]
pred_iou_thresh = 0.5 # Controling the model's predicted mask quality in range [0, 1].
stability_score_thresh = 0.6 # Controling the stability of the mask in range [0, 1].
evaluator = Evaluator(2)

ckpt_pth = "./../../Ckpt/HyperFree-b.pth"
save_dir = './../../Outputs/hyperspectral_change_detection/'
GSD = 30 # Ground sampling distance (m/pixel)
ratio_threshold = 0.74 # a float, pixels with the change score higher than ratio_threshold quantile are considered as changes

HyperFree = HyperFree_model_registry["vit_b"](checkpoint=ckpt_pth, encoder_global_attn_indexes=-1, merge_indexs = None).to(device)
HyperFree = HyperFree.to(device)
mask_generator = SamAutomaticMaskGenerator(HyperFree, pred_iou_thresh = pred_iou_thresh, stability_score_thresh = stability_score_thresh, points_per_side=64)

for i in range(len(img1_paths)):
    path1 = img1_paths[i]
    path2 = img2_paths[i]
    mask_path = mask_paths[i]
           
    img1 = read_img(path1)
    if img1.max() > 500:
        #img1 = img1/(img1.max()/500)
        #img1 = enhance_contrast_histogram(img1)
        pass
    else:
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img1 = (255 * img1).astype(np.uint8)
    img1_uint8 = img1[:,:,band_subset_start:band_subset_end]

    img2 = read_img(path2)
    if img2.max() > 500:
        #img2 = img2/(img2.max()/500)
        #img2 = enhance_contrast_histogram(img2)
        pass
    else:
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img2 = (255 * img2).astype(np.uint8)
    img2_uint8 = img2[:,:,band_subset_start:band_subset_end]

    mask = read_img(mask_path)

    ratio = 1024/(max(img1_uint8.shape[0], img1_uint8.shape[1]))
    GSDS = GSD/ratio
    GSDS = torch.tensor([GSDS])

    change_map, img1_all_masks, img2_all_masks = Hyperspectral_CD(mask_generator, img1_uint8, img2_uint8, wavelengths, GSDS, ratio_threshold)
    cv2.imwrite(os.path.join(save_dir, str(i+1) + 'change_map.jpg'),change_map*255)
    show_anns(img1_all_masks, save_dir + str(i+1) + 'img1.jpg')
    show_anns(img2_all_masks, save_dir + str(i+1) + 'img2.jpg')
    evaluator.add_batch(mask.astype('int'), change_map.astype('int'))

OA = evaluator.Overall_Accuracy()
UA = evaluator.User_Accuracy_Class()
PA = evaluator.Producer_Accuracy_Class()
IoU = evaluator.Intersection_over_Union()
F1_score = evaluator.F1_score()

print('OA: ' + str(OA))
print('UA: ' + str(UA))
print('PA: ' + str(PA))
print('IoU: ' + str(IoU))
print('F1_score: ' + str(F1_score))
