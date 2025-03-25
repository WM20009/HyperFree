import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import numpy as np
import torch

from HyperFree.utils.spectral_process_utils import read_img, write_img
from HyperFree import SamAutomaticMaskGenerator, HyperFree_model_registry
import scipy.io as sio
from prompt_mask_feature_interaction import Hyperspectral_TD, dilate_binary_image, save_heatmap, set_random_seed
from sklearn import metrics

"""
HyperFree
Hyperspectral Target Detection in Tuning-free Manner
"""

device = 'cuda:0'
wavelengths = [
    650.000000,  660.000000,  670.000000,  680.000000,  690.000000,  700.000000,
    710.000000,  720.000000,  730.000000,  740.000000,  750.000000,  760.000000,
    770.000000,  780.000000,  790.000000,  800.000000,  810.000000,  820.000000,
    830.000000,  840.000000,  850.000000,  860.000000,  870.000000,  880.000000,
    890.000000,  900.000000,  910.000000,  920.000000,  930.000000,  940.000000,
    950.000000,  960.000000,  970.000000,  980.000000,  990.000000, 1000.000000,
    1010.000000, 1020.000000, 1030.000000, 1040.000000, 1050.000000, 1060.000000,
    1070.000000, 1080.000000, 1090.000000, 1100.000000
]

pred_iou_thresh = 0.6 # Controling the model's predicted mask quality in range [0, 1].
stability_score_thresh = 0.5 # Controling the stability of the mask in range [0, 1].
cosine_simi_thresh = 0.94 # If the corresponding features of masks are higher than the threshold, they are considered to be of the same type 
feature_index_id = 5 # Deciding which stage of encoder features to use
GSDS = 0.07 # Ground sampling distance (m/pixel)

ckpt_pth = "./../../Ckpt/HyperFree-b.pth"
save_dir = './../../Outputs/hyperspectral_target_detection'
img_pth = './../../Data/hyperspectral_target_detection/Stone.mat'
target_spectrum = './../../Data/hyperspectral_target_detection/target_spectrum_stone.txt' # Storing the target spectraum

with open(target_spectrum, 'r') as file_:
    lines = file_.readlines()
spectral_list = [float(line.strip()) for line in lines]
target_spectral = np.array(spectral_list)

img = sio.loadmat(img_pth)['data']
gt = sio.loadmat(img_pth)['map']

if img.max() > 500:
    img_uint8 = np.clip(img, 0, 500)
else:
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_uint8 = (255 * img_normalized).astype(np.uint8)

ratio = 1024/(max(img.shape[0], img.shape[1]))
GSDS = GSDS/ratio
GSDS = torch.tensor([GSDS])
HyperFree = HyperFree_model_registry["vit_b"](checkpoint=ckpt_pth).to(device)
HyperFree = HyperFree.to(device)
mask_generator = SamAutomaticMaskGenerator(HyperFree, pred_iou_thresh = pred_iou_thresh, stability_score_thresh = stability_score_thresh, points_per_side=64)


detection_map = Hyperspectral_TD(mask_generator, img_uint8, target_spectral, cosine_simi_thresh, wavelengths, feature_index_id, GSDS)
detection_map = dilate_binary_image(detection_map, kernel_size=7)
save_heatmap(detection_map, working_dir=save_dir)

y_l = np.reshape(gt, [-1, 1], order='F')
y_p = np.reshape(detection_map, [-1, 1], order='F')

## calculate the AUC value
fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
fpr = fpr[1:]
tpr = tpr[1:]
threshold = threshold[1:]
auc1 = round(metrics.auc(fpr, tpr), 4)
auc2 = round(metrics.auc(threshold, fpr), 4)
auc3 = round(metrics.auc(threshold, tpr), 4)
auc4 = round(auc1 + auc3 - auc2, 4)
auc5 = round(auc3 / auc2, 4)
print('{:.{precision}f}'.format(auc1, precision=4))
print('{:.{precision}f}'.format(auc2, precision=4))
print('{:.{precision}f}'.format(auc3, precision=4))
print('{:.{precision}f}'.format(auc4, precision=4))
print('{:.{precision}f}'.format(auc5, precision=4))