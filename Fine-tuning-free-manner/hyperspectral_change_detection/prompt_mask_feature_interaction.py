import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
import torch.nn.functional as F
import torch
import random

"""
Utilizing HyperFree for hyperspectral change detection task directly without fine tuning.
We develop different Prompt—Mask—Feature interaction workflows for each task.
"""


def Hyperspectral_CD(mask_generator, img1, img2, spectral_lengths, GSD, ratio_threshold):
    """
    Compute the change map between img1 and img2 with HyperFree
    Args:
        mask_generator: object instance from SamAutomaticMaskGenerator
        img1: hyperspectral imagery at the first time with shape [H, W, C] in range [0,255]
        img2: hyperspectral imagery at the second time with shape [H, W, C] in range [0,255]
        spectral_lengths: a list, storing wavelengths for each hyperspectral channel 
        GSD: ground sampling distance (m/pixel). list, such as [1.0] or tensor, such as torch.tensor([1.0])
        ratio_threshold: a float, pixels with the change score higher than ratio_threshold quantile are considered as changes
    Returns:
        a binary change map
    """

    img1_all_masks = mask_generator.generate(img1,spectral_lengths, GSD)

    mask_generator.predictor.set_image(img1)
    img1_features = mask_generator.predictor.features
    img1_features = torch.nn.functional.interpolate(img1_features, (img1.shape[0],img1.shape[1]))

    img2_all_masks = mask_generator.generate(img2, spectral_lengths, GSD)

    mask_generator.predictor.set_image(img2)
    img2_features = mask_generator.predictor.features
    img2_features = torch.nn.functional.interpolate(img2_features, (img1.shape[0],img1.shape[1]))

    change_map1 = get_change_location(img2_all_masks, img1_features, img2_features)

    change_map2 = get_change_location(img1_all_masks, img1_features, img2_features)

    change_map = np.maximum(change_map1,change_map2)

    thresh = np.quantile(change_map, ratio_threshold)
    change_map[change_map > thresh] = 1
    change_map[change_map != 1] = 0

    return change_map, img1_all_masks, img2_all_masks


def get_change_location(img1_masks, img1_features, img2_features):
    """
    Compute the change map in the direction from img1 to img2
    Args:
        img1_masks: (list), storing all the segmented masks of img1
        img1_features: (tensor) backbone features of img1
        img2_features: (tensor) backbone features of img2
    Returns:
        change density map, where a higher value represnets a higher change possibility
    """
    if len(img1_masks) == 0:
        return
    sorted_anns = sorted(img1_masks, key=(lambda x: x['area']), reverse=True)
    change_map = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))

    for ann in sorted_anns:
        m = ann['segmentation']
        if (ann['area'])/(img1_masks[0]['segmentation'].shape[0]*img1_masks[0]['segmentation'].shape[1]) > 0.9:
            continue

        locs = np.where(m == True)
        rows, cols = locs

        time1_feature = img1_features[0,:,:,:][:,rows,cols].mean(1)
        time2_feature = img2_features[0,:,:,:][:,rows,cols].mean(1)

        change_value =  1 - cosine_similarity(time1_feature, time2_feature)
        score = change_value.item()
        change_value = max(0, score)
        change_value = min(score, 1)
        change_map[locs[0], locs[1]] = score

    return change_map


def cosine_similarity(vec1, vec2):
    norm1 = torch.norm(vec1, dim=0)
    norm2 = torch.norm(vec2, dim=0)
    dot_product = torch.dot(vec1, vec2)
    cosine_sim = dot_product / (norm1 * norm2 + 1e-8) 
    return cosine_sim


def show_anns(anns, save_path=''):
    if len(anns) == 0:
        print("len=0")
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    res = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        if (ann['area'])/(anns[0]['segmentation'].shape[0]*anns[0]['segmentation'].shape[1]) > 0.9:
            continue
        color_mask = np.random.random((1, 3)).tolist()[0]
        locs = np.where(m == True)
        res[locs[0], locs[1], :] = np.array(color_mask)*255

    cv2.imwrite(save_path, res)


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False

    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 4)
        return OA

    def User_Accuracy_Class(self): 
        self._pre_cal() if not self.pre_cal else 0
        UA = self.TP + self.FP
        UA = np.where(UA == 0, 0, self.TP / UA)
        UA = np.round(UA, 4)
        return UA[-1]

    def Producer_Accuracy_Class(self):
        self._pre_cal() if not self.pre_cal else 0
        PA = self.TP + self.FN
        PA = np.where(PA == 0, 0, self.TP / PA)
        PA = np.round(PA, 4)
        return PA[-1]

    def Mean_Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        MIoU = self.TP + self.FN + self.FP
        MIoU = np.where(MIoU == 0, 0, self.TP / MIoU)
        MIoU = np.round(np.nanmean(MIoU), 4)
        return MIoU

    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP[1] + self.FN[1] + self.FP[1]
        IoU = np.where(IoU == 0, 0, self.TP[1] / IoU)
        IoU = np.round(np.nanmean(IoU), 4)
        return IoU

    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        F1 = np.round(F1, 4)
        return F1[-1]

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.TN = np.sum(self.confusion_matrix) - self.TP - self.FP - self.FN
        self.pre_cal = True

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False
        
def enhance_contrast_histogram(hsi):
    hsi = np.array(hsi, dtype=np.float32)    
    H, W, C = hsi.shape    
    enhanced_hsi = np.zeros_like(hsi, dtype=np.float32)
    
    for c in range(C):
        band = hsi[:, :, c]   
        band_scaled = cv2.normalize(band, None, 0, 500, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        enhanced_band = cv2.equalizeHist(band_scaled)
        enhanced_hsi[:, :, c] = enhanced_band
    
    return enhanced_hsi
    
def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if deterministic:
        torch.backends.cudnn.deterministic = True
