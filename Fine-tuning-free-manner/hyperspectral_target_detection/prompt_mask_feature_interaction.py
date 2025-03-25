import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
import torch.nn.functional as F
import torch, random

"""
Utilizing HyperFree for hyperspectral target detection task directly without fine tuning.
We develop different Prompt—Mask—Feature interaction workflows for each task.
"""

def Hyperspectral_TD(mask_generator, image, target_spectral, cosine_simi_thresh, spectral_lengths, feature_index_id, GSD):
    """
    Utilizing HyperFree for hyperspectral target detection directly without fine tuning.
    Args:
        mask_generator: object instance from SamAutomaticMaskGenerator
        image: input hyperspectral image with shape [H, W, C] in range [0,255]
        few_shots: a list, storing binary maps for each class, where the non-zero location represents the corresponding sample
        target_spectral: array, storing the target spectraum
        cosine_simi_thresh: float,if the corresponding features of masks are higher than the threshold, they are considered to be of the same type 
        spectral_lengths: a list, storing wavelengths for each hyperspectral channel 
        feature_index_id: deciding which stage of encoder features to use, in range [0, 5]
        GSD: ground sampling distance (m/pixel). list, such as [1.0] or tensor, such as torch.tensor([1.0])

    Returns:
        detection_map: a binary map with 1 for targets and 0 for background
    """

    anns = mask_generator.generate(image, spectral_lengths, GSD)
    mask = anns2mask(anns)
    mask_generator.predictor.set_image(image, False, spectral_lengths, GSD)
    all_features = mask_generator.predictor.model.image_encoder.multi_stage_features[feature_index_id]

    all_features = all_features.detach().cpu()
    all_features = F.interpolate(all_features, (max(mask.shape[1], mask.shape[2]),max(mask.shape[1], mask.shape[2]) ))
    all_features = all_features[:,:, :mask.shape[1], :mask.shape[2]]

    target_locs = find_min_cosine_distance(image, target_spectral)

    mask_index = np.where(mask[:,target_locs[0], target_locs[1]] == 1)[0]
    few_shot_label = mask[mask_index.tolist(), :, :][0,:,:]
    target_locs = np.where(few_shot_label == 1)
    target_feature = all_features[0:1, :, (target_locs[0]), (target_locs[1])]
    target_feature = target_feature.mean((2))[0,:].detach().cpu().numpy()

    mask_number = mask.shape[0]
    detection_map = np.zeros((mask.shape[1], mask.shape[2]))
    for i in range(mask_number):
        seg_mask = mask[i:i+1,:,:]
        locs = np.where(seg_mask == 1)

        seg_mask_feature = all_features[:,:, locs[1], locs[2]].mean(2)[0,:]

        cosine = cosine_similarity(seg_mask_feature.detach().cpu().numpy(), target_feature)

        if cosine > cosine_simi_thresh:
            detection_map[locs[1], locs[2]] = 1
        else:
            detection_map[locs[1], locs[2]] = 0

    return detection_map


def cosine_similarity(vector1, vector2):
    """
    Calculating the cosine similarity between two feature vectors
    """
    dot_product = np.dot(vector1, vector2)
    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)
    return dot_product / (norm_v1 * norm_v2)
    

def find_min_cosine_distance(image, target_spectral):
    """
    Find the pixel with smallest distance from target_spectral
    """
    image_reshaped = image.reshape(-1, image.shape[2])
    target_spectral_repeated = np.tile(target_spectral, (image_reshaped.shape[0], 1))

    cos_distances = np.sum(image_reshaped * target_spectral_repeated, axis=1) / (
        np.linalg.norm(image_reshaped, axis=1) * np.linalg.norm(target_spectral_repeated, axis=1)
    )

    max_index = np.argmax(cos_distances)
    max_index_2d = np.unravel_index(max_index, (image.shape[0], image.shape[1]))
    return max_index_2d
    

def anns2mask(anns):
    """
    Convert the segmentation results from list(str, dict) format to np.array
    """
    if len(anns) == 0:
        print("len=0")
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    res = np.empty((1, anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1]))

    for ann in sorted_anns:
        m = ann['segmentation']
        area_ratio = (ann['area'])/(anns[0]['segmentation'].shape[0]*anns[0]['segmentation'].shape[1])
        if area_ratio > 0.9:
            continue
        locs = np.where(m == True)
        res_t = np.zeros((1, anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1]))
        res_t[0, locs[0], locs[1]] = 1
        res = np.concatenate([res_t, res], axis=0)
    return res


def dilate_binary_image(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Postprocess the detection map with dilation operation
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image


def save_heatmap(data, working_dir='', save_path ='', save_width=20, save_height=20, dpi=300, file_name=''):

    if save_path == '':
        save_path = os.path.join(working_dir, file_name + "heatmap.png")
    plt.figure(figsize=(save_width, save_height))
    sns.heatmap(data, cmap="jet", cbar=False)
    plt.show()
    plt.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight') 
    plt.close()

def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if deterministic:
        torch.backends.cudnn.deterministic = True