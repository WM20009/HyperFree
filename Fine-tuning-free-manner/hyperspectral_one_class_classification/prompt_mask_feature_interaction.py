import os
import re
import cv2
import numpy as np
import torch.nn.functional as F
from HyperFree.utils.spectral_process_utils import write_img, read_img
from sklearn.metrics import precision_score, recall_score, f1_score

"""
Utilizing HyperFree for hyperspectral one class classification tasks directly without fine tuning.
We develop different Prompt—Mask—Feature interaction workflows for each task.
"""

def hyperspectral_OCC(mask_generator, image, few_shot_label, class_prior, spectral_lengths, feature_index_id, GSD):
    """
    Utilizing HyperFree for hyperspectral one class classification directly without fine tuning.
    Args:
        mask_generator: object instance from SamAutomaticMaskGenerator
        image: input hyperspectral image with shape [H, W, C] in range [0,255]
        few_shot_label: a binary map with shape [H, W], where the non-zero location represents the corresponding sample
        class_prior: a float number represneting the area ratio of target object
        spectral_lengths: a list, storing wavelengths for each hyperspectral channel
        feature_index_id: deciding which stage of encoder features to use, in range [0, 5]
        GSD: ground sampling distance (m/pixel). list, such as [1.0] or tensor, such as torch.tensor([1.0])

    Returns:
        detection_res[least_index]: binary map with shape [H, W]
    """
    anns = mask_generator.generate(image, spectral_lengths, GSD)
    mask = mask_generator.anns2mask(anns)
    mask_generator.predictor.set_image(image, True, spectral_lengths, GSD)

    all_features = mask_generator.predictor.model.image_encoder.multi_stage_features[feature_index_id]

    all_features = all_features.detach().cpu()
    all_features = F.interpolate(all_features,
                                 (max(mask.shape[1], mask.shape[2]), max(mask.shape[1], mask.shape[2])))
    all_features = all_features[:, :, :mask.shape[1], :mask.shape[2]]

    target_features = []
    detection_map = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)

    target_locs = np.where(few_shot_label == 1)
    mask_index = np.where(mask[:, target_locs[0], target_locs[1]] == 1)[0]
    assert mask_index.size != 0, ("The setting hyper-parameters lead to no mask in given target location")
    few_shot_label = mask[mask_index.tolist(), :, :][0, :, :]

    target_locs = np.where(few_shot_label == 1)
    target_feature = all_features[0:1, :, (target_locs[0]), (target_locs[1])]
    target_feature = target_feature.mean((2))[0, :].detach().cpu().numpy()
    target_features.append(target_feature)

    least_distance = 1e+3
    least_index = -1
    cosine_t_index = 0

    detection_res = []
    for cosine_t in np.arange(0.8, 0.98, 0.01):
        mask_number = mask.shape[0]
        detection_map *= 0

        for i in range(mask_number - 1):

            seg_mask = mask[i:i + 1, :, :]
            locs = np.where(seg_mask == 1)

            seg_mask_feature = all_features[:, :, locs[1], locs[2]].mean(2)[0, :]

            highest_score = -1
            for j in range(len(target_features)):
                target_feature = target_features[j]
                cosine = mask_generator.cosine_similarity(seg_mask_feature.detach().cpu().numpy(), target_feature)
                if cosine > highest_score:
                    highest_score = cosine

            if cosine > cosine_t:
                detection_map[locs[1], locs[2]] = 1
            else:
                detection_map[locs[1], locs[2]] = 0

        ratio = np.sum(detection_map) / (
                detection_map.shape[0] * detection_map.shape[1])

        distance = abs(ratio - class_prior)
        if ratio > 0 and distance < least_distance:
            least_index = cosine_t_index
            least_distance = distance

        cosine_t_index += 1
        detection_res.append(np.copy(detection_map))

    return detection_res[least_index]


def show_ann(ann, save_dir=''):
    if ann.size == 0:
        print("len=0")
        return

    res = np.zeros((ann.shape[0], ann.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    locs = np.where(ann[:, :] == True)
    res[locs[0], locs[1], :] = np.array(color_mask) * 255

    save_path = os.path.join(save_dir, 'Interested class')
    cv2.imwrite(save_path + '.png', res)
    write_img(ann, save_path + '.tif')
    print('The result have saved to {}'.format(save_dir))


def evaluate(ann, prompt_point, gt_path=''):
    gt = read_img(gt_path)
    class_id = gt[prompt_point[0], prompt_point[1]]

    # Obtain the mask of non-background pixels in gt image
    mask = gt.astype(bool).reshape(-1)

    gt[np.where(gt != class_id)] = 0
    gt[np.where(gt == class_id)] = 1

    gt = gt.reshape(-1)
    ann = ann.reshape(-1)

    gt = gt[mask]
    ann = ann[mask]

    precision = precision_score(gt, ann, pos_label=1, zero_division=0)
    recall = recall_score(gt, ann, pos_label=1, zero_division=0)
    f1 = f1_score(gt, ann, pos_label=1, zero_division=0)

    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    print("f1:{}".format(f1))


def load_wavelength(path):
    with open(path, 'r') as file:
        content = file.read()

    keyword_index = content.find('wavelength')
    if keyword_index == -1:
        raise ValueError(f"Keyword 'wavelength' not found in file.")

    equals_index = content.find('=', keyword_index)
    if equals_index == -1:
        raise ValueError("Equals sign '=' not found after keyword.")

    list_start_index = content.find('[', equals_index)
    if list_start_index == -1:
        raise ValueError("List start '[' not found after equals sign.")

    # 定定位到列表的结束
    list_end_index = content.find(']', list_start_index)
    if list_end_index == -1:
        raise ValueError("List end ']' not found.")

    list_str = content[list_start_index + 1:list_end_index].strip()

    wavelength_list = [float(num) for num in list_str.split(',') if num.strip()]

    return wavelength_list
