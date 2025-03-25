import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(root_path)

import numpy as np
import torch
import argparse

from HyperFree.utils.spectral_process_utils import read_img, write_img
from HyperFree import SamAutomaticMaskGenerator, HyperFree_model_registry
from prompt_mask_feature_interaction import hyperspectral_OCC, show_ann, evaluate, load_wavelength

"""
HyperFree
Hyperspectral One-Class Classification in Tuning-free Manner
"""


def Argparse():
    parser = argparse.ArgumentParser(
        description='HyperFree Hyperspectral One-Class Classification in Fine-tuning Manner')
    parser.add_argument('-ds', '--data_path', type=str,
                        default='./../../Data/hyperspectral_one_class_classification/subset_honghu.tif',
                        help='Dataset')
    parser.add_argument('-sl', '--wavelengths', type=str,
                        default='./../../Data/hyperspectral_one_class_classification/subset_honghu_wavelengths.txt',
                        help='Central wavelength of sensor')
    parser.add_argument('-i', '--pred_iou_thresh', type=str, default=0.6,
                        help='Controling the predicted mask quality in range [0, 1] of model')
    parser.add_argument('-s', '--stability_score_thresh', type=float, default=0.7,
                        help='Controling the stability of the mask in range [0, 1]')
    parser.add_argument('-f', '--feature_index_id', type=int, default=1,
                        help='Deciding which stage of encoder features to use')
    parser.add_argument('-c', '--class_prior', type=float, default=0.3769,
                        help='The occurrence probability of interested class within the study area')
    parser.add_argument('-g', '--GSD', type=float, default='0.043', help='Ground sampling distance (m/pixel)')
    parser.add_argument('-p', '--prompt_point', nargs='+', type=int, default=[600, 90],
                        help='The index of prompt_point')
    parser.add_argument('-d', '--device', type=str, default='0', help='GPU_ID')

    return parser.parse_args()


if __name__ == '__main__':
    args = Argparse()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # wavelengths = [
    #     401.809998, 404.031006, 406.252014, 408.472992, 410.694000, 412.915009,
    #     415.135986, 417.356995, 419.578003, 421.799988, 424.020996, 426.242004,
    #     428.463013, 430.683990, 432.904999, 435.126007, 437.346985, 439.567993,
    #     441.789001, 444.010010, 446.230988, 448.451996, 450.674011, 452.894989,
    #     455.115997, 457.337006, 459.558014, 461.778992, 464.000000, 466.221008,
    #     468.441986, 470.662994, 472.884003, 475.105011, 477.326996, 479.548004,
    #     481.769012, 483.989990, 486.210999, 488.432007, 490.653015, 492.873993,
    #     495.095001, 497.316010, 499.536987, 501.757996, 503.979004, 506.200989,
    #     508.421997, 510.643005, 512.864014, 515.085022, 517.306030, 519.526978,
    #     521.747986, 523.968994, 526.190002, 528.411011, 530.632019, 532.854004,
    #     535.075012, 537.296021, 539.517029, 541.737976, 543.958984, 546.179993,
    #     548.401001, 550.622009, 552.843018, 555.064026, 557.284973, 559.505981,
    #     561.728027, 563.948975, 566.169983, 568.390991, 570.612000, 572.833008,
    #     575.054016, 577.275024, 579.495972, 581.716980, 583.937988, 586.158997,
    #     588.380981, 590.601990, 592.822998, 595.044006, 597.265015, 599.486023,
    #     601.706970, 603.927979, 606.148987, 608.369995, 610.591003, 612.812012,
    #     615.033020, 617.255005, 619.476013, 621.697021, 623.918030, 626.138977,
    #     628.359985, 630.580994, 632.802002, 635.023010, 637.244019, 639.465027,
    #     641.685974, 643.908020, 646.129028, 648.349976, 650.570984, 652.791992,
    #     655.013000, 657.234009, 659.455017, 661.676025, 663.896973, 666.117981,
    #     668.338989, 670.559998, 672.781982, 675.002991, 677.223999, 679.445007,
    #     681.666016, 683.887024, 686.107971, 688.328979, 690.549988, 692.770996,
    #     694.992004, 697.213013, 699.434998, 701.656006, 703.877014, 706.098022,
    #     708.318970, 710.539978, 712.760986, 714.981995, 717.203003, 719.424011,
    #     721.645020, 723.866028, 726.086975, 728.309021, 730.530029, 732.750977,
    #     734.971985, 737.192993, 739.414001, 741.635010, 743.856018, 746.077026,
    #     748.297974, 750.518982, 752.739990, 754.961975, 757.182983, 759.403992,
    #     761.625000, 763.846008, 766.067017, 768.288025, 770.508972, 772.729980,
    #     774.950989, 777.171997, 779.393005, 781.614014, 783.835999, 786.057007,
    #     788.278015, 790.499023, 792.719971, 794.940979, 797.161987, 799.382996,
    #     801.604004, 803.825012, 806.046021, 808.267029, 810.489014, 812.710022,
    #     814.931030, 817.151978, 819.372986, 821.593994, 823.815002, 826.036011,
    #     828.257019, 830.478027, 832.698975, 834.919983, 837.140991, 839.362976,
    #     841.583984, 843.804993, 846.026001, 848.247009, 850.468018, 852.689026,
    #     854.909973, 857.130981, 859.351990, 861.572998, 863.794006, 866.015991,
    #     868.237000, 870.458008, 872.679016, 874.900024, 877.120972, 879.341980,
    #     881.562988, 883.783997, 886.005005, 888.226013, 890.447021, 892.668030,
    #     894.890015, 897.111023, 899.331970, 901.552979, 903.773987, 905.994995,
    #     908.216003, 910.437012, 912.658020, 914.879028, 917.099976, 919.320984,
    #     921.543030, 923.763977, 925.984985, 928.205994, 930.427002, 932.648010,
    #     934.869019, 937.090027, 939.310974, 941.531982, 943.752991, 945.973999,
    #     948.195007, 950.416992, 952.638000, 954.859009, 957.080017, 959.301025,
    #     961.521973, 963.742981, 965.963989, 968.184998, 970.406006, 972.627014,
    #     974.848022, 977.070007, 979.291016, 981.512024, 983.732971, 985.953979,
    #     988.174988, 990.395996, 992.617004, 994.838013, 997.059021, 999.280029]

    wavelengths = load_wavelength(args.wavelengths)

    pred_iou_thresh = args.pred_iou_thresh  # Controling the model's predicted mask quality in range [0, 1].
    stability_score_thresh = args.stability_score_thresh  # Controling the stability of the mask in range [0, 1].
    feature_index_id = args.feature_index_id  # Deciding which stage of encoder features to use
    class_prior = args.class_prior  # The occurrence probability of interested class within the study area.
    GSD = args.GSD  # Ground sampling distance (m/pixel)

    data_path = args.data_path
    gt_path = args.data_path[:-4] + '-gt.tif'
    ckpt_pth = root_path + "/Ckpt/HyperFree-b.pth"
    save_dir = root_path + "/Outputs/hyperspectral_one_class_classification/"

    img = read_img(data_path)
    if img.max() > 500:
        img_uint8 = np.clip(img, 0, 500)
    else:
        img_normalized = (img - img.min()) / (img.max() - img.min())
        img_uint8 = (255 * img_normalized).astype(np.uint8)
    height, width = img.shape[0], img.shape[1]

    ratio = 1024 / (max(height, width))
    GSD = GSD / ratio
    GSD = torch.tensor([GSD])

    HyperFree = HyperFree_model_registry["vit_b"](checkpoint=ckpt_pth).to(device)
    HyperFree = HyperFree.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=HyperFree,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )

    few_shot_label = np.zeros((height, width))  # Store the binary graph of the interested class
    prompt_point = [600, 90]  # The index of prompt point
    few_shot_label[prompt_point[0], prompt_point[1]] = 1

    detection_map = hyperspectral_OCC(
        mask_generator=mask_generator,
        image=img_uint8,
        few_shot_label=few_shot_label,
        class_prior=class_prior,
        wavelengths=wavelengths,
        feature_index_id=feature_index_id,
        GSD=GSD
    )

    show_ann(
        ann=detection_map,
        save_dir=save_dir
    )

    evaluate(
        ann=detection_map,
        prompt_point=prompt_point,
        gt_path=gt_path)
