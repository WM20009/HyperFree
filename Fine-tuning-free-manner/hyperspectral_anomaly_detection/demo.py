import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import numpy as np
import torch

from HyperFree import SamAutomaticMaskGenerator, HyperFree_model_registry
from prompt_mask_feature_interaction import save_heatmap, hyperspectral_anomaly_detection, compute_auc, enhance_contrast_histogram
import scipy.io as sio

"""
HyperFree
Hyperspectral Anomaly Detection in Tuning-free Manner
"""

device = 'cuda:4'
path = './../../Data/hyperspectral_anomaly_detection/abu-beach-2.mat'
img = sio.loadmat(path)['data']
gt = sio.loadmat(path)['map']

wavelengths = [429.410004,  439.230011,  449.059998,  458.890015,  468.709991,  478.540009,
  488.369995,  498.190002,  508.019989,  517.840027,  527.669983,  537.489990,
  547.320007,  557.140015,  566.960022,  576.789978,  586.609985,  596.429993,
  606.250000,  616.080017,  625.900024,  635.719971,  645.539978,  655.359985,
  665.179993,  675.000000,  682.789978,  692.330017,  701.869995,  711.409973,
  720.950012,  730.489990,  740.030029,  749.570007,  759.119995,  768.659973,
  778.200012,  787.750000,  797.289978,  806.840027,  816.390015,  825.929993,
  835.479980,  845.030029,  854.580017,  864.119995,  873.669983,  883.219971,
  892.770020,  902.330017,  911.880005,  921.429993,  930.979980,  946.349976,
  955.760010,  965.169983,  974.580017,  983.989990,  993.390015, 1002.799988,
 1012.210022, 1021.619995, 1031.030029, 1040.439941, 1049.839966, 1059.250000,
 1068.660034, 1078.060059, 1087.469971, 1096.880005, 1106.280029, 1115.689941,
 1125.099976, 1134.500000, 1143.910034, 1153.310059, 1162.719971, 1172.119995,
 1181.520020, 1190.930054, 1200.329956, 1209.729980, 1219.140015, 1228.540039,
 1237.939941, 1247.349976, 1256.750000, 1265.540039, 1275.510010, 1285.479980,
 1295.459961, 1305.430054, 1315.400024, 1325.369995, 1335.339966, 1345.300049,
 1425.030029, 1434.989990, 1444.959961, 1454.920044, 1464.880005, 1474.839966,
 1484.800049, 1494.760010, 1504.719971, 1514.680054, 1524.640015, 1534.589966,
 1544.550049, 1554.510010, 1564.459961, 1574.420044, 1584.369995, 1594.319946,
 1604.280029, 1614.229980, 1624.180054, 1634.130005, 1644.089966, 1654.040039,
 1663.989990, 1673.939941, 1683.880005, 1693.829956, 1703.780029, 1713.729980,
 1723.670044, 1733.619995, 1743.560059, 1753.510010, 1763.449951, 1773.400024,
 1783.339966, 1793.280029, 1803.219971, 1941.319946, 1951.369995, 1961.420044,
 1971.469971, 1981.510010, 1991.550049, 2001.589966, 2011.630005, 2021.660034,
 2031.689941, 2041.719971, 2051.750000, 2061.770020, 2071.790039, 2081.810059,
 2091.820068, 2101.830078, 2111.840088, 2121.850098, 2131.860107, 2141.860107,
 2151.860107, 2161.850098, 2171.850098, 2181.840088, 2191.830078, 2201.810059,
 2211.800049, 2221.780029, 2231.760010, 2241.729980, 2251.709961, 2261.679932,
 2271.649902, 2281.610107, 2291.570068, 2301.530029, 2311.489990, 2321.449951,
 2331.399902, 2341.350098, 2351.300049, 2361.239990, 2371.179932, 2381.120117,
 2391.060059, 2400.989990, 2410.929932, 2420.850098, 2430.780029, 2440.709961,
 2450.629883, 2460.550049]  # list, input wavelenths

pred_iou_thresh = 0.4 # Controling the model's predicted mask quality in range [0, 1].
stability_score_thresh = 0.4 # Controling the stability of the mask in range [0, 1].
ckpt_pth = "./../../Ckpt/HyperFree-b.pth"
area_ratio_threshold = 0.0009 # Decide how small the targets are anomalies

GSDS = 7.5 # Ground sampling distance (m/pixel)
save_dir = "./../../Outputs/hyperspectral_anomaly_detection/" # Location to ouput anomaly heatmap

if img.max() > 500:
    #img_uint8 = img/(img.max()/500)
    #img_uint8 = enhance_contrast_histogram(img_uint8)
    img_uint8 = img
else:
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_uint8 = (255 * img_normalized).astype(np.uint8)

ratio = 1024/(max(img_uint8.shape[0], img_uint8.shape[1]))
GSDS = GSDS/ratio
GSDS = torch.tensor([GSDS])

HyperFree = HyperFree_model_registry["vit_b"](checkpoint=ckpt_pth).to(device)
HyperFree = HyperFree.to(device)
mask_generator = SamAutomaticMaskGenerator(HyperFree, pred_iou_thresh = pred_iou_thresh, stability_score_thresh = stability_score_thresh, points_per_side=96)

masks = mask_generator.generate(img_uint8, wavelengths,GSDS)
anomaly_map = hyperspectral_anomaly_detection(masks, area_ratio_threshold)
compute_auc(gt, anomaly_map)
save_heatmap(data=anomaly_map, working_dir = save_dir)
