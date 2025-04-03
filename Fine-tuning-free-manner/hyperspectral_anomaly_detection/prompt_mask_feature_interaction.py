import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import integrate


"""
Utilizing HyperFree for hyperspectral anomaly detection task directly without fine tuning.
We develop different Prompt—Mask—Feature interaction workflows for each task.
"""

def save_heatmap(data, working_dir='', save_path ='', save_width=20, save_height=20, dpi=300, file_name=''):

    if save_path == '':
        save_path = os.path.join(working_dir, file_name + "_heatmap.png")
    plt.figure(figsize=(save_width, save_height))
    sns.heatmap(data, cmap="jet", cbar=False)
    plt.show()
    plt.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight') 
    plt.close()


def hyperspectral_anomaly_detection(anns, area_ratio_threshold = 0.0009):
    """
    Filter out targets with area ratios below area_ratio_threshold as anomalies
    Args:
        anns: (list): A list over records for masks.
        area_ratio_threshold (float): Decide how small the targets are anomalies

    Returns:
        binary map: 1 represents anomalies and 0 represnets the background
    """

    if len(anns) == 0:
        print("Annotations len=0")
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    res = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1]))

    for ann in sorted_anns:
        m = ann['segmentation']
        if (ann['area'])/(anns[0]['segmentation'].shape[0]*anns[0]['segmentation'].shape[1]) >= area_ratio_threshold:
            continue
        
        locs = np.where(m == True)
        res[locs[0], locs[1]] = 1

    return res


def compute_auc(gt, anomaly_map):
    y_score, y_true = [], []
    a = anomaly_map[np.where(gt == 0)]  
    b = anomaly_map[np.where(gt == 1)]
    y_score = a.tolist()  
    y_true += np.zeros(len(a)).tolist()
    y_score += b.tolist()
    y_true += np.ones(len(b)).tolist() 

    scoreDF = roc_auc_score(y_true, y_score)

    print("scoreDF: " + str(scoreDF))

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    scoreDt = abs(integrate.trapz(tpr, thresholds))
    scoreFt = abs(integrate.trapz(fpr, thresholds))
    scoreTD = scoreDF + scoreDt
    scoreBS = scoreDF - scoreFt
    scoreODP = scoreDF + scoreDt - scoreFt
    scoreTDBS = scoreDt - scoreFt
    scoreSNPR = scoreDt / scoreFt
    print("scoreDt: " + str(scoreDt))
    print("scoreFt: " + str(scoreFt))
    print("scoreTD: " + str(scoreTD))
    print("scoreBS: " + str(scoreBS))
    print("scoreODP: " + str(scoreODP))
    print("scoreTDBS: " + str(scoreTDBS))
    print("scoreSNPR: " + str(scoreSNPR))

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
