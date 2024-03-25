import random
import pandas as pd
import numpy as np

from alz.alz.viz.brain.clean_labels import VBMDataCleaner
from alz.alz.viz.brain.roi_map import VBMRegionOfInterestMap
from alz.alz.viz.brain.roi_map import FSRegionOfInterestMap

# source : https://gitlab.com/minds-mines/alzheimers/miccai2020-alz-cld/-/blob/master/experiments/draw_brains.py

def draw_brains(path_to_results_dir):
    fs_labels = pd.read_csv("alz/data/fs_atlas_labels.csv")

    vbm_cleaner = VBMDataCleaner()
    vbm_cleaner.load_data("alz/data/longitudinal imaging measures_VBM_mod_final.xlsx")
    vbm_labels = vbm_cleaner.clean()

    path_to_weights = path_to_results_dir + 'W_array/W_all.npy'
    # fs_weights = pd.read_csv("fs_weights.csv").values.flatten()
    # vbm_weights = pd.read_csv("vbm_weights.csv").values.flatten()

    # Create FreeSurfer ROI Map
    fs_roi_map = FSRegionOfInterestMap()
    for index, row in fs_labels.iterrows():
        atlas = row["Atlas"]
        rois = row[atlas].split("+")
        [fs_roi_map.add_roi(roi, fs_weights[index], atlas) for roi in rois]

    fs_roi_map.build_map(smoothed=True)
    #fs_roi_map.plot(time)
    fs_roi_map.save("fs_fig.png", "title")

    # Create VBM ROI Map
    vbm_roi_map = VBMRegionOfInterestMap()
    for label, weight in zip(vbm_labels, vbm_weights):
        vbm_roi_map.add_roi(label, weight)

    vbm_roi_map.build_map(smoothed=True)
    #vbm_roi_map.plot(time)
    vbm_roi_map.save("vbm_fig.png", "title")
