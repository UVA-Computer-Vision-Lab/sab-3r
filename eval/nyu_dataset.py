import os
import numpy as np
import mat73

DATA_PATH = os.environ.get("SAB3R_NYU_DATA_PATH", "./data/nyuv2")

def load_nyu_data():
    nyuv2_dict = mat73.loadmat(f"{DATA_PATH}/nyu_depth_v2_labeled.mat")
    return np.transpose(nyuv2_dict['images'], (3, 0, 1, 2)), np.transpose(nyuv2_dict['depths'], (2, 0, 1))