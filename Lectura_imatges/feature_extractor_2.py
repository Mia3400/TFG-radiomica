import sys
sys.path.append('C:/Users/miacr/TFG-radiomica/Lectura_imatges')
from Imatge_ct_vs_seg import load_img,load_mask,visualize,find_tag_recursively,get_paths
import SimpleITK as sitk
import matplotlib.pyplot as plt
from collections import Counter
import pandas 
import numpy as np
from radiomics import featureextractor
import os

ROOT = "C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg"

def feature_extractor(patient_idx,image,mask):
    extractor = featureextractor.RadiomicsFeatureExtractor(label = 2)
    features =extractor.execute(image,mask)
    for key,value in dict(features).items():
        if "diagnostics" in key:
            del features[key]
    features.update({'Patient':patient_idx})
    return features


feature_data =[]
patients = []
for patient_idx, img_dirpath, mask_filepath in get_paths(ROOT):
     print(f'Processing {patient_idx}')
     patients.append(patient_idx)
     img_dcmset, img_array = load_img(img_dirpath)
     mask_dcm, mask_array = load_mask(mask_filepath, img_dcmset, img_array)
     mask_array = np.where(mask_array != 2, 0, mask_array)
     mask = sitk.GetImageFromArray(mask_array)
     image = sitk.GetImageFromArray(img_array)
     feature_data.append(feature_extractor(patient_idx,image,mask))
col = Counter()
for k in feature_data:
    col.update(k)
features = pandas.DataFrame([k.values() for k in feature_data], columns = col.keys())
features = features.set_index('Patient')

features.to_csv("feature_data_2.csv")

