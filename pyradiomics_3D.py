import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, Counter
import pandas 
from radiomics import featureextractor,  imageoperations

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("C:/Users/miacr/OneDrive/Documentos/TFG/Dades/manifest-1678029820376/HCC-TACE-Seg/HCC_003/09-12-1997-NA-AP LIVER-64595/4.000000-Recon 2 LIVER 3 PHASE AP-18688")
reader.SetFileNames(dicom_names)
image = reader.Execute()
seg_image = reader.GetGDCMSeriesFileNames("C:/Users/miacr/OneDrive/Documentos/TFG/Dades/HCC_003/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632")
reader.SetFileNames(seg_image)
seg = reader.Execute()


extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder') 
features =extractor.execute(image, seg)
for key, value in dict(features).items():
    if  "firstorder" not in key:
        del features[key]

ord_list = [features]

col = Counter()
for k in ord_list:
    col.update(k)

df = pandas.DataFrame([k.values() for k in ord_list], columns = col.keys())

print(df)