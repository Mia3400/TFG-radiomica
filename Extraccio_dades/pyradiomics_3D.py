import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom 
from collections import OrderedDict, Counter
import skimage.transform
from skimage import data
from skimage import io, transform
from skimage.transform import resize
import pandas 
from radiomics import featureextractor,  imageoperations


reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("C:/Users/miacr/OneDrive/Documentos/TFG/Dades/manifest-1678029820376/HCC-TACE-Seg/HCC_004/10-06-1997-NA-LIVERPELVIS-14785/4.000000-Recon 2 LIVER 3 PHASE AP-69759")
reader.SetFileNames(dicom_names)
image = reader.Execute()
image_array =  sitk.GetArrayFromImage(image).astype(float)

seg_path= sitk.ReadImage("C:/Users/miacr/OneDrive/Documentos/TFG/Dades/manifest-1678029820376/HCC-TACE-Seg/HCC_004/10-06-1997-NA-LIVERPELVIS-14785/300.000000-Segmentation-39561/1-1.dcm")

seg_arr = sitk.GetArrayFromImage(seg_path).astype(float)

seg_red = skimage.transform.resize(seg_arr, (image.GetSize()[2],512,512))
# rotated_seg_image = np.zeros_like(seg)

# # Loop over each slice of the SEG image and rotate it by 180 degrees
# for i in range(seg.shape[0]):
#     rotated_seg_image[i] = np.rot90(np.rot90(seg[i]))
# seg = np.flip(seg, axis = 2)
# seg = cv2.flip(seg,flipCode=0)
seg_red[seg_red > 0 ] = 1

seg = sitk.GetImageFromArray(seg_red)


extractor = featureextractor.RadiomicsFeatureExtractor(geometryTolerance = 1000)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder') 
features =extractor.execute(image_array, seg_red)
for key, value in dict(features).items():
    if  "firstorder" not in key:
        del features[key]

ord_list = [features]

col = Counter()
for k in ord_list:
    col.update(k)

df = pandas.DataFrame([k.values() for k in ord_list], columns = col.keys())

print(df)