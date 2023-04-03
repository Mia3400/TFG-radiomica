import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import skimage.transform
import numpy as np
from skimage import data
import cv2
from skimage import io, transform
from skimage.transform import resize

#Detecta el FIJI per imatges 3D interactives
os.environ['SITK_SHOW_COMMAND'] ="C:/Users/miacr/fiji-win64/Fiji.app/ImageJ-win64.exe"

reader = sitk.ImageSeriesReader()
#Path de les imatges DICOM + lector de una succesió de diapositives
seg_path= sitk.ReadImage("C:/TFG-code/manifest-1680446438379/HCC-TACE-Seg/HCC_001/11-30-1999-NA-CT-CAP WWO CON-00377/300.000000-Segmentation-99942/1-1.dcm")

seg_arr = sitk.GetArrayFromImage(seg_path).astype(float)

seg_red = skimage.transform.resize(seg_arr, (43,512,512),preserve_range=True)

# rotated_seg_image = np.zeros_like(seg)

# Loop over each slice of the SEG image and rotate it by 180 degrees
# for i in range(seg.shape[0]):
#     rotated_seg_image[i] = np.rot90(np.rot90(seg[i]))
# seg = np.flip(seg, axis = 2)
# seg = cv2.flip(seg,flipCode=0)
seg_red[seg_red > 0 ] = 1

seg_sitk = sitk.GetImageFromArray(seg_red)
print(seg_sitk.GetSize())


#Mostrar la imatge

#sitk.Show(seg_sitk)
