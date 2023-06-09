import sys
from Imatge_ct_vs_seg import load_img,load_mask,visualize,find_tag_recursively,get_paths,visualize
import SimpleITK as sitk
import matplotlib.pyplot as plt
from collections import Counter
import pandas 
import numpy as np
from radiomics import featureextractor
import os

img_dirpath = "C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_059/05-10-2002-NA-LIVERC AP-80478/6.000000-Recon 3 LIVER 2PHASE CAP-44613"
mask_filepath = "C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_059/05-10-2002-NA-LIVERC AP-80478/300.000000-Segmentation-32669"

img_dcmset, img = load_img(img_dirpath)
mask_dcm, mask = load_mask(mask_filepath, img_dcmset, img)
mask= np.where(mask != 2, 0, mask)
# mask = sitk.GetImageFromArray(mask_array)
print(f'Img: {img.shape}; Mask: {mask.shape}')

pixel_spacing = find_tag_recursively(mask_dcm, 'PixelSpacing')
slice_spacing = find_tag_recursively(mask_dcm, 'SpacingBetweenSlices')
px_size = (slice_spacing.value.real, pixel_spacing.value[0].real, pixel_spacing.value[1].real)

visualize("HCC_027", img, mask, px_size=px_size, axial_slices=0, coronal_slices=1, show=True)
