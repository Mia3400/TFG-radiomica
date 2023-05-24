from Imatge_ct_vs_seg import load_img,load_mask,visualize,find_tag_recursively
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor
import os
#Intentaré emprar el teu mètode per extreure l'image i la mask i emprar el feature extractr
os.environ['SITK_SHOW_COMMAND'] ="C:/Users/miacr/fiji-win64/Fiji.app/ImageJ-win64.exe"

image_path = "C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_028/10-30-1999-NA-abdpelvis-98668/6.000000-Recon 3 LIVER 3 PHASE AP-38475"
seg_path = "C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_028/10-30-1999-NA-abdpelvis-98668/300.000000-Segmentation-11681/1-1.dcm"

img_dcmset, img = load_img(image_path)
mask_dcm, mask = load_mask(seg_path, img_dcmset, img)
pixel_spacing = find_tag_recursively(mask_dcm, 'PixelSpacing')
slice_spacing = find_tag_recursively(mask_dcm, 'SpacingBetweenSlices')
px_size = (slice_spacing.value.real, pixel_spacing.value[0].real, pixel_spacing.value[1].real)

mask_prova = np.where(mask != 2, 0, mask)
mask_prova_image = sitk.GetImageFromArray(mask_prova)

image = sitk.GetImageFromArray(img)

extractor = featureextractor.RadiomicsFeatureExtractor(label = 2)
features =extractor.execute(image, mask_prova_image)
for key,value in dict(features).items():
    if "diagnostics" in key:
        del features[key]
    features.update({'Patient':"HCC_028"})

print(features)


