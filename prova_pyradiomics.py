import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from radiomics import featureextractor

#Deteca el FIJI per imatges 3D interactives
os.environ['SITK_SHOW_COMMAND'] ="C:/Users/miacr/fiji-win64/Fiji.app/ImageJ-win64.exe"

image=  sitk.ReadImage("C:/Users/miacr/OneDrive/Documentos/TFG/manifest-1676659403087/HCC-TACE-Seg/HCC_001/04-21-2000-NA-CT ABDPEL WC-49771/3.000000-Recon 2 PRE LIVER-07012/1-36.dcm"
)

otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)
otsu_filter.SetOutsideValue(1)
otsu_mask = otsu_filter.Execute(image)

# Visualització de la mask
#img_arr = sitk.GetArrayFromImage(image)
#mask_arr = sitk.GetArrayFromImage(otsu_mask)

#fig, axs = plt.subplots(1, 2)
#axs[0].imshow(img_arr[0,:,:], cmap='gray')
#axs[1].imshow(mask_arr[0,:,:], cmap='gray')
#plt.show

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableImageTypeByName('Original') 
extractor.enableFeatureClassByName('shape') 

# Calculate the radiomics features
features = extractor.execute(image,otsu_mask)
