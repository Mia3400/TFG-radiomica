import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io, transform
import skimage.transform
from PIL import Image
import numpy
from skimage.transform import resize
import os

#Detecta el FIJI per imatges 3D interactives
os.environ['SITK_SHOW_COMMAND'] ="C:/Users/miacr/fiji-win64/Fiji.app/ImageJ-win64.exe"

image_path = "C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_035/03-16-2004-NA-CAP LIVER-74044/6.000000-Recon 3 LIVER 2 PHASE CAP-31168"
seg_path = "C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_035/03-16-2004-NA-CAP LIVER-74044/300.000000-Segmentation-21743/1-1.dcm"

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(image_path)
reader.SetFileNames(dicom_names)
image = reader.Execute()

image_array = sitk.GetArrayFromImage(image).astype(float)
#Mostrar la imatge
# plt.imshow(image_array[0,:,:])
# sitk.Show(image)

seg= sitk.ReadImage(seg_path)
rif = sitk.ResampleImageFilter()
rif.SetReferenceImage(image)
rif.SetOutputPixelType(seg.GetPixelID())
rif.SetInterpolator(sitk.sitkNearestNeighbor)
seg_resampled = rif.Execute(seg)

seg_array = numpy.flip(sitk.GetArrayFromImage(seg_resampled).astype(float),axis = 0)
seg_resampled = sitk.GetImageFromArray(seg_array)


#Mostrar la imatge
# plt.imshow(seg_array[0,:,:])
# sitk.Show(seg_resampled)

# print(seg_array.shape)
ct_slice = image_array[67,:,:]
# plt.imshow(ct_slice, plt.cm.Greys)
seg_slice = seg_array[67,:,:]
# plt.imshow(seg_slice, plt.cm.Greys) 
# plt.show()


plt.imshow(ct_slice, plt.cm.bone) # I would add interpolation='none'

plt.imshow(seg_slice, cmap='jet', alpha=0.4)
plt.show()