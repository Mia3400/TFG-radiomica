import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io, transform
import skimage.transform
from skimage.transform import resize
import os

#Detecta el FIJI per imatges 3D interactives
os.environ['SITK_SHOW_COMMAND'] ="C:/Users/miacr/fiji-win64/Fiji.app/ImageJ-win64.exe"

reader = sitk.ImageSeriesReader()
#Path de les imatges DICOM + lector de una succesió de diapositives
dicom_names = reader.GetGDCMSeriesFileNames("C:/TFG-code/manifest-1680446438379/HCC-TACE-Seg/HCC_001/11-30-1999-NA-CT-CAP WWO CON-00377/3.000000-C-A-P-42120")
reader.SetFileNames(dicom_names)
image = reader.Execute()


image_array = sitk.GetArrayFromImage(image).astype(float)


#Mostrar la imatge
plt.imshow(image_array[0,:,:])
sitk.Show(image)
