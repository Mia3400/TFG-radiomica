import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

#Detecta el FIJI per imatges 3D interactives
os.environ['SITK_SHOW_COMMAND'] ="C:/Users/miacr/fiji-win64/Fiji.app/ImageJ-win64.exe"

reader = sitk.ImageSeriesReader()
#Path de les imatges DICOM + lector de una succesió de diapositives
dicom_names = reader.GetGDCMSeriesFileNames("C:/Users/miacr/OneDrive/Documentos/TFG/manifest-1676659403087/HCC-TACE-Seg/HCC_001/04-21-2000-NA-CT ABDPEL WC-49771/3.000000-Recon 2 PRE LIVER-07012"
)
reader.SetFileNames(dicom_names)
image = reader.Execute()

image_array = sitk.GetArrayFromImage(image)

#Mostrar la imatge
plt.imshow(image_array[0,:,:])

sitk.Show(image)
