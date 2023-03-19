import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

#Detecta el FIJI per imatges 3D interactives
os.environ['SITK_SHOW_COMMAND'] ="C:/Users/miacr/fiji-win64/Fiji.app/ImageJ-win64.exe"

reader = sitk.ImageSeriesReader()
#Path de les imatges DICOM + lector de una succesió de diapositives
dicom_names = reader.GetGDCMSeriesFileNames("C:/Users/miacr/OneDrive/Documentos/TFG/Dades/HCC_003/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632"
)
reader.SetFileNames(dicom_names)
image = reader.Execute()

image_array = sitk.GetArrayFromImage(image)

#Mostrar la imatge
plt.imshow(image_array[0,:,:])

sitk.Show(image)
