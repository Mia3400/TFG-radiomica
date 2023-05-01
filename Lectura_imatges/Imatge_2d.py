import pydicom
import matplotlib.pyplot as plt

#Visualització d'un tall per matplotlib 
path = "C:/Users/miacr/OneDrive/Documentos/TFG/manifest-1676659403087/HCC-TACE-Seg/HCC_001/04-21-2000-NA-CT ABDPEL WC-49771/3.000000-Recon 2 PRE LIVER-07012/1-36.dcm"
ds = pydicom.dcmread(path)
plt.imshow(ds.pixel_array, plt.cm.Greys) 
plt.show()