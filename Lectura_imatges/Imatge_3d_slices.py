import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import math

dir_path = "C:/Users/miacr/OneDrive/Documentos/TFG/manifest-1676659403087/HCC-TACE-Seg/HCC_001/04-21-2000-NA-CT ABDPEL WC-49771/3.000000-Recon 2 PRE LIVER-07012"
slices = []

for filename in os.listdir(dir_path):
    if filename.endswith(".dcm"):
        ds = pydicom.dcmread(os.path.join(dir_path, filename))
        slices.append(ds)
print("Hay" , len(slices) ,"slices")
        
def Visualitzar_slices(slices, rows=6, cols=6):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = i*(math.trunc(len(slices)/(rows*cols)))
        ax[int(i/rows),int(i % rows)]
        ax[int(i/rows),int(i % rows)].imshow(slices[ind].pixel_array,plt.cm.bone)
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

Visualitzar_slices(slices)