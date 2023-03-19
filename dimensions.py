import pandas
import pydicom

seg_header = pydicom.dcmread("C:/Users/miacr/OneDrive/Documentos/TFG/Dades/HCC_003/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm")
print("seg dimension")
print(seg_header[0x5200,0x9229][0][0x0028,0x9110][0][0x0018,0x0050].description,seg_header[0x5200,0x9229][0][0x0028,0x9110][0][0x0018,0x0050].value)
print(seg_header[0x5200,0x9229][0][0x0028,0x9110][0][0x0028,0x0030].description, seg_header[0x5200,0x9229][0][0x0028,0x9110][0][0x0028,0x0030].value)

CT_header = pydicom.dcmread("C:/Users/miacr/OneDrive/Documentos/TFG/Dades/HCC_003/09-12-1997-NA-AP LIVER-64595/4.000000-Recon 2 LIVER 3 PHASE AP-18688/1-001.dcm")
print("ct dimension")
print(CT_header[0x0018,0x0050].description,CT_header[0x0018,0x0050].value)
print(CT_header[0x0028,0x0030].description, CT_header[0x0028,0x0030].value)

# seg dimension
# <bound method DataElement.description of (0018, 0050) Slice Thickness                     DS: '2.5'> 2.500000e+00
# <bound method DataElement.description of (0028, 0030) Pixel Spacing                       DS: [7.031250e-01, 7.031250e-01]> [7.031250e-01, 7.031250e-01]   
# ct dimension
# <bound method DataElement.description of (0018, 0050) Slice Thickness                     DS: '2.5'> 2.500000
# <bound method DataElement.description of (0028, 0030) Pixel Spacing                       DS: [0.703125, 0.703125]> [0.703125, 0.703125]