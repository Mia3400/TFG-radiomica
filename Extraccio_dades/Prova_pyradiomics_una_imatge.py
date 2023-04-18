import SimpleITK as sitk
import pandas
from radiomics import featureextractor
from collections import Counter

# Read CT
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_105/07-08-2006-NA-CT CHESTABDPEL LIVER-56667/4.000000-2.5 STANDARD-23481")
reader.SetFileNames(dicom_names)
ct = reader.Execute()

# Read segmentation
seg = sitk.ReadImage("C:/TFG-code/manifest-1643035385102/HCC-TACE-Seg/HCC_105/07-08-2006-NA-CT CHESTABDPEL LIVER-56667/300.000000-Segmentation-72224/1-1.dcm")

# Resample segmentation to CT
#   https://pyradiomics.readthedocs.io/en/latest/faq.html?highlight=tolerance#geometry-mismatch-between-image-and-mask
#   https://github.com/AIM-Harvard/pyradiomics/blob/master/bin/resampleMask.py
rif = sitk.ResampleImageFilter()
rif.SetReferenceImage(ct)
rif.SetOutputPixelType(seg.GetPixelID())
rif.SetInterpolator(sitk.sitkNearestNeighbor)
seg_resampled = rif.Execute(seg)

# Extract features
#   https://pyradiomics.readthedocs.io/en/latest/faq.html?highlight=tolerance#valueerror-label-not-present-in-mask-choose-from
extractor = featureextractor.RadiomicsFeatureExtractor(label=255)
# extractor.disableAllFeatures()
# extractor.enableFeatureClassByName() 
features = extractor.execute(ct, seg_resampled)

# Delete non-features from dict
for key, value in dict(features).items():
    if "diagnostics" in key:
        del features[key]

ord_list = [features]

col = Counter()
for k in ord_list:
    col.update(k)

df = pandas.DataFrame([k.values() for k in ord_list], columns = col.keys())

print(df)