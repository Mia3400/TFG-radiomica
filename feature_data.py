import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor,  imageoperations
from collections import OrderedDict,Counter
import pydicom
from iteration_utilities import duplicates
import pandas

filepath = "C:/Users/miacr/OneDrive/Documentos/TFG/Dades/manifest-1678029820376/"

def feature_data(path_dades):
    feature_data =[]
    patients = []
    metadata_path = path_dades + "metadata.csv"
    metadata_df = pandas.read_csv(metadata_path)

    for i in range(0,len(metadata_df)):
        name = metadata_df.iloc[i,metadata_df.columns.get_indexer(["Data Description URI"])][0]
        if name not in patients:
            if get_seg_ct_filepath(name,metadata_df) is not None:
                feature_data.append(feature_extractor(path_dades,get_seg_ct_filepath(name,metadata_df)))
            patients.append(name)

    col = Counter()
    for k in feature_data:
        col.update(k)
    features = pandas.DataFrame([k.values() for k in feature_data], columns = col.keys())

    features = features.set_index('Patient')
    return features

def get_seg_ct_filepath(name,df):
    selected_rows = pandas.DataFrame()
    patient = df.loc[df["Data Description URI"] == name]
    date = min(list(set(patient.loc[:,"Study Description"].values)))
    if len(patient[patient["Study Date"].isin(["Segmentation"])]) == 0:
        return None
    else:
        selected_rows = patient[(patient["Study Description"] == date) & (patient["Study Date"].isin(["Recon 2 LIVER 3 PHASE AP", "Segmentation"]))]
        return selected_rows.loc[:,["File Location", "Data Description URI","Study Date"]]
    
def feature_extractor(filepath,path_data):
    name = path_data.loc[:,"Data Description URI"].values[0]
    seg_location = (path_data[path_data["Study Date"] == "Segmentation"]).loc[:,"File Location"].values[0]
    image_location = (path_data[path_data["Study Date"] == "Recon 2 LIVER 3 PHASE AP"]).loc[:,"File Location"].values[0]
    
    image_file = os.path.join(filepath,image_location)
    seg_file = os.path.join(filepath,seg_location)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_file)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    seg_image = reader.GetGDCMSeriesFileNames(seg_file)
    seg = reader.Execute()

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder') 
    features =extractor.execute(image, seg)
    for key,value in dict(features).items():
        if  "firstorder" not in key:
            del features[key]
    features.update({'Patient':name})
    return features

print(feature_data(filepath))

#         original_firstorder_10Percentile original_firstorder_90Percentile  ... original_firstorder_Uniformity original_firstorder_Variance
# Patient                                                                    ...
# HCC_003                              1.0                              1.0  ...                            1.0                          0.0
# HCC_004                              1.0                              1.0  ...                            1.0                          0.0