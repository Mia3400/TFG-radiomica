import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor,  imageoperations
from collections import OrderedDict,Counter
import pydicom
from datetime import datetime
from iteration_utilities import duplicates
import pandas
import skimage.transform
from skimage import data
from skimage import io, transform
from skimage.transform import resize

filepath = "C:/TFG-code/manifest-1680446438379/"
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
            else:
                print(name,"no tiene suficiente información")
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
    dates = list(set(patient.loc[:,"Study Description"].values))
    min_date = min([datetime.strptime(date, '%m-%d-%Y') for date in dates])
    date =min_date.strftime('%m-%d-%Y')
    if len(patient[(patient["Study Description"] == date) & (patient["Study Date"]== "Segmentation")]) == 0:
        return None
    if  len(patient[(patient["Study Description"] == date) & (patient["Study Date"].isin(["Recon 2 LIVER 3 PHASE AP", "Recon 2 LIVER 3 PHASE CAP","Recon 2 LIVER 2PHASE CAP"]))])==0:
        return None
    selected_rows = patient[(patient["Study Description"] == date) & (patient["Study Date"].isin(["Recon 2 LIVER 3 PHASE CAP","Recon 2 LIVER 3 PHASE AP","Recon 2 LIVER 2PHASE CAP", "Segmentation"]))]
    if selected_rows.empty:
        return None
    return selected_rows.loc[:,["File Location", "Data Description URI","Study Date"]]
    
def feature_extractor(filepath,path_data):
    name = path_data.loc[:,"Data Description URI"].values[0]
    seg_location = (path_data[path_data["Study Date"] == "Segmentation"]).loc[:,"File Location"].values[0]
    image_location = (path_data[path_data["Study Date"].isin(["Recon 2 LIVER 3 PHASE AP","Recon 2 LIVER 3 PHASE CAP", "Recon 2 LIVER 2PHASE CAP"])]).loc[:,"File Location"].values[0]

    image_file = os.path.join(filepath,image_location)
    seg_file = os.path.normpath(os.path.join(filepath,seg_location,"1-1.dcm"))

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_file)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    seg_path= sitk.ReadImage(seg_file)
    seg_arr = sitk.GetArrayFromImage(seg_path).astype(float)
    seg_red = skimage.transform.resize(seg_arr, (image.GetSize()[2],512,512))

    seg_red[seg_red > 0 ] = 1
    seg = sitk.GetImageFromArray(seg_red)

    extractor = featureextractor.RadiomicsFeatureExtractor(geometryTolerance = 1000)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder') 
    features =extractor.execute(image, seg)
    for key,value in dict(features).items():
        if  "firstorder" not in key:
            del features[key]
    features.update({'Patient':name})
    return features


#  original_firstorder_10Percentile original_firstorder_90Percentile  ... original_firstorder_Uniformity original_firstorder_Variance
# Patient                                                                    ...
# HCC_002                            -85.0                            122.0  ...            0.12300849001467277           25978.967140903456
# HCC_003                            -87.0                            240.0  ...            0.07845103287706469            24496.12233357337
# HCC_004                            -90.0                            243.0  ...            0.07447756182937455           20984.161205143224
# HCC_005                           -130.0                             87.0  ...            0.10830028890005927           15437.265207713051
# HCC_006                            -77.0                            140.0  ...            0.09171696410415671            14247.06296710463
# HCC_007                            -79.0                            152.0  ...            0.11793326885775714           21074.392922913223 