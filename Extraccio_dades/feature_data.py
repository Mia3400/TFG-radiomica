import SimpleITK as sitk
import os
from radiomics import featureextractor
from collections import Counter
from datetime import datetime
import pandas

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
    if  len(patient[(patient["Study Description"] == date) & (patient["Study Date"].isin(["Recon 2 LIVER 3 PHASE AP","2.5 SOFT","Recon 3 LIVER 2 PHASE CAP","LIVER 3 PHASE AP","Recon 2 LIVER  2PHASE WITH CON","Recon 3 LIVER 3 PHASE CAP","Recon 2 3 PHASE LIVER ABD", "Recon 2 LIVER 3 PHASE CAP","Recon 2 LIVER 2PHASE CAP","Recon 2 LIVER 2 PHASE CAP","2.5 STANDARD"]))])==0:
        return None
    selected_rows = patient[(patient["Study Description"] == date) & (patient["Study Date"].isin(["Recon 2 LIVER 3 PHASE CAP","2.5 SOFT","Recon 3 LIVER 2 PHASE CAP","Recon 3 LIVER 3 PHASE CAP","Recon 2 LIVER  2PHASE WITH CON","LIVER 3 PHASE AP","Recon 2 3 PHASE LIVER ABD","Recon 2 LIVER 3 PHASE AP","Recon 2 LIVER 2PHASE CAP","Recon 2 LIVER 2 PHASE CAP", "Segmentation","2.5 STANDARD"]))]
    if selected_rows.empty:
        return None
    return selected_rows.loc[:,["File Location", "Data Description URI","Study Date","Number of Images"]]
    
def feature_extractor(filepath,path_data):
    name = path_data.loc[:,"Data Description URI"].values[0]
    seg_location = (path_data[path_data["Study Date"] == "Segmentation"]).loc[:,"File Location"].values[0]
    # ct_images = path_data[path_data["Study Date"].isin(["Recon 2 LIVER 3 PHASE AP","Recon 2 3 PHASE LIVER ABD","Recon 3 LIVER 2 PHASE CAP","2.5 SOFT","Recon 2 LIVER 2 PHASE CAP","Recon 2 LIVER  2PHASE WITH CON","Recon 3 LIVER 3 PHASE CAP","LIVER 3 PHASE AP","Recon 2 LIVER 3 PHASE CAP", "Recon 2 LIVER 2PHASE CAP","2.5 STANDARD"])]
    ct_images_path = path_data.loc[:,["File Location","Number of Images"]]
    max_slices = max(ct_images_path.loc[:,"Number of Images"].values)
    image_location = ct_images_path[ct_images_path["Number of Images"]==max_slices].loc[:,"File Location"].values[0]
    image_file = os.path.join(filepath,image_location)
    seg_file = os.path.normpath(os.path.join(filepath,seg_location,"1-1.dcm"))

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_file)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    seg= sitk.ReadImage(seg_file)
    rif = sitk.ResampleImageFilter()
    rif.SetReferenceImage(image)
    rif.SetOutputPixelType(seg.GetPixelID())
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_resampled = rif.Execute(seg)

    extractor = featureextractor.RadiomicsFeatureExtractor(label=255)
    features =extractor.execute(image, seg_resampled)
    for key,value in dict(features).items():
        if "diagnostics" in key:
            del features[key]
    features.update({'Patient':name})
    return features