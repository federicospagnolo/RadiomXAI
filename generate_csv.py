################
#written by Federico Spagnolo
#usage: python generate_csv.py
################

import numpy as np
import nibabel as nib
import os
import sys
import glob
import six
from radiomics import featureextractor, imageoperations
import SimpleITK as sitk
import pandas as pd
import logging
import scipy.ndimage as ndimage
from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm

# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

# Groups to evaluate and current folder
groups = ["FP", "TP"]

# Dilation structure for lesions
#struct1 = np.array(ndimage.generate_binary_structure(3, 2)) # define shape of dilation

# Check if the script is running in an interactive environment
if hasattr(sys, 'ps1'):
    script_folder = '/home/fede/storage/groups/think/Federico/Radiomics'
else:
    script_folder = os.path.dirname(os.path.realpath(__file__))

# Radiomics parameters
params = {}
params['binWidth'] = 10
params['sigma'] = [1,2,3]
params['verbose'] = False

# Define extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**params, provenance_on=True)
print("Extraction parameters:\n\t", extractor.settings)

# Enable all features
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
#extractor.enableFeatureClassByName('shape')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('ngtdm')

def process_session(session, group):
    saliency_files = glob.glob(session + 'flair*')
    maskName = glob.glob(session + 'group*')
    if maskName:
        maskName = glob.glob(session + 'group*')[0]
    else:
        return    
    mask = nib.load(maskName).get_fdata()
    affine = nib.load(maskName).affine

    # Dilate lesion masks
    d_mask = ndimage.maximum_filter(mask, 5)
    d_mask[mask != 0] = mask[mask != 0]
    nib.save(nib.Nifti1Image(d_mask, affine), session + "dilated.nii.gz")
    maskName = session + "dilated.nii.gz"

    data = []

    for imageName in saliency_files:
        # Load the image using SimpleITK
        image = sitk.ReadImage(imageName)
        
        # Normalize the image using PyRadiomics function
        normalized_image = imageoperations.normalizeImage(image)
        
        # Save the normalized image to a temporary file
        normalized_image_name = session + "normalized_" + imageName.split('/')[-1]
        sitk.WriteImage(normalized_image, normalized_image_name)

        # Calculate the features (segment-based):
        label = int((imageName.split("les_", 1)[1]).split("_4031", 1)[0])
        result = extractor.execute(normalized_image_name, maskName, label=label)
        items = list(six.iteritems(result))
        for key, val in items[22:]:
            data.append({"key": key, "val": float(val), "group": group, "candidate": imageName})
    
    # Remove temporary files
    os.remove(maskName)
    for imageName in saliency_files:
        normalized_image_name = session + "normalized_" + imageName.split('/')[-1]
        os.remove(normalized_image_name)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    print("Processing sessions...")
    df = pd.DataFrame()
    
    for group in tqdm(groups):
        path = os.path.join(script_folder, f"../IES/attention_maps/{group}/")
        sessions = glob.glob(path + '*/')
        args = [(session, group) for session in sessions]

        with Pool(processes=12) as pool:
            dfs = pool.starmap(process_session, args)

        for d in dfs:
            df = pd.concat([df, d], ignore_index=True, sort=False)
    
    # Pivot the DataFrame
    df['key_id'] = df.groupby(['key', 'group']).cumcount()
    dataframe = df.pivot_table(index=['group', 'key_id', 'candidate'], columns='key', values='val', aggfunc='first').reset_index()
    dataframe = dataframe.drop(columns=['key_id'])

    csv_filename = os.path.join(script_folder, 'test_features_full.csv')
    dataframe.to_csv(csv_filename, index=False) 
