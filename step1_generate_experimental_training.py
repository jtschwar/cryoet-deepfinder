import my_polnet_utils as utils
import os, glob, shutil
import numpy as np
import starfile

################### Input parameters ###################

tomoPixSize = 10
tomo_path = 'Reza_Tomograms'

# List for How Large the Target Sizes should be
radius_list = [0, 4, 4, 5, 7, 5, 7]

proteinsFullName = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin','virus-like-particle']

tomo_path = 'MiniSlab_Tomograms'
proteinsFullName = ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'virus-like-particle', 'junk']
proteinsFullName = ['junk']

##########################################################

tomoIDs = ['TS_1_1','TS_1_2','TS_1_3','TS_1_5','TS_5_3','TS_1_4','TS_90_1','TS_5_2']
for tomoInd in range(len(tomoIDs)):    

    tomoID = tomoIDs[tomoInd].split('/')[-1]

    # First Start off With Converting Coordinates into XML File (Optional)
    for protein in proteinsFullName:
        path = os.path.join(tomo_path,tomoID,f'{protein}.json')
        coords = utils.read_copick_json(path) 
        utils.write_relion_output(protein, tomoID, coords, tomo_path)

    # Create Membrane Segmentation (Optional)
    # utils.segment_experimental_tomogram()

    Read the Star File and Convert to XML
    xml_objects = []
    for protein in range(len(proteinsFullName)):
        path = os.path.join(tomo_path,tomoID,f'{tomoID}_{proteinsFullName[protein]}.star')
        df = starfile.read(path)
        for ii in range(df.shape[0]):
            xml_objects.append({'tomo_idx': tomoInd, 'class_label': protein + 2, 'x': df['rlnCoordinateX'].iloc[ii], 'y': df['rlnCoordinateY'].iloc[ii], 'z': df['rlnCoordinateZ'].iloc[ii], 'phi': 0, 'psi': 0, 'the': 0 })        
        
    if tomoInd < len(tomoIDs) - 3:      split = 'train'
    elif tomoInd < len(tomoIDs) - 1:    split = 'valid'
    else:                               split = 'test'

    Process ctf-Deconvolved Tomograms
    os.makedirs(os.path.join(tomo_path, 'dctf'),exist_ok=True)
    utils.process_experimental_input(tomo_path, tomoInd, tomoID, 'dctf', split, xml_objects )

    # Process Denoised Tomograms
    os.makedirs(os.path.join(tomo_path, 'Denoised'),exist_ok=True)    
    utils.process_experimental_input(tomo_path, tomoInd, tomoID, 'Denoised', split, xml_objects )

    # Process WBP Tomograms
    os.makedirs(os.path.join(tomo_path, 'WBP'),exist_ok=True)        
    utils.process_experimental_input(tomo_path, tomoInd, tomoID, 'WBP', split, xml_objects )

# Code for Estimating Class Weight:
#   from sklearn.utils import class_weight
#   import numpy as np
#
#   y_train_flat = y_train.flatten()
#   
#   Compute class weights
#   classes = np.unique(y_train_flat)
#   class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train_flat)
#   class_weight_dict = dict(zip(classes, class_weights))
# 
#   first_class_weight = class_weight_dict[0.0]
#   normalized_class_weight_dict = {k: v / first_class_weight for k, v in class_weight_dict.items()}
