from deepfinder.utils.target_build import TargetBuilder
from copick.impl.filesystem import CopickRootFSSpec
from mrc2omezarr.proc import convert_mrc_to_ngff
import deepfinder.utils.objl as ol
import my_polnet_utils as utils
import numpy as np
import os, shutil

################### Input parameters ###################

# List for How Large the Target Sizes should be
radius_list = [0, 4, 5, 7, 5, 7, 5]

# CoPickPath 
copickFolder = 'copick_pickathon_June2024/valid'

# Voxel Size of Interest
voxelSize = 10 

##########################################################

# Load CoPick root
copickRoot = CopickRootFSSpec.from_file(os.path.join(copickFolder, 'filesystem_overlay_only.json'))

# Load TomoIDs 
tomoIDs = [run.name for run in copickRoot.runs]

# Create Temporary Empty Folder 
os.makedirs('test',exist_ok=True)

# Add Spherical Targets to Mebranes
tbuild = TargetBuilder()

for tomoInd in range(len(tomoIDs)):            

    # Extract TomoID and Associated Run
    tomoID = tomoIDs[tomoInd]
    copickRun = copickRoot.get_run(tomoID)
    
    # Read Particle Coordinates and Write as XML (Necessary?)
    xml_objects = []
    for proteinInd in range(len(copickRun.picks)):
        picks = copickRun.picks[proteinInd]
        classLabel = utils.get_pickable_object_label(copickRoot, picks.pickable_object_name)

        if classLabel == None:
            print('Missing Protein Label: ', picks.pickable_object_name)
            exit()

        for ii in range(len(picks.points)): 
            xml_objects.append({'tomo_name': tomoID,
                               'tomo_idx': tomoInd, 
                               'class_label': classLabel,
                               'x': picks.points[ii].location.x / voxelSize,
                               'y': picks.points[ii].location.y / voxelSize,
                               'z': picks.points[ii].location.z / voxelSize,
                               'phi': 0, 'psi': 0, 'the': 0})

    # Write XML File
    xmlFname =  os.path.join(copickFolder, 'ExperimentRun', tomoID, 'Picks', f'TS_{tomoInd}_objl.xml')
    xmlFname = os.path.join('test',f'TS_{tomoInd}_objl.xml')
    utils.write_xml_file(xml_objects, xmlFname)

    # Create Empty Target Volume
    target_vol = utils.get_target_empty_tomogram(copickRoot)

    # Generate 3D Volume with Spheres
    objl_coords = ol.read_xml(xmlFname)
    target = tbuild.generate_with_spheres(objl_coords, target_vol, radius_list).astype(np.uint8)    

    # Save target with Spheres:
    mrcTargetPath = os.path.join('test',f'TS_{tomoInd}_target.mrc')
    utils.my_write_mrc(target, mrcTargetPath, voxelSize=voxelSize)  

    # Convert MRC Target into ZarrFile
    zarrTargetPath = os.path.join(copickFolder, 'ExperimentRuns', tomoID, f'VoxelSpacing{voxelSize:.3f}', f'spheretargets.zarr')
    convert_mrc_to_ngff(mrcTargetPath, zarrTargetPath, scale_factors=[1])

# Remove the MRC Template 
shutil.rmtree('test')

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
