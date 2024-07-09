from copick.impl.filesystem import CopickRootFSSpec
import deepfinder.utils.copick_tools as tools
from deepfinder.inference import Segment
import deepfinder.utils.smap as sm
import my_polnet_utils as utils
import scipy.ndimage as ndimage
from tqdm import tqdm 
import numpy as np
import os, glob

####################################################################

voxelSize = 10
tomoAlg   = 'denoised'

# Input parameters:
path_tomo_test = 'relative/path/to/copick/project/' # tomogram to be segmented
path_weights = f'relative/path/to/training_results/net_weights_FINAL.h5' # weights for neural network (obtained from training)
Nclass       = 3  # including background class
patch_size   = 160 # must be multiple of 4

# ScoreMap Zarr Name
scoreName = 'score-map'

# LabelMap Zarr Name
labelName = 'label-map'

# Copick UserID
userID = 'predict-resunet-deepfinder'

# Output parameter:
pathOutput = f'relative/path/to/copick/project/'

proteins = {'apo': {'name': 'apo-ferritin', 'diameter': 130}, 
            'ribo80S': {'name': 'ribosome', 'diameter': 310} }

relionWriteDirectory = None

############## (Step 1) Initialize segmentation task: ##############

# Load CoPick root
copickRoot = CopickRootFSSpec.from_file(os.path.join(path_tomo_test, 'filesystem_overlay_only.json'))

seg  = Segment(Ncl=Nclass, path_weights=path_weights, patch_size=patch_size)
tags = list(proteins)            

# Load Evaluate TomoIDs
evalTomos = [run.name for run in copickRoot.runs]

# Create Temporary Empty Folder 
for tomoInd in range(len(evalTomos)): 

    # Extract TomoID and Associated Run
    tomoID = evalTomos[tomoInd]
    copickRun = copickRoot.get_run(tomoID)

    # Load data:
    tomo = tools.get_copick_tomogram(copickRoot, voxelSize=voxelSize, tomoAlgorithm=tomoAlg, tomoID=tomoID)

    # Segment tomogram:
    scoremaps = seg.launch(tomo[:])
    # tools.write_ome_zarr_segmentation(copickRun, scoremaps, voxelSize, scoreName, userID)

    # Get labelmap from scoremaps:
    labelmap  = sm.to_labelmap(scoremaps)
    tools.write_ome_zarr_segmentation(copickRun, labelmap, voxelSize, labelName, userID)    

    for label in range(2,Nclass):

        proteinName = proteins[tags[label-2]]['name']
        print(f'Finding Predictions for : {proteinName}')
        label_objs, num_features = ndimage.label(labelmap == label)

        # Estimate Coordiantes from CoM for LabelMaps
        deepFinderCoords = []
        for object_num in tqdm(range(1, num_features+1)):
            com = ndimage.center_of_mass(label_objs == object_num)
            swapped_com = (com[2], com[1], com[0])
            deepFinderCoords.append(swapped_com)
        deepFinderCoords = np.array(deepFinderCoords)

        # Estimate Distance Threshold Based on 1/2 of Particle Diameter
        threshold = np.ceil(  proteins[tags[label-2]]['diameter'] / (voxelSize * 3) )

        try: 
            # Remove Double Counted Coordinates
            deepFinderCoords = utils.remove_repeated_picks(deepFinderCoords, threshold)

            # Convert from Voxel to Physical Units
            deepFinderCoords *= voxelSize

            # Append Euler Angles to Coordinates [ Expand Dimensions from Nx3 -> Nx6 ]
            deepFinderCoords = np.concatenate((deepFinderCoords, np.zeros(deepFinderCoords.shape)),axis=1)

            # Write the Starfile for Artiax / ChimeraX Visualization
            if relionWriteDirectory is not None:
                os.makedirs( os.path.join(relionWriteDirectory,tomoID), exist_ok=True)
                tools.write_relion_output(proteinName, None, deepFinderCoords,
                                          os.path.join(relionWriteDirectory,tomoID), pixelSize=1) 
        
        except Exception as e:

            print(f"Error processing label {proteinName} in tomo {tomoID}: {e}")
            deepFinderCoords = np.array([]).reshape(0,6)

        # Save Picks in Copick Format / Directory 
        tools.write_copick_output(proteinName, tomoID, deepFinderCoords, pathOutput, sessionID = '23')
