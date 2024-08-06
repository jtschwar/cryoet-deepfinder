from copick.impl.filesystem import CopickRootFSSpec
import deepfinder.utils.copick_tools as tools
from deepfinder.inference import Segment
import deepfinder.utils.objl as objl
import deepfinder.utils.smap as sm
import my_polnet_utils as utils
import scipy.ndimage as ndimage
import glob, os, json, zarr
from mpi4py import MPI
from tqdm import tqdm 
import numpy as np
import argparse

# Initialize MPI (Get Rank and nProc)
comm = MPI.COMM_WORLD; rank = comm.Get_rank(); nProcess = comm.Get_size()

####################################################################

# Input parameters:
voxelSize = 10
tomoAlg   = 'denoised'

# Input parameters:
path_tomo_test = '/path/to/copick_project/config.json' # tomogram to be segmented
Nclass       = 8  # including background class

# LabelMap Zarr Name
labelName = 'label-map'

# Copick UserID
userID = f'predict-unet-deepfinder'

# Output parameter:
pathOutput = f'/path/to/copick_project/ExperimentRuns'

proteins = {'apo': {'name': 'apo-ferritin', 'diameter': 130}, 
            'ribo80S': {'name': 'ribosome', 'diameter': 310}}

############## (Step 1) Initialize segmentation task: ##############

tags = list(proteins)            

# Load CoPick root
copickRoot = CopickRootFSSpec.from_file(os.path.join(path_tomo_test))

# Load Evaluate TomoIDs
evalTomos = [run.name for run in copickRoot.runs]

# Create Temporary Empty Folder 
for tomoInd in range(len(evalTomos)): 

    tomoID = evalTomos[tomoInd]
    copickRun = copickRoot.get_run(tomoID)
    labelmap = tools.get_copick_segmentation(copickRun, labelName, userID)[:]

    for label in range(2, Nclass):

        if (label + 1) % nProcess == rank:

            print('Finding Predictions for : ', proteins[tags[label-2]]['name'])
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

                # # Write the Starfile for Visualization
                os.makedirs( 'star_coords/'+tomoID, exist_ok=True)
                tools.write_relion_output(proteins[tags[label-2]]['name'], None, deepFinderCoords, 'star_coords/'+tomoID, pixelSize=1) 

            except:
                deepFinderCoords = np.array([]).reshape(0,6)

            # Save Picks in Copick Format / Directory 
            tools.write_copick_output(proteins[tags[label-2]]['name'], tomoID, deepFinderCoords, pathOutput, pickMethod='deepfinder', sessionID = '0')

print('Extraction of Particle Coordinates Complete!')
