from copick.impl.filesystem import CopickRootFSSpec
import deepfinder.utils.copick_tools as tools
from deepfinder.inference import Segment
import deepfinder.utils.common as cm
import deepfinder.utils.objl as objl
import deepfinder.utils.smap as sm
import my_polnet_utils as utils
import scipy.ndimage as ndimage
from tqdm import tqdm 
import glob, os, json
import numpy as np

voxelSize = 10
tomoAlg   = 'denoised'

# Input parameters:
path_tomo_test = 'copick_pickathon_June2024/test' # tomogram to be segmented
path_weights = f'copick_pickathon_June2024/{tomoAlg}_training_results/net_weights_epoch60.h5' # weights for neural network (obtained from training)
Nclass       = 8  # including background class
patch_size   = 160 # must be multiple of 4

# Output parameter:
pathOutput = f'copick_pickathon_June2024/{tomoAlg}_training_results/evaluate/'

proteins = {'apo': {'name': 'apo-ferritin', 'diameter': 130}, 
            'betaGal': {'name': 'beta-galactosidase', 'diameter': 180}, 
            'ribo80S': {'name': 'ribosome', 'diameter': 310},
            'Thg': {'name': 'thyroglobulin', 'diameter': 290},
            'VLP': {'name': 'virus-like-particle', 'diameter': 285},
            'junk': {'name': 'junk', 'diameter': 200}  }

############## (Step 1) Initialize segmentation task: ##############

# Load CoPick root
copickRoot = CopickRootFSSpec.from_file(os.path.join(path_tomo_test, 'filesystem_overlay_only.json'))

segMetrics = {}
seg  = Segment(Ncl=Nclass, path_weights=path_weights, patch_size=patch_size)

tags = list(proteins)

# Load Evaluate TomoIDs
evalTomos = [run.name for run in copickRoot.runs]

# Create Temporary Empty Folder 
os.makedirs(pathOutput, exist_ok=True)
for tomoInd in range(len(evalTomos)): 

    # Extract TomoID and Associated Run
    tomoID = evalTomos[tomoInd]

    # Load data:
    tomo = tools.get_copick_tomogram(copickRoot, voxelSize=voxelSize, tomoAlgorithm=tomoAlg, tomoID=tomoID)

    # Segment tomogram:
    scoremaps = seg.launch(tomo[:])

    # Get labelmap from scoremaps:
    labelmap  = sm.to_labelmap(scoremaps)

    # Save labelmaps (Optional):
    cm.write_array(scoremaps , pathOutput + f'{tomoID}_scoremap.mrc')    
    cm.write_array(labelmap , pathOutput + f'{tomoID}_labelmap.mrc')

    # Start from Label 2 because label == 0 is background and label == 1 is membrane
    for label in range(2,Nclass):

        print('Finding Predictions for : ', proteins[tags[label-2]]['name'])
        label_objs, num_features = ndimage.label(labelmap == label)

        # Estimate Coordiantes from CoM for LabelMaps
        deepFinderCoords = []
        for object_num in tqdm(range(1, num_features+1)):
            com = ndimage.center_of_mass(label_objs == object_num)
            swapped_com = (com[2], com[1], com[0])
            deepFinderCoords.append(swapped_com)
        deepFinderCoords = np.array(deepFinderCoords)

        # Extract Ground Truth Labels
        copickRun = copickRoot.get_run(tomoID)
        classLabel = tools.get_pickable_object_label(copickRun, proteins[tags[label-2]]['name'])
        groundTruthCoords = tools.get_ground_truth_coordinates(copickRun, voxelSize, classLabel)

        # Estimate Distance Threshold Based on 1/3 of Particle Diameter
        threshold = np.ceil( proteins[tags[label - 2]]['diameter'] / (voxelSize * 3) )

        # Remove Double Counted Coordinates
        deepFinderCoords = utils.remove_repeated_picks(deepFinderCoords, threshold)

        # Write the Starfile for Visualization
        utils.write_relion_output(tags[label-2], None, np.hstack( (deepFinderCoords,  np.zeros((deepFinderCoords.shape[0], 3)))), pathOutput, pixelSize=1) 

        # Compute Metrics
        try: 
            stats = utils.compute_metrics(groundTruthCoords, deepFinderCoords, proteins[tags[label - 2]]['diameter']/ voxelSize)

            # Ensure the key tomoID exists in segMetrics
            segMetrics.setdefault(tomoID, {})

            # Store the Stats in Dictionary 
            segMetrics[tomoID][tags[label-2]] = stats     
        except:
            pass

# Save Segmentation Metrics as a JSON File
with open(os.path.join(pathOutput,'segmentation_metrics.json'), 'w') as json_file:
    json.dump(segMetrics, json_file, indent=4)


