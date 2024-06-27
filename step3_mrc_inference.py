import deepfinder.utils.copick_tools as tools
from deepfinder.inference import Segment
import deepfinder.utils.objl as objl
import deepfinder.utils.eval as eval
import deepfinder.utils.common as cm
import deepfinder.utils.smap as sm
import my_polnet_utils as utils
import scipy.ndimage as ndimage
from tqdm import tqdm 
import glob, os, json
import numpy as np

# Input parameters:
path_tomo_test = 'path/to/test/dataset' # tomogram to be segmented
path_weights = 'path/to/training_results/net_weights_FINAL.h5' # weights for neural network (obtained from training)
Nclass       = 3  # including background class
patch_size   = 160 # must be multiple of 4

# Output parameter:
pathOutput = 'path/to/evaluate/'

proteins = {'apo': {'name': 'apo-ferritin', 'diameter': 130}, 
            'ribo80S': {'name': 'ribosome', 'diameter': 310} }

############## (Step 1) Initialize segmentation task: ##############

segMetrics = {}
seg  = Segment(Ncl=Nclass, path_weights=path_weights, patch_size=patch_size)
tags = list(proteins)

evalTomos = glob.glob( os.path.join(path_tomo_test, 'TS_*_objl.xml') )
os.makedirs(pathOutput, exist_ok=True)
for coordFile in tqdm(evalTomos): 

    # Load data:
    tomoID = '_'.join(coordFile.split('/')[-1].split('_')[:2])
    tomoPath = os.path.join( path_tomo_test, tomoID + '.mrc')
    tomo = cm.read_array(tomoPath)

    # Segment tomogram:
    scoremaps = seg.launch(tomo)

    # Get labelmap from scoremaps:
    labelmap  = sm.to_labelmap(scoremaps)

    # Save labelmaps (Optional):
    cm.write_array(scoremaps , pathOutput + f'{tomoID}_scoremap.mrc')    
    cm.write_array(labelmap , pathOutput + f'{tomoID}_labelmap.mrc')

    ground_truth_xml = objl.read_xml( coordFile )
    for label in range(2,Nclass):

        print('Finding Predictions for : ', proteins[label-2])
        label_objs, num_features = ndimage.label(labelmap == label)

        # Estimate Coordiantes from CoM for LabelMaps
        deepFinderCoords = []
        for object_num in tqdm(range(1, num_features+1)):
            com = ndimage.center_of_mass(label_objs == object_num)
            swapped_com = (com[2], com[1], com[0])
            deepFinderCoords.append(swapped_com)
        deepFinderCoords = np.array(deepFinderCoords)

        # Extract Ground Truth Labels
        filteredXML = [entry for entry in ground_truth_xml if entry['label'] == label]
        groundTruthCoords = np.array([(entry['x'], entry['y'], entry['z']) for entry in filteredXML])
        
        # Estimate Distance Threshold Based on 1/3 of Particle Diameter
        threshold = np.ceil(  proteins[tags[label - 2]]['diameter'] / (10 * 3) )

        # Remove Double Counted Coordinates
        deepFinderCoords = eval.remove_repeated_picks(deepFinderCoords, threshold)

        # Write the Starfile for Visualization
        tools.write_relion_output(proteins[label-2], None, np.hstack( (deepFinderCoords,  np.zeros((deepFinderCoords.shape[0], 3)))), pathOutput, pixelSize=1) 

        # Compute Metrics
        stats = eval.compute_metrics(groundTruthCoords, deepFinderCoords, threshold)

        # Ensure the key tomoID exists in segMetrics
        segMetrics.setdefault(tomoID, {})

        # Store the Stats in Dictionary 
        segMetrics[tomoID][proteins[label - 2]] = stats     

# Save Segmentation Metrics as a JSON File
with open(os.path.join(pathOutput,'segmentation_metrics.json'), 'w') as json_file:
    json.dump(segMetrics, json_file, indent=4)


