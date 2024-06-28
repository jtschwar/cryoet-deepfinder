import os

import numpy as np
from copick.impl.filesystem import CopickRootFSSpec

import deepfinder.utils.copick_tools as tools
from deepfinder.utils.target_build import TargetBuilder

################### Input parameters ###################

# List for How Large the Target Sizes should be (in voxels)
radius_list = [0, 1, 3]

# CoPickPath
copickFolder = "relative/path/to/copick/project"

# Voxel Size of Interest
voxelSize = 10

# Target Zarr Name
targetName = "spheretargets"

# Copick User Id
userID = "train-deepfinder"

##########################################################

# Load CoPick root
copickRoot = CopickRootFSSpec.from_file(os.path.join(copickFolder, "filesystem_overlay_only.json"))

# Load TomoIDs
tomoIDs = [run.name for run in copickRoot.runs]

# Add Spherical Targets to Mebranes
tbuild = TargetBuilder()

# Create Empty Target Volume
target_vol = tools.get_target_empty_tomogram(copickRoot)

# Iterate Through All Runs
for tomoInd in range(len(tomoIDs)):
    # Extract TomoID and Associated Run
    tomoID = tomoIDs[tomoInd]
    copickRun = copickRoot.get_run(tomoID)

    # Read Particle Coordinates and Write as Segmentation
    objl_coords = []
    for proteinInd in range(len(copickRun.picks)):
        picks = copickRun.picks[proteinInd]
        classLabel = tools.get_pickable_object_label(copickRoot, picks.pickable_object_name)

        if classLabel is None:
            print("Missing Protein Label: ", picks.pickable_object_name)
            exit()

        for ii in range(len(picks.points)):
            objl_coords.append(
                {
                    "label": classLabel,
                    "x": picks.points[ii].location.x / voxelSize,
                    "y": picks.points[ii].location.y / voxelSize,
                    "z": picks.points[ii].location.z / voxelSize,
                    "phi": 0,
                    "psi": 0,
                    "the": 0,
                },
            )

    # Reset Target As Empty Array
    # (If Membranes or Organelle Segmentations are Available, add that As Well)
    target_vol[:] = 0

    # Create Target For the Given Coordinates and Sphere Diameters
    target = tbuild.generate_with_spheres(objl_coords, target_vol, radius_list).astype(np.uint8)

    # Write the Target Tomogram as OME Zarr
    tools.write_ome_zarr_segmentation(copickRun, target, voxelSize, targetName, userID)

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
