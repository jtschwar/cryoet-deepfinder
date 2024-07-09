from copick.impl.filesystem import CopickRootFSSpec
import deepfinder.utils.copick_tools as tools
from deepfinder.inference import Segment
import deepfinder.utils.smap as sm
import pycuda.driver as cuda
from mpi4py import MPI
import os

# Initialize MPI (Get Rank and nProc)
comm = MPI.COMM_WORLD; rank = comm.Get_rank(); nProcess = comm.Get_size()

cuda.init()
locGPU = rank % cuda.Device.count() 

####################################################################

# Input parameters:
voxelSize = 10
tomoAlg   = 'denoised'

# Input parameters:
path_tomo_test = 'data_split_study_copick_June2024/' # tomogram to be segmented
path_weights = f'train_results/net_weights_epoch50.h5' # weights for neural network (obtained from training)
Nclass       = 8  # including background class
patch_size   = 160 # must be multiple of 4

# ScoreMap Zarr Name
scoreName = 'score-map'

# LabelMap Zarr Name
labelName = 'label-map'

# Copick UserID
userID = f'predict-unet-deepfinder'

############## (Step 1) Initialize segmentation task: ##############

# Load CoPick root
copickRoot = CopickRootFSSpec.from_file(os.path.join(path_tomo_test, 'filesystem_data_split_study.json'))

seg  = Segment(Ncl=Nclass, path_weights=path_weights, patch_size=patch_size, gpuID = locGPU)

# # Load Evaluate TomoIDs
evalTomos = [run.name for run in copickRoot.runs]

# Create Temporary Empty Folder 
for tomoInd in range(len(evalTomos)): 

    if (tomoInd + 1) % nProcess == rank: 

        # Extract TomoID and Associated Run
        tomoID = evalTomos[tomoInd]

        # Load data:
        tomo = tools.get_copick_tomogram(copickRoot, voxelSize=voxelSize, tomoAlgorithm=tomoAlg, tomoID=tomoID)

        # Segment tomogram:
        scoremaps = seg.launch(tomo[:])

        # Get labelmap from scoremaps:
        labelmap  = sm.to_labelmap(scoremaps)
        copickRun = copickRoot.get_run(tomoID)
        tools.write_ome_zarr_segmentation(copickRun, labelmap, voxelSize, labelName, userID)    

print('Segmentations Complete!')
