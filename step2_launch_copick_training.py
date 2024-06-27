from deepfinder.training_copick import Train
import deepfinder.utils.common as cm
import os, glob

#############################################################################

# Load dataset:
path_train = f'relative/path/to/copick/project/training'
path_valid = f'relative/path/to/copick/project/validation'

# Copick Input Parameters
trainVoxelSize        = 10
trainTomoAlg          = 'denoised'

# Path To Save Training Results
output_path = f'relative/path/to/copick/project/{trainTomoAlg}_training_results/'

# Input parameters:
Nclass = 3
dim_in = 52  # patch size 

# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.path_out         = output_path # output path
trainer.batch_size       = 5
trainer.epochs           = 20
trainer.steps_per_epoch  = 250
trainer.Nvalid           = 20 # steps per validation
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True

# Segmentation Target Name And Corresponding UserID
trainer.labelName        = 'spheretargets'
trainer.labelUserID      = 'train-deepfinder'

# keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0
trainer.class_weights = None

# Use following line if you want to resume a previous training session:
# trainer.net.load_weights('relative/path/to/copick/project/training_results/net_weights_FINAL.h5')

# A Certain Number of Tomograms are Loaded Prior to Training (sample_size)
# And picks from these tomograms are trained for a specified number of epochs (NsubEpoch)
trainer.NsubEpoch          = 10
trainer.sample_size        = 15 

#############################################################################

# Create output Path
os.makedirs(output_path,exist_ok=True)

# Copick Input Parameters
trainer.voxelSize        = trainVoxelSize
trainer.tomoAlg          = trainTomoAlg

# Finally, launch the training procedure:

# Option 1:
# Split the Entire Copick Project into Train / Validation / Test
tomoIDs = glob.glob('copick_path/ExperimentRuns/TS_*')
tomoIDs = [path.split('/')[-1] for path in tomoIDs]
(trainList, validationList, testList) = cm.split_datasets(tomoIDs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, savePath = output_path)

# Pass the Run IDs to the Training Class
trainer.validTomoIDs     = validationList
trainer.trainTomoIDs     = trainList

# Train
trainer.launch(path_train, path_valid)

# Option 2: 
    # The Data is Already Split into two Copick Projects
    # trainer.launch(path_train, path_valid)
