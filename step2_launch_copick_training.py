from deepfinder.training_copick import Train
import deepfinder.utils.common as cm
import os, glob

#############################################################################

# Load dataset:
path_train = f'copick_pickathon_June2024/train/filesystem_overlay_only.json'
path_valid = f'copick_pickathon_June2024/valid/filesystem_overlay_only.json'

# Copick Input Parameters
trainVoxelSize        = 10
trainTomoAlg          = 'denoised'

# Path To Save Training Results
output_path = f'copick_pickathon_June2024/{trainTomoAlg}_training_results/'

# Input parameters:
Nclass = 8
dim_in = 52  # patch size

# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.path_out         = output_path # output path
trainer.batch_size       = 15
trainer.epochs           = 65
trainer.steps_per_epoch  = 250
trainer.Nvalid           = 20 # steps per validation
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True

# Segmentation Target Name And Corresponding UserID
trainer.labelName        = 'spheretargets'
trainer.labelUserID      = 'train-deepfinder'

# Experimental Weights - [background, membrane, apo, betaAmylase, betaGal, ribo80S, thg, vlp]
trainer.class_weights    = {0:1, 1:3000, 2:6500, 3:70790, 4:800, 5:20225, 6:10300, 7:28000}

# Use following line if you want to resume a previous training session:
trainer.net.load_weights('synthetic_dataset_10A/training_results/net_weights_FINAL.h5')

#############################################################################

# Create output Path
os.makedirs(output_path,exist_ok=True)

# Copick Input Parameters
trainer.voxelSize        = trainVoxelSize
trainer.tomoAlg          = trainTomoAlg

# Finally, launch the training procedure:

# Option 1: 
    # trainer.launch(path_train, path_valid)
# Option 2:
# Split the Entire Copick Project into Train / Validation / Test
tomoIDs = glob.glob('copick_path/ExperimentRuns/TS_*')
tomoIDs = [path.split('/')[-1] for path in tomoIDs]
(trainList, validationList, testList) = cm.split_datasets(tomoIDs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, savePath = output_path)
trainer.validTomoIDs     = validationList
trainer.trainTomoIDs     = trainList

# Train
trainer.launch(path_train)
