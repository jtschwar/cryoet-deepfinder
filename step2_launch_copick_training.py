from deepfinder.training_copick import Train
import deepfinder.utils.objl as ol
from deepfinder.utils.dataloader import Dataloader
import os

#############################################################################

# Load dataset:
path_train = f'copick_pickathon_June2024/train/filesystem_overlay_only.json'
path_valid = f'copick_pickathon_June2024/valid/filesystem_overlay_only.json'

# Copick Input Parameters
trainVoxelSize        = 10
trainTomoAlg          = 'denoised'

# Input parameters:
Nclass = 8
dim_in = 52  # patch size

#############################################################################

# Create output Path
output_path = f'copick_pickathon_June2024/{trainTomoAlg}_training_results/'
os.makedirs(output_path,exist_ok=True)

# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.path_out         = output_path # output path
trainer.batch_size       = 15
trainer.epochs           = 65
trainer.steps_per_epoch  = 250
trainer.Nvalid           = 20 # steps per validation
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
# Copick Input Parameters
trainer.voxelSize        = trainVoxelSize
trainer.tomoAlg          = trainTomoAlg

# Experimental Weights - [background, membrane, apo, betaAmylase, betaGal, ribo80S, thg, vlp]
trainer.class_weights    = {0:1, 1:3000, 2:6500, 3:70790, 4:800, 5:20225, 6:10300, 7:28000}

# Use following line if you want to resume a previous training session:
trainer.net.load_weights('synthetic_dataset_10A/training_results/net_weights_FINAL.h5')

# Finally, launch the training procedure:
trainer.launch(path_train, path_valid)

