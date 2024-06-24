from deepfinder.training_mrc import Train
import deepfinder.utils.objl as ol
from deepfinder.utils.dataloader import Dataloader
import os

# Load dataset:
voxSize = 10
path_dset = f'Reza_Tomograms/dctf/'
output_path = f'Reza_Tomograms/dctf/training_results/'
path_data, path_target, objl_train, objl_valid = Dataloader()(path_dset)
os.makedirs(output_path,exist_ok=True)

# Input parameters:
Nclass = 8
dim_in = 52  # patch size

# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.path_out         = output_path # output path
trainer.batch_size       = 15
trainer.epochs           = 100
trainer.steps_per_epoch  = 250
trainer.Nvalid           = 20 # steps per validation
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
# trainer.class_weights    = None # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0

# Synthetic Weights
# trainer.class_weights    = {0:1, 1:150, 2:800, 3:2000, 4:2000, 5:100, 6:1000, 7:500}

# Experimental Weights - [background, membrane, apo, betaAmylase, betaGal, ribo80S, thg, vlp]
trainer.class_weights    = {0:1, 1:150, 2:5600, 3:17000, 4:8000, 5:700, 6:70000, 7:17000}

# Use following line if you want to resume a previous training session:
trainer.net.load_weights('small_training_results_10A/net_weights_FINAL.h5')

# Finally, launch the training procedure:
trainer.launch(path_data, path_target, objl_train, objl_valid)

