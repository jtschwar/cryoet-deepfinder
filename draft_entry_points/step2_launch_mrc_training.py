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
Nclass = 3
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

# keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0
# trainer.class_weights    = None 

# Use following line if you want to resume a previous training session:
trainer.net.load_weights('relative/path/to/copick/project/{trainTomoAlg}_training_results')

# Finally, launch the training procedure:
trainer.launch(path_data, path_target, objl_train, objl_valid)

