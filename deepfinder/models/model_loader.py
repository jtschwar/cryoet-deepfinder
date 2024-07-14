from deepfinder.models.res_unet import my_res_unet_model
from deepfinder.models.unet import my_unet_model 
import os

def load_model(dim_in, Ncl, model_name, trained_weights_path):

    # Play with Model to Use
    assert model_name in ['unet', 'res_unet'], "Invalid model name specified. Use 'unet' or 'res_unet'."

    if model_name == 'unet':
        net = my_unet_model(dim_in, Ncl)
    elif model_name == 'res_unet':
        net = my_res_unet_model(dim_in, Ncl)
    else:
        raise ValueError("Invalid model name specified. Valid options {unet, or res_unet}")

    if trained_weights_path is not None:
        if not os.path.exists(trained_weights_path):
            raise FileNotFoundError(f"The specified path for trained weights does not exist: {trained_weights_path}")
        net.load_weights(trained_weights_path)
        print(f'\nTraining {model_name} with {trained_weights_path} Weights\n')
    else:
        print(f'\nTraining {model_name} with Randomly Initialized Weights\n')        

    return net