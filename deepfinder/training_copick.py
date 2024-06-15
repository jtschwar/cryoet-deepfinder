# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import numpy as np
import os, time, h5py

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from . import models
from . import losses
from .utils import core
from .utils import common as cm
from .utils import copick_tools as copicktools
from copick.impl.filesystem import CopickRootFSSpec

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Custom callback to save model weights every 10 epochs
class SaveWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, path_out):
        super().__init__()
        self.path_out = path_out
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            self.model.save(self.path_out + f'net_weights_epoch{epoch + 1}.h5')       

# Callback to clear session to free up memory after each epoch
class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()   

def log_images_func(model, validation_data, steps):
    val_data, val_target = [], []
    for step, (x, y) in enumerate(validation_data):
        if step >= steps:
            break
        val_data.append(x.numpy())
        val_target.append(y.numpy())
    
    val_data = np.concatenate(val_data, axis=0)
    val_target = np.concatenate(val_target, axis=0)

    val_predict = model.predict(val_data)
    
    return val_data, val_predict        

class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, writer, log_images_func, validation_data, steps, prefix='train'):
        super().__init__()
        self.writer = writer
        self.log_images_func = log_images_func
        self.validation_data = validation_data
        self.steps = steps
        self.prefix = prefix

    def extract_vol_infrance(self, inVol):
        return np.expand_dims(np.expand_dims(np.array(inVol), axis=0), axis=-1)

    def log_images(self, writer, images, predictions, step, prefix='train'):
        with writer.as_default():
            for i, (image, prediction) in enumerate(zip(images, predictions)):
                
                img_tensor = self.extract_vol_infrance(image)
                pred_tensor = self.extract_vol_infrance(prediction)

                # Ensure tensors are not None
                if img_tensor is not None and pred_tensor is not None:
                    # Check the rank of the tensor
                    if len(img_tensor.shape) == 4:

                        # Select a middle slice for visualization
                        img_tensor = img_tensor[img_tensor.shape[0] // 2, :, :]
                        pred_tensor = pred_tensor[pred_tensor.shape[0] // 2, :, :]

                        # Add batch and channel dimensions
                        img_tensor = np.expand_dims(np.expand_dims(img_tensor, axis=0), axis=-1)
                        pred_tensor = np.expand_dims(np.expand_dims(pred_tensor, axis=0), axis=-1)

                        tf.summary.image(f"{prefix}_image_{i}", img_tensor, step=step)
                        tf.summary.image(f"{prefix}_prediction_{i}", pred_tensor, step=step)

class TrainingPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, validation_steps, path_out, label_list):
        super().__init__()
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.path_out = path_out
        self.label_list = label_list
        self.history = {
            'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [],
            'val_f1': [], 'val_recall': [], 'val_precision': []
        }

    def on_epoch_end(self, epoch, logs=None):
        # Append training metrics
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))

        # Retrieve validation data and predictions
        val_data, val_target = [], []
        for step, (x, y) in enumerate(self.validation_data):
            if step >= self.validation_steps:
                break
            val_data.append(x.numpy())
            val_target.append(y.numpy())

        val_data = np.concatenate(val_data, axis=0)
        val_target = np.concatenate(val_target, axis=0)

        val_predict = np.argmax(self.model.predict(val_data), axis=-1)
        val_targ = np.argmax(val_target, axis=-1)

        # Calculate precision, recall, and F1-score
        scores = precision_recall_fscore_support(val_targ.flatten(), val_predict.flatten(), average=None, labels=self.label_list)

        self.history['val_f1'].append(scores[2])
        self.history['val_recall'].append(scores[1])
        self.history['val_precision'].append(scores[0])

        # Save history and plot
        core.save_history(self.history, self.path_out + 'net_train_history.h5')
        core.plot_history(self.history, self.path_out + 'net_train_history_plot.png')

# TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs
class Train(core.DeepFinder):
    def __init__(self, Ncl, dim_in):
        core.DeepFinder.__init__(self)
        self.path_out = './'
        self.h5_dset_name = 'dataset' # if training set is stored as .h5 file, specify here in which h5 dataset the arrays are stored

        # Check GPU memory limit
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for the GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found.")  

        # Network parameters:
        self.Ncl = Ncl  # Ncl
        self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.net = models.my_model(self.dim_in, self.Ncl)

        self.label_list = []
        for l in range(self.Ncl): self.label_list.append(l) # for precision_recall_fscore_support
                                                            # (else bug if not all labels exist in batch)

        # Training parameters:
        self.batch_size = 25
        self.epochs = 100
        self.steps_per_epoch = 100
        self.steps_per_valid = 10  # number of samples for validation
        self.optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.loss = losses.tversky_loss

        self.flag_direct_read = 1
        self.flag_batch_bootstrap = 0
        self.Lrnd = 13  # random shifts applied when sampling data- and target-patches (in voxels)

        self.class_weight = None
        self.sample_weights = None  # np array same lenght as objl_train

        self.check_attributes()

    def check_attributes(self):
        self.is_positive_int(self.Ncl, 'Ncl')
        self.is_multiple_4_int(self.dim_in, 'dim_in')
        self.is_positive_int(self.batch_size, 'batch_size')
        self.is_positive_int(self.epochs, 'epochs')
        self.is_positive_int(self.steps_per_epoch, 'steps_per_epoch')
        self.is_positive_int(self.steps_per_valid, 'steps_per_valid')
        self.is_int(self.Lrnd, 'Lrnd')

    # This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
    # with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
    # are saved.
    # INPUTS:
    #   path_data     : a list containing the paths to data files (i.e. tomograms)
    #   path_target   : a list containing the paths to target files (i.e. annotated volumes)
    #   objlist_train : list of dictionaries containing information about annotated objects (e.g. class, position)
    #                   In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
    #                   See utils/objl.py for more info about object lists.
    #                   During training, these coordinates are used for guiding the patch sampling procedure.
    #   objlist_valid : same as 'objlist_train', but objects contained in this list are not used for training,
    #                   but for validation. It allows to monitor the training and check for over/under-fitting. Ideally,
    #                   the validation objects should originate from different tomograms than training objects.
    # The network is trained on small 3D patches (i.e. sub-volumes), sampled from the larger tomograms (due to memory
    # limitation). The patch sampling is not realized randomly, but is guided by the macromolecule coordinates contained
    # in so-called object lists (objlist).
    # Concerning the loading of the dataset, two options are possible:
    #    flag_direct_read=0: the whole dataset is loaded into memory
    #    flag_direct_read=1: only the patches are loaded into memory, each time a training batch is generated. This is
    #                        usefull when the dataset is too large to load into memory. However, the transfer speed
    #                        between the data server and the GPU host should be high enough, else the procedure becomes
    #                        very slow.
    # TODO: delete flag_direct_read. Launch should detect if direct_read is desired by checking if input data_list and
    #       target_list contain str (path) or numpy array
    def launch(self, path_train, path_valid):
        """This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
        with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
        are saved.

        Args:
            path_data (list of string): contains paths to data files (i.e. tomograms)
            path_target (list of string): contains paths to target files (i.e. annotated volumes)
            objlist_train (list of dictionaries): contains information about annotated objects (e.g. class, position)
                In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
                See utils/objl.py for more info about object lists.
                During training, these coordinates are used for guiding the patch sampling procedure.
            objlist_valid (list of dictionaries): same as 'objlist_train', but objects contained in this list are not
                used for training, but for validation. It allows to monitor the training and check for over/under-fitting.
                Ideally, the validation objects should originate from different tomograms than training objects.

        Note:
            The function saves following files at regular intervals:
                net_weights_epoch*.h5: contains current network weights

                net_train_history.h5: contains arrays with all metrics per training iteration

                net_train_history_plot.png: plotted metric curves

        """
        self.check_attributes()
        # self.check_arguments(path_data, path_target, objlist_train, objlist_valid)

        # TensorBoard writer
        log_dir = self.path_out + "tensorboard_logs/"
        writer = tf.summary.create_file_writer(log_dir)        
        tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='500,520')

        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)

        # Build network (not in constructor, else not possible to init model with weights from previous train round):
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        self.batch_data = np.zeros((self.batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        self.batch_target = np.zeros((self.batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        train_dataset = self.create_tf_dataset(path_train, self.batch_size, self.dim_in, self.Ncl, self.flag_batch_bootstrap)
        valid_dataset = self.create_tf_dataset(path_valid, self.batch_size, self.dim_in, self.Ncl, self.flag_batch_bootstrap)

        # Callbacks for Save weights and Clear Memory
        clear_memory_callback = ClearMemoryCallback()        
        save_weights_callback = SaveWeightsCallback(self.path_out)
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        plotting_callback = TrainingPlotCallback(validation_data=valid_dataset, validation_steps=self.steps_per_valid, path_out=self.path_out, label_list=self.label_list)

        self.display('Launch training ...')

        # Train the model using model.fit()
        history = self.net.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            class_weight=self.class_weights,
            validation_data=valid_dataset,
            validation_steps=self.steps_per_valid,
            callbacks=[tf_callback, save_weights_callback,plotting_callback,learning_rate_callback],
            verbose=1
        )

        self.net.save(self.path_out+'net_weights_FINAL.h5')

    def check_arguments(self, path_data, path_target, objlist_train, objlist_valid):
        self.is_list(path_data, 'path_data')
        self.is_list(path_target, 'path_target')
        self.are_lists_same_length([path_data, path_target], ['data_list', 'target_list'])
        self.is_list(objlist_train, 'objlist_train')
        self.is_list(objlist_valid, 'objlist_valid')

    def create_tf_dataset(self, copick_path, batch_size, dim_in, Ncl, flag_batch_bootstrap):

        copickRoot = CopickRootFSSpec.from_file(copick_path)
        dataset = tf.data.Dataset.from_generator(
            lambda: self.copick_data_generator(copickRoot, batch_size, dim_in, Ncl, flag_batch_bootstrap),
            output_signature=(
                tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, Ncl), dtype=tf.float32)
            )
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def copick_data_generator(self, copickRoot, batch_size, dim_in, Ncl, flag_batch_bootstrap):
        
        p_in = int(np.floor(dim_in / 2))

        # Load TomoIDs 
        tomoIDs = [run.name for run in copickRoot.runs]        
        tomodim = copicktools.get_copick_tomogram_shape(copickRoot, self.voxelSize)      

        while True:
            
            if flag_batch_bootstrap:
                pool = core.get_copick_boostrap_idx(copickRoot, Nbs=batch_size)
            else:
                pool = range(0, len(objlist))

            idx_list = []
            for i in range(batch_size):
                
                randomBSidx = np.random.choice(pool['bs_idx'])
                idx_list.append(randomBSidx)

                index = np.where( pool['bs_idx'] == randomBSidx )[0][0]
                copickRun = copickRoot.get_run(pool['tomoID_idx'][index])

                sample_data = copicktools.get_copick_tomogram(copickRoot,self.voxelSize,  self.tomoAlg, pool['tomoID_idx'][index])
                sample_target = copicktools.get_copick_tomogram(copickRoot, self.voxelSize, 'spheretargets', pool['tomoID_idx'][index])

                x, y, z = core.get_copick_patch_position(tomodim, p_in, self.Lrnd, self.voxelSize,
                                                         copickRun.picks[pool['protein_idx'][index]].points[pool['pick_idx'][index]] )

                patch_data = sample_data[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
                patch_target = sample_target[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]

                patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)
                patch_target_onehot = to_categorical(patch_target, Ncl)

                self.batch_data[i, :, :, :, 0] = patch_data
                self.batch_target[i] = patch_target_onehot

                if np.random.uniform() < 0.5:
                    self.batch_data[i] = np.rot90(self.batch_data[i], k=2, axes=(0, 2))
                    self.batch_target[i] = np.rot90(self.batch_target[i], k=2, axes=(0, 2))

            yield self.batch_data, self.batch_target



