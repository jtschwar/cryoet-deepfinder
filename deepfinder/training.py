import h5py
import numpy as np
import time

from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.metrics import precision_recall_fscore_support

from . import models
from . import losses
# import models
# import losses

from .utils import core
from .utils import common as cm
# import .utils.core as core
# import .utils.common as cm


class TargetBuilder(core.DeepFinder):
    def __init__(self):
        core.DeepFinder.__init__(self)

    # Generates segmentation targets from object list. Here macromolecules are annotated with their shape.
    # INPUTS
    #   objl: list of dictionaries. Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   ref_list: list of binary 3D arrays (expected to be cubic). These reference arrays contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
    #             The references order in list should correspond to the class label
    #             For ex: 1st element of list -> reference of class 1
    #                     2nd element of list -> reference of class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_shapes(self, objl, target_array, ref_list):
        N = len(objl)
        dim = target_array.shape
        for p in range(len(objl)):
            self.display('Annotating object ' + str(p + 1) + ' / ' + str(N) + ' ...')
            lbl = int(objl[p]['label'])
            x = int(objl[p]['x'])
            y = int(objl[p]['y'])
            z = int(objl[p]['z'])
            phi = objl[p]['phi']
            psi = objl[p]['psi']
            the = objl[p]['the']

            ref = ref_list[lbl - 1]
            centeroffset = np.int(np.floor(ref.shape[0] / 2)) # here we expect ref to be cubic

            # Rotate ref:
            if phi!=None and psi!=None and the!=None:
                ref = cm.rotate_array(ref, (phi, psi, the))
                ref = np.int8(np.round(ref))

            # Get the coordinates of object voxels in target_array
            obj_voxels = np.nonzero(ref == 1)
            x_vox = obj_voxels[2] + x - centeroffset +1
            y_vox = obj_voxels[1] + y - centeroffset +1
            z_vox = obj_voxels[0] + z - centeroffset +1

            for idx in range(x_vox.size):
                xx = x_vox[idx]
                yy = y_vox[idx]
                zz = z_vox[idx]
                if xx >= 0 and xx < dim[2] and yy >= 0 and yy < dim[1] and zz >= 0 and zz < dim[0]:  # if in tomo bounds
                    target_array[zz, yy, xx] = lbl

        return np.int8(target_array)

    # Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
    # This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
    # On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'
    # INPUTS
    #   objl: list of dictionaries. Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   radius_list: list of sphere radii (in voxels).
    #             The radii order in list should correspond to the class label
    #             For ex: 1st element of list -> sphere radius for class 1
    #                     2nd element of list -> sphere radius for class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_spheres(self, objl, target_array, radius_list):
        Rmax = max(radius_list)
        dim = [2*Rmax, 2*Rmax, 2*Rmax]
        ref_list = []
        for idx in range(len(radius_list)):
            ref_list.append(cm.create_sphere(dim, radius_list[idx]))
        target_array = self.generate_with_shapes(objl, target_array, ref_list)
        return target_array


class Train(core.DeepFinder):
    def __init__(self, Ncl):
        core.DeepFinder.__init__(self)
        self.path_out = './'
        self.h5_dset_name = 'dataset' # if training set is stored as .h5 file, specify here in which h5 dataset the arrays are stored

        # Network parameters:
        self.Ncl = Ncl  # Ncl
        self.dim_in = 56  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.net = models.my_model(self.dim_in, self.Ncl)

        # Training parameters:
        self.batch_size = 25
        self.epochs = 100
        self.steps_per_epoch = 100
        self.Nvalid = 100  # number of samples for validation TODO: is not implemented !!!
        self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.flag_direct_read = 1
        self.flag_batch_bootstrap = 0
        self.Lrnd = 13  # random shifts applied when sampling data- and target-patches (in voxels)

        # Build network:
        self.net.compile(optimizer=self.optimizer, loss=losses.tversky_loss, metrics=['accuracy'])

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
    def launch(self, path_data, path_target, objlist_train, objlist_valid):
        # Load whole dataset:
        if self.flag_direct_read == False:
            self.display('Loading dataset ...')
            data_list, target_list = core.load_dataset(path_data, path_target, self.h5_dset_name)

        self.display('Launch training ...')

        # Declare lists for storing training statistics:
        hist_loss_train = []
        hist_acc_train  = []
        hist_loss_valid = []
        hist_acc_valid  = []
        hist_f1         = []
        hist_recall     = []
        hist_precision  = []
        process_time    = []

        # Training loop:
        for e in range(self.epochs):
            # TRAINING:
            start = time.time()
            for it in range(self.steps_per_epoch):
                if self.flag_direct_read:
                    batch_data, batch_target = self.generate_batch_direct_read(path_data, path_target, self.batch_size,
                                                                               objlist_train)
                else:
                    batch_data, batch_target = self.generate_batch_from_array(data_list, target_list, self.batch_size,
                                                                              objlist_train)
                loss_train = self.net.train_on_batch(batch_data, batch_target)

                self.display('epoch %d/%d - it %d/%d - loss: %0.3f - acc: %0.3f' % (
                e + 1, self.epochs, it + 1, self.steps_per_epoch, loss_train[0], loss_train[1]))
            hist_loss_train.append(loss_train[0])
            hist_acc_train.append(loss_train[1])

            # VALIDATION (compute statistics to monitor training):
            if self.flag_direct_read:
                batch_data_valid, batch_target_valid = self.generate_batch_direct_read(path_data, path_target,
                                                                                       self.batch_size, objlist_valid)
            else:
                batch_data_valid, batch_target_valid = self.generate_batch_from_array(data_list, target_list,
                                                                                      self.batch_size, objlist_valid)
            loss_val = self.net.evaluate(batch_data_valid, batch_target_valid, verbose=0)

            batch_pred = self.net.predict(batch_data_valid)
            scores = precision_recall_fscore_support(batch_target_valid.argmax(axis=-1).flatten(),
                                                     batch_pred.argmax(axis=-1).flatten(), average=None)

            hist_loss_valid.append(loss_val[0])
            hist_acc_valid.append(loss_val[1])
            hist_f1.append(scores[2])
            hist_recall.append(scores[1])
            hist_precision.append(scores[0])

            end = time.time()
            process_time.append(end - start)
            self.display('-------------------------------------------------------------')
            self.display('EPOCH %d/%d - valid loss: %0.3f - valid acc: %0.3f - %0.2fsec' % (
            e + 1, self.epochs, loss_val[0], loss_val[1], end - start))
            self.display('=============================================================')

            # Save and plot training history:
            history = {'loss': hist_loss_train, 'acc': hist_acc_train, 'val_loss': hist_loss_valid,
                       'val_acc': hist_acc_valid, 'val_f1': hist_f1, 'val_recall': hist_recall,
                       'val_precision': hist_precision}
            core.save_history(history, self.path_out+'net_train_history.h5')
            core.plot_history(history, self.path_out+'net_train_history_plot.png')

            if (e + 1) % 10 == 0:  # save weights every 10 epochs
                self.net.save(self.path_out+'net_weights_epoch' + str(e + 1) + '.h5')

        self.display("Model took %0.2f seconds to train" % np.sum(process_time))
        self.net.save(self.path_out+'net_weights_FINAL.h5')

    # Generates batches for training and validation. In this version, the dataset is not loaded into memory. Only the
    # current batch content is loaded into memory, which is useful when memory is limited.
    # Is called when self.flag_direct_read=True
    # !! For now only works for h5 files !!
    # Batches are generated as follows:
    #   - The positions at which patches are sampled are determined by the coordinates contained in the object list.
    #   - Two data augmentation techniques are applied:
    #       .. To gain invariance to translations, small random shifts are added to the positions.
    #       .. 180 degree rotation around tilt axis (this way missing wedge orientation remains the same).
    #   - Also, bootstrap (i.e. re-sampling) can be applied so that we have an equal amount of each macromolecule in
    #     each batch. This is useful when a class is under-represented. Is applied when self.flag_batch_bootstrap=True
    # INPUTS:
    #   path_data: list of strings '/path/to/tomo.h5'
    #   path_target: list of strings '/path/to/target.h5'
    #   batch_size: int
    #   objlist: list of dictionnaries
    # OUTPUT:
    #   batch_data: numpy array [batch_idx, z, y, x, channel] in our case only 1 channel
    #   batch_target: numpy array [batch_idx, z, y, x, class_idx] is one-hot encoded
    def generate_batch_direct_read(self, path_data, path_target, batch_size, objlist=None):
        p_in = np.int(np.floor(self.dim_in / 2))

        batch_data = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        # The batch is generated by randomly sampling data patches.
        if self.flag_batch_bootstrap:  # choose from bootstrapped objlist
            pool = core.get_bootstrap_idx(objlist, Nbs=batch_size)
        else:  # choose from whole objlist
            pool = range(0, len(objlist))

        for i in range(batch_size):
            # Choose random object in training set:
            index = np.random.choice(pool)

            tomoID = int(objlist[index]['tomo_idx'])

            h5file = h5py.File(path_data[tomoID], 'r')
            tomodim = h5file['dataset'].shape  # get tomo dimensions without loading the array
            h5file.close()

            x, y, z = core.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)

            # Load data and target patches:
            h5file = h5py.File(path_data[tomoID], 'r')
            patch_data = h5file['dataset'][z - p_in:z + p_in, y - p_in:y + p_in, x - p_in:x + p_in]
            h5file.close()

            h5file = h5py.File(path_target[tomoID], 'r')
            patch_target = h5file['dataset'][z - p_in:z + p_in, y - p_in:y + p_in, x - p_in:x + p_in]
            h5file.close()

            # Process the patches in order to be used by network:
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
            patch_target_onehot = to_categorical(patch_target, self.Ncl)

            # Store into batch array:
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i] = patch_target_onehot

            # Data augmentation (180degree rotation around tilt axis):
            if np.random.uniform() < 0.5:
                batch_data[i] = np.rot90(batch_data[i], k=2, axes=(0, 2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0, 2))

        return batch_data, batch_target

    # Generates batches for training and validation. In this version, the whole dataset has already been loaded into
    # memory, and batch is sampled from there. Apart from that does the same as above.
    # Is called when self.flag_direct_read=False
    # INPUTS:
    #   data: list of numpy arrays
    #   target: list of numpy arrays
    #   batch_size: int
    #   objlist: list of dictionnaries
    # OUTPUT:
    #   batch_data: numpy array [batch_idx, z, y, x, channel] in our case only 1 channel
    #   batch_target: numpy array [batch_idx, z, y, x, class_idx] is one-hot encoded
    def generate_batch_from_array(self, data, target, batch_size, objlist=None):
        p_in = np.int(np.floor(self.dim_in / 2))

        batch_data = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        # The batch is generated by randomly sampling data patches.
        if self.flag_batch_bootstrap:  # choose from bootstrapped objlist
            pool = core.get_bootstrap_idx(objlist, Nbs=batch_size)
        else:  # choose from whole objlist
            pool = range(0, len(objlist))

        for i in range(batch_size):
            # choose random sample in training set:
            index = np.random.choice(pool)

            tomoID = int(objlist[index]['tomo_idx'])

            tomodim = data[tomoID].shape

            sample_data = data[tomoID]
            sample_target = target[tomoID]

            dim = sample_data.shape

            # Get patch position:
            x, y, z = core.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)

            # Get patch:
            patch_data  = sample_data[  z - p_in:z + p_in, y - p_in:y + p_in, x - p_in:x + p_in]
            patch_batch = sample_target[z - p_in:z + p_in, y - p_in:y + p_in, x - p_in:x + p_in]

            # Process the patches in order to be used by network:
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
            patch_batch_onehot = to_categorical(patch_batch, self.Ncl)

            # Store into batch array:
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i] = patch_batch_onehot

            # Data augmentation (180degree rotation around tilt axis):
            if np.random.uniform() < 0.5:
                batch_data[i] = np.rot90(batch_data[i], k=2, axes=(0, 2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0, 2))

        return batch_data, batch_target