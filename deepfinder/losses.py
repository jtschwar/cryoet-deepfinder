# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import tensorflow as tf
from tensorflow.keras import backend as K

# had to replace sometimes K by tf, because else: TypeError: An op outside of the function building code is being passed
#     a "Graph" tensor. It is possible to have Graph tensors
#     leak out of the function building context by including a
#     tf.init_scope in your function building code.
# Reason was: So the main issue here is that custom loss function is returning a Symbolic KerasTensor and not a Tensor.
#     And this is happening because inputs to the custom loss function are in Symbolic KerasTensor form.
#     ref: https://github.com/tensorflow/tensorflow/issues/43650

# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = tf.cast(tf.shape(y_true)[-1], 'float32')
    return Ncl - T

def focal_loss(y_true, y_pred):
    r"""Compute focal loss for predictions.

        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = y_true.

    Args:
     y_pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y_true: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(y_pred)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # y_true > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(y_true > zeros, y_true - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # y_true > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)
